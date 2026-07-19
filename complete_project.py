from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


@dataclass
class ExperimentConfig:
    # "quick" verifies the full pipeline; "standard" is a balanced run;
    # "report" is the recommended final-report run.
    profile: str = "report"
    seed: int = 42
    data_dir: str = "./data"
    output_dir: str = "./vae_ddpm_fashionmnist_results"
    resume_if_available: bool = True

    num_workers: int = 2
    classifier_batch_size: int = 256
    vae_batch_size: int = 256
    ddpm_batch_size: int = 128

    classifier_lr: float = 1e-3
    vae_lr: float = 2e-3
    ddpm_lr: float = 2e-4
    weight_decay: float = 1e-5
    vae_beta: float = 1.0
    latent_dim: int = 32
    ema_decay: float = 0.995
    grad_clip_norm: float = 1.0

    # Filled by apply_profile().
    classifier_epochs: int = 12
    vae_epochs: int = 25
    ddpm_epochs: int = 35
    diffusion_steps: int = 100
    eval_samples: int = 5000
    kid_repeats: int = 20
    kid_subset_size: int = 1000
    nn_generated_samples: int = 128
    nn_reference_samples: int = 10000
    trajectory_frames: int = 10

    def apply_profile(self) -> "ExperimentConfig":
        profiles = {
            "quick": {
                "classifier_epochs": 2,
                "vae_epochs": 2,
                "ddpm_epochs": 2,
                "diffusion_steps": 50,
                "eval_samples": 500,
                "kid_repeats": 5,
                "kid_subset_size": 300,
                "nn_generated_samples": 32,
                "nn_reference_samples": 2000,
                "trajectory_frames": 8,
            },
            "standard": {
                "classifier_epochs": 8,
                "vae_epochs": 15,
                "ddpm_epochs": 20,
                "diffusion_steps": 100,
                "eval_samples": 2000,
                "kid_repeats": 10,
                "kid_subset_size": 750,
                "nn_generated_samples": 96,
                "nn_reference_samples": 7500,
                "trajectory_frames": 10,
            },
            "report": {
                "classifier_epochs": 12,
                "vae_epochs": 25,
                "ddpm_epochs": 35,
                "diffusion_steps": 100,
                "eval_samples": 5000,
                "kid_repeats": 20,
                "kid_subset_size": 1000,
                "nn_generated_samples": 128,
                "nn_reference_samples": 10000,
                "trajectory_frames": 10,
            },
        }
        if self.profile not in profiles:
            raise ValueError(f"Unknown profile: {self.profile}. Use quick, standard, or report.")
        for key, value in profiles[self.profile].items():
            setattr(self, key, value)
        return self


CFG = ExperimentConfig(
    profile=os.environ.get("EXPERIMENT_PROFILE", "report").lower()
).apply_profile()

# Throughput-oriented defaults for modern 16 GB GPUs (including RTX 5070 Ti).
# Environment overrides make it easy to dial a batch down if this is run on a
# smaller GPU later without editing the source again.
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 14 * 1024**3:
    CFG.num_workers = int(os.environ.get("NUM_WORKERS", "4"))
    CFG.classifier_batch_size = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "1024"))
    CFG.vae_batch_size = int(os.environ.get("VAE_BATCH_SIZE", "1024"))
    CFG.ddpm_batch_size = int(os.environ.get("DDPM_BATCH_SIZE", "1024"))

OUTPUT_DIR = Path(CFG.output_dir)
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"
SAMPLE_DIR = OUTPUT_DIR / "samples"
for directory in (OUTPUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR, LOG_DIR, SAMPLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

if USE_AMP:
    # TensorFloat-32 uses the GPU's tensor cores for float32 matrix operations;
    # training convolutions still use the existing FP16 autocast path below.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode improves repeatability, but may be slower.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)

if DEVICE.type == "cpu" and CFG.profile == "report":
    warnings.warn("The report profile is compute-intensive on CPU. Use a Colab GPU or change CFG.profile to quick.")

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Profile: {CFG.profile}")
print(json.dumps(asdict(CFG), indent=2))

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(
    root=CFG.data_dir, train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=CFG.data_dir, train=False, download=True, transform=transform
)

loader_kwargs = {
    "num_workers": CFG.num_workers,
    "pin_memory": DEVICE.type == "cuda",
    "persistent_workers": CFG.num_workers > 0,
}
if CFG.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4

classifier_train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.classifier_batch_size,
    shuffle=True,
    drop_last=False,
    **loader_kwargs,
)
classifier_test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.classifier_batch_size,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)
vae_train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.vae_batch_size,
    shuffle=True,
    drop_last=False,
    **loader_kwargs,
)
ddpm_train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.ddpm_batch_size,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)

print(f"Training images: {len(train_dataset):,}")
print(f"Test images: {len(test_dataset):,}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def amp_context():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def gpu_peak_memory_mb() -> float:
    if DEVICE.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_gpu_peak_memory() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def save_json(data: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value)}")
    path.write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")


def save_tensor_grid(
    images: torch.Tensor,
    path: Path,
    title: str,
    nrow: int = 8,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> None:
    images = images.detach().cpu().float().clamp(*value_range)
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    array = grid.squeeze(0).numpy() if grid.shape[0] == 1 else np.transpose(grid.numpy(), (1, 2, 0))
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(array, cmap="gray" if grid.shape[0] == 1 else None)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_pair_grid(
    first: torch.Tensor,
    second: torch.Tensor,
    path: Path,
    first_label: str,
    second_label: str,
    max_pairs: int = 16,
) -> None:
    count = min(max_pairs, len(first), len(second))
    paired = torch.stack([first[:count], second[:count]], dim=1).reshape(-1, 1, 28, 28)
    grid = make_grid(paired, nrow=2, padding=2)
    array = grid.squeeze(0).numpy() if grid.shape[0] == 1 else np.transpose(grid.numpy(), (1, 2, 0))
    figure = plt.figure(figsize=(5, max(5, count * 1.25)))
    plt.imshow(array, cmap="gray" if grid.shape[0] == 1 else None)
    plt.axis("off")
    plt.title(f"Left: {first_label}   |   Right: {second_label}")
    plt.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def model_state_path(name: str) -> Path:
    signature = f"{CFG.profile}_z{CFG.latent_dim}_T{CFG.diffusion_steps}"
    return MODEL_DIR / f"{name}_{signature}.pt"


def save_checkpoint(model: nn.Module, path: Path, extra: Optional[dict] = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=DEVICE)
    model.load_state_dict(payload["state_dict"])
    return payload

class FashionClassifier(nn.Module):
    """CNN classifier used both for recognizability and feature-space metrics."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 14 x 14
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 7 x 7
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.2860) / 0.3530

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        return self.embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


@torch.inference_mode()
def evaluate_classifier(
    model: FashionClassifier,
    loader: DataLoader,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        predictions = logits.argmax(dim=1)
        total_loss += loss.item()
        total_correct += (predictions == labels).sum().item()
        total_count += labels.numel()
        for target, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion[target, prediction] += 1

    return total_loss / total_count, total_correct / total_count, confusion


def train_classifier(model: FashionClassifier) -> Tuple[dict, float, float]:
    checkpoint_path = model_state_path("fashion_classifier")
    payload = load_checkpoint(model, checkpoint_path) if CFG.resume_if_available else None
    if payload is not None:
        print("Loaded existing classifier checkpoint.")
        return payload.get("history", {}), payload.get("training_seconds", 0.0), payload.get("peak_gpu_mb", 0.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.classifier_lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.classifier_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    history = {"train_loss": [], "test_loss": [], "test_accuracy": [], "lr": []}
    best_accuracy = -1.0
    best_state = None

    reset_gpu_peak_memory()
    start = time.perf_counter()

    for epoch in range(1, CFG.classifier_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(classifier_train_loader, desc=f"Classifier {epoch}/{CFG.classifier_epochs}")
        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        test_loss, test_accuracy, _ = evaluate_classifier(model, classifier_test_loader)
        train_loss = running_loss / seen
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, "
            f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.2%}"
        )

    training_seconds = time.perf_counter() - start
    peak_gpu_mb = gpu_peak_memory_mb()
    model.load_state_dict(best_state)
    save_checkpoint(
        model,
        checkpoint_path,
        {
            "history": history,
            "training_seconds": training_seconds,
            "peak_gpu_mb": peak_gpu_mb,
            "best_accuracy": best_accuracy,
        },
    )
    return history, training_seconds, peak_gpu_mb


def plot_classifier_history(history: dict) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["test_loss"], marker="o", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Fashion-MNIST classifier loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(FIG_DIR / "classifier_loss.png")
    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, np.asarray(history["test_accuracy"]) * 100, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fashion-MNIST classifier accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "classifier_accuracy.png")
    plt.close(figure)


def plot_confusion_matrix(confusion: np.ndarray) -> None:
    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    normalized = confusion / row_sums
    figure = plt.figure(figsize=(9, 8))
    image = plt.imshow(normalized, interpolation="nearest")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(10), FASHION_MNIST_CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), FASHION_MNIST_CLASSES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Classifier normalized confusion matrix")
    plt.tight_layout()
    figure.savefig(FIG_DIR / "classifier_confusion_matrix.png")
    plt.close(figure)

class ConvolutionalVAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 14 x 14
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 7 x 7
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.SiLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14 x 14
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 28 x 28
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decoder_input(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return logits, mu, logvar


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, target, reduction="sum"
    ) / target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total = reconstruction + CFG.vae_beta * kl
    return total, reconstruction, kl


@torch.inference_mode()
def evaluate_vae(model: ConvolutionalVAE) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
    count = 0
    for images, _ in classifier_test_loader:
        images = images.to(DEVICE, non_blocking=True)
        logits, mu, logvar = model(images)
        total, reconstruction, kl = vae_loss(logits, images, mu, logvar)
        batch_size = images.size(0)
        totals["total"] += total.item() * batch_size
        totals["reconstruction"] += reconstruction.item() * batch_size
        totals["kl"] += kl.item() * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def train_vae(model: ConvolutionalVAE) -> Tuple[dict, float, float]:
    checkpoint_path = model_state_path("vae")
    payload = load_checkpoint(model, checkpoint_path) if CFG.resume_if_available else None
    if payload is not None:
        print("Loaded existing VAE checkpoint.")
        return payload.get("history", {}), payload.get("training_seconds", 0.0), payload.get("peak_gpu_mb", 0.0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.vae_lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.vae_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    history = {"total": [], "reconstruction": [], "kl": [], "test_total": [], "lr": []}

    reset_gpu_peak_memory()
    start = time.perf_counter()

    for epoch in range(1, CFG.vae_epochs + 1):
        model.train()
        running = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0}
        seen = 0

        progress = tqdm(vae_train_loader, desc=f"VAE {epoch}/{CFG.vae_epochs}")
        for images, _ in progress:
            images = images.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                logits, mu, logvar = model(images)
                total, reconstruction, kl = vae_loss(logits, images, mu, logvar)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running["total"] += total.item() * batch_size
            running["reconstruction"] += reconstruction.item() * batch_size
            running["kl"] += kl.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{total.item():.2f}", kl=f"{kl.item():.2f}")

        validation = evaluate_vae(model)
        history["total"].append(running["total"] / seen)
        history["reconstruction"].append(running["reconstruction"] / seen)
        history["kl"].append(running["kl"] / seen)
        history["test_total"].append(validation["total"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        print(
            f"Epoch {epoch:02d}: total={history['total'][-1]:.2f}, "
            f"recon={history['reconstruction'][-1]:.2f}, "
            f"KL={history['kl'][-1]:.2f}, test={validation['total']:.2f}"
        )

    training_seconds = time.perf_counter() - start
    peak_gpu_mb = gpu_peak_memory_mb()
    save_checkpoint(
        model,
        checkpoint_path,
        {
            "history": history,
            "training_seconds": training_seconds,
            "peak_gpu_mb": peak_gpu_mb,
        },
    )
    return history, training_seconds, peak_gpu_mb


@torch.inference_mode()
def sample_vae(
    model: ConvolutionalVAE,
    n_samples: int,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, float]:
    model.eval()
    outputs = []
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for start_index in tqdm(range(0, n_samples, batch_size), desc="Sampling VAE"):
        current = min(batch_size, n_samples - start_index)
        z = torch.randn(current, model.latent_dim, device=DEVICE)
        outputs.append(model.decode(z).cpu())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return torch.cat(outputs, dim=0), elapsed


@torch.inference_mode()
def create_vae_reconstruction_figure(model: ConvolutionalVAE) -> None:
    model.eval()
    images, _ = next(iter(classifier_test_loader))
    images = images[:16].to(DEVICE)
    logits, _, _ = model(images)
    reconstructions = torch.sigmoid(logits)
    save_pair_grid(
        images.cpu(),
        reconstructions.cpu(),
        FIG_DIR / "vae_reconstructions.png",
        "real image",
        "VAE reconstruction",
        max_pairs=16,
    )


@torch.inference_mode()
def create_vae_latent_interpolation(model: ConvolutionalVAE, steps: int = 12) -> None:
    model.eval()
    first_image, _ = test_dataset[0]
    second_image, _ = test_dataset[1]
    batch = torch.stack([first_image, second_image]).to(DEVICE)
    mu, _ = model.encode(batch)
    alphas = torch.linspace(0, 1, steps, device=DEVICE).unsqueeze(1)
    interpolated = (1 - alphas) * mu[0:1] + alphas * mu[1:2]
    decoded = model.decode(interpolated).cpu()
    save_tensor_grid(
        decoded,
        FIG_DIR / "vae_latent_interpolation.png",
        "VAE latent interpolation",
        nrow=steps,
    )


def plot_vae_history(history: dict) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["total"], marker="o", label="Train total")
    plt.plot(epochs, history["test_total"], marker="o", label="Test total")
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO loss per image")
    plt.title("VAE total loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(FIG_DIR / "vae_total_loss.png")
    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["reconstruction"], marker="o", label="Reconstruction")
    plt.plot(epochs, history["kl"], marker="o", label="KL divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per image")
    plt.title("VAE ELBO components")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(FIG_DIR / "vae_elbo_components.png")
    plt.close(figure)

def sinusoidal_time_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    exponent = -math.log(10000) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    ) / max(half - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dimension % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x).reshape(batch, channels, height * width).transpose(1, 2)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return x + attended


class SmallUNet(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_input_dim = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(1, base_channels, 3, padding=1)

        self.down_block_1 = ResidualTimeBlock(base_channels, base_channels, time_dim)
        self.downsample_1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)

        self.down_block_2 = ResidualTimeBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample_2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)

        self.mid_block_1 = ResidualTimeBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attention = SpatialSelfAttention(base_channels * 4)
        self.mid_block_2 = ResidualTimeBlock(base_channels * 4, base_channels * 4, time_dim)

        self.upsample_1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up_block_1 = ResidualTimeBlock(base_channels * 4, base_channels * 2, time_dim)

        self.upsample_2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.up_block_2 = ResidualTimeBlock(base_channels * 2, base_channels, time_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = sinusoidal_time_embedding(timesteps, self.time_input_dim)
        time_embedding = self.time_mlp(time_embedding)

        x = self.input_conv(x)
        skip_1 = self.down_block_1(x, time_embedding)

        x = self.downsample_1(skip_1)
        skip_2 = self.down_block_2(x, time_embedding)

        x = self.downsample_2(skip_2)
        x = self.mid_block_1(x, time_embedding)
        x = self.mid_attention(x)
        x = self.mid_block_2(x, time_embedding)

        x = self.upsample_1(x)
        x = torch.cat([x, skip_2], dim=1)
        x = self.up_block_1(x, time_embedding)

        x = self.upsample_2(x)
        x = torch.cat([x, skip_1], dim=1)
        x = self.up_block_2(x, time_embedding)
        return self.output(x)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


class DiffusionProcess:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device

        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_previous = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        ).clamp(min=1e-20)

    @staticmethod
    def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus = self.extract(
            self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    @torch.inference_mode()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        betas_t = self.extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_t = self.extract(
            self.sqrt_one_minus_alpha_bars, timesteps, x.shape
        )
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, timesteps, x.shape)

        predicted_noise = model(x, timesteps)
        model_mean = sqrt_recip_alpha_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_t
        )
        posterior_variance_t = self.extract(
            self.posterior_variance, timesteps, x.shape
        )
        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().reshape(
            timesteps.shape[0], *((1,) * (x.ndim - 1))
        )
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        return_trajectory: bool = False,
        trajectory_frames: int = 10,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        model.eval()
        x = torch.randn(batch_size, 1, 28, 28, device=self.device)
        trajectory: List[torch.Tensor] = []
        capture_steps = set(
            np.linspace(self.timesteps - 1, 0, trajectory_frames, dtype=int).tolist()
        )

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full(
                (batch_size,), step, device=self.device, dtype=torch.long
            )
            x = self.p_sample(model, x, timesteps)
            if return_trajectory and step in capture_steps:
                trajectory.append(((x[:1].clamp(-1, 1) + 1) / 2).cpu())

        samples = ((x.clamp(-1, 1) + 1) / 2).cpu()
        return samples, trajectory


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        for key in target:
            if target[key].dtype.is_floating_point:
                target[key].mul_(self.decay).add_(source[key], alpha=1.0 - self.decay)
            else:
                target[key].copy_(source[key])


def train_ddpm(
    model: SmallUNet,
    diffusion: DiffusionProcess,
) -> Tuple[nn.Module, dict, float, float]:
    checkpoint_path = model_state_path("ddpm")
    ema_checkpoint_path = model_state_path("ddpm_ema")

    payload = load_checkpoint(model, checkpoint_path) if CFG.resume_if_available else None
    if payload is not None and ema_checkpoint_path.exists():
        ema_model = copy.deepcopy(model)
        load_checkpoint(ema_model, ema_checkpoint_path)
        ema_model.eval()
        print("Loaded existing DDPM and EMA checkpoints.")
        return (
            ema_model,
            payload.get("history", {}),
            payload.get("training_seconds", 0.0),
            payload.get("peak_gpu_mb", 0.0),
        )

    ema = EMA(model, CFG.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.ddpm_lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.ddpm_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    history = {"noise_mse": [], "lr": [], "epoch_seconds": []}

    reset_gpu_peak_memory()
    training_start = time.perf_counter()

    for epoch in range(1, CFG.ddpm_epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        seen = 0

        progress = tqdm(ddpm_train_loader, desc=f"DDPM {epoch}/{CFG.ddpm_epochs}")
        for images, _ in progress:
            x_start = images.to(DEVICE, non_blocking=True) * 2.0 - 1.0
            batch_size = x_start.size(0)
            timesteps = torch.randint(
                0, diffusion.timesteps, (batch_size,), device=DEVICE
            )
            noise = torch.randn_like(x_start)
            noisy_images = diffusion.q_sample(x_start, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                predicted_noise = model(noisy_images, timesteps)
                loss = F.mse_loss(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(mse=f"{loss.item():.4f}")

        epoch_seconds = time.perf_counter() - epoch_start
        history["noise_mse"].append(running_loss / seen)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_seconds"].append(epoch_seconds)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}: noise_MSE={history['noise_mse'][-1]:.5f}, "
            f"time={epoch_seconds:.1f}s"
        )

    training_seconds = time.perf_counter() - training_start
    peak_gpu_mb = gpu_peak_memory_mb()

    save_checkpoint(
        model,
        checkpoint_path,
        {
            "history": history,
            "training_seconds": training_seconds,
            "peak_gpu_mb": peak_gpu_mb,
        },
    )
    save_checkpoint(ema.model, ema_checkpoint_path)
    return ema.model, history, training_seconds, peak_gpu_mb


@torch.inference_mode()
def sample_ddpm(
    model: nn.Module,
    diffusion: DiffusionProcess,
    n_samples: int,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, float]:
    outputs = []
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for start_index in tqdm(range(0, n_samples, batch_size), desc="Sampling DDPM"):
        current = min(batch_size, n_samples - start_index)
        batch, _ = diffusion.sample(model, current)
        outputs.append(batch)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return torch.cat(outputs, dim=0), elapsed


@torch.inference_mode()
def create_ddpm_trajectory(
    model: nn.Module,
    diffusion: DiffusionProcess,
) -> None:
    _, trajectory = diffusion.sample(
        model,
        batch_size=1,
        return_trajectory=True,
        trajectory_frames=CFG.trajectory_frames,
    )
    frames = torch.cat(trajectory, dim=0)
    save_tensor_grid(
        frames,
        FIG_DIR / "ddpm_denoising_trajectory.png",
        "DDPM reverse-denoising trajectory: noise to image",
        nrow=len(frames),
    )


def plot_ddpm_history(history: dict) -> None:
    epochs = np.arange(1, len(history["noise_mse"]) + 1)
    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["noise_mse"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE")
    plt.title("DDPM training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "ddpm_noise_prediction_loss.png")
    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["epoch_seconds"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("DDPM epoch training time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "ddpm_epoch_time.png")
    plt.close(figure)

@torch.inference_mode()
def classifier_outputs(
    classifier: FashionClassifier,
    images: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    probabilities = []
    features = []

    for start_index in range(0, len(images), batch_size):
        batch = images[start_index:start_index + batch_size].to(
            DEVICE, non_blocking=True
        )
        logits = classifier(batch)
        probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
        features.append(classifier.extract_features(batch).cpu().numpy())

    return np.concatenate(probabilities, axis=0), np.concatenate(features, axis=0)


def custom_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    real_features = real_features.astype(np.float64)
    generated_features = generated_features.astype(np.float64)

    mu_real = real_features.mean(axis=0)
    mu_generated = generated_features.mean(axis=0)
    covariance_real = np.cov(real_features, rowvar=False)
    covariance_generated = np.cov(generated_features, rowvar=False)

    covariance_product = covariance_real @ covariance_generated
    covariance_sqrt = sqrtm(covariance_product)
    if np.iscomplexobj(covariance_sqrt):
        covariance_sqrt = covariance_sqrt.real

    mean_difference = mu_real - mu_generated
    score = (
        mean_difference.dot(mean_difference)
        + np.trace(covariance_real)
        + np.trace(covariance_generated)
        - 2.0 * np.trace(covariance_sqrt)
    )
    return float(max(score, 0.0))


def polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coefficient: float = 1.0,
) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coefficient) ** degree


def unbiased_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("KID requires at least two samples per set.")

    kernel_xx = polynomial_kernel(x, x)
    kernel_yy = polynomial_kernel(y, y)
    kernel_xy = polynomial_kernel(x, y)

    sum_xx = (kernel_xx.sum() - np.trace(kernel_xx)) / (m * (m - 1))
    sum_yy = (kernel_yy.sum() - np.trace(kernel_yy)) / (n * (n - 1))
    sum_xy = kernel_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def custom_kid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    subset_size: int,
    repeats: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, len(real_features), len(generated_features))
    scores = []
    for _ in range(repeats):
        real_indices = rng.choice(len(real_features), subset_size, replace=False)
        generated_indices = rng.choice(
            len(generated_features), subset_size, replace=False
        )
        scores.append(
            unbiased_mmd2(
                real_features[real_indices],
                generated_features[generated_indices],
            )
        )
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def normalized_entropy(distribution: np.ndarray) -> float:
    distribution = distribution / distribution.sum()
    safe = distribution[distribution > 0]
    entropy = -(safe * np.log(safe)).sum()
    return float(entropy / np.log(len(distribution)))


def inception_score_from_probabilities(
    probabilities: np.ndarray,
    splits: int = 10,
) -> Tuple[float, float]:
    split_scores = []
    for split in np.array_split(probabilities, splits):
        marginal = split.mean(axis=0, keepdims=True)
        kl = split * (
            np.log(split.clip(1e-12))
            - np.log(marginal.clip(1e-12))
        )
        split_scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(split_scores)), float(np.std(split_scores, ddof=1))


def classifier_based_statistics(probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    counts = np.bincount(predicted, minlength=10)
    distribution = counts / counts.sum()
    minimum_count = max(5, int(0.001 * len(predicted)))
    coverage = int((counts >= minimum_count).sum())
    inception_score, inception_score_std = inception_score_from_probabilities(
        probabilities, splits=min(10, len(probabilities))
    )
    return {
        "mean_classifier_confidence": float(confidence.mean()),
        "median_classifier_confidence": float(np.median(confidence)),
        "recognizability_rate_at_0.8": float((confidence >= 0.8).mean()),
        "class_coverage": coverage,
        "class_distribution_entropy": normalized_entropy(distribution),
        "inception_score": inception_score,
        "inception_score_std": inception_score_std,
        "class_counts": counts,
        "class_distribution": distribution,
        "predicted_classes": predicted,
        "confidence": confidence,
    }


def sharpness_score(images: torch.Tensor) -> float:
    """Mean Sobel gradient magnitude. Higher usually means sharper edges."""
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    images = images.float().cpu()
    gradient_x = F.conv2d(images, kernel_x, padding=1)
    gradient_y = F.conv2d(images, kernel_y, padding=1)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)
    return float(magnitude.mean())


def random_feature_pair_distance(
    features: np.ndarray,
    pairs: int = 5000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(features), size=pairs)
    second = rng.integers(0, len(features), size=pairs)
    unequal = first != second
    distances = np.linalg.norm(features[first[unequal]] - features[second[unequal]], axis=1)
    return float(distances.mean())


@torch.inference_mode()
def collect_dataset_images(dataset, count: int) -> torch.Tensor:
    count = min(count, len(dataset))
    indices = np.random.default_rng(CFG.seed).choice(len(dataset), count, replace=False)
    return torch.stack([dataset[int(index)][0] for index in indices])


def chunked_nearest_neighbors(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    query_chunk_size: int = 128,
    reference_chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    query_tensor = torch.from_numpy(query_features).float().to(DEVICE)
    reference_tensor = torch.from_numpy(reference_features).float().to(DEVICE)
    all_distances = []
    all_indices = []

    for query_start in tqdm(
        range(0, len(query_tensor), query_chunk_size),
        desc="Nearest-neighbor search",
    ):
        query_chunk = query_tensor[query_start:query_start + query_chunk_size]
        best_distance = torch.full(
            (len(query_chunk),), float("inf"), device=DEVICE
        )
        best_index = torch.full(
            (len(query_chunk),), -1, dtype=torch.long, device=DEVICE
        )

        for reference_start in range(0, len(reference_tensor), reference_chunk_size):
            reference_chunk = reference_tensor[
                reference_start:reference_start + reference_chunk_size
            ]
            distances = torch.cdist(query_chunk, reference_chunk)
            local_distance, local_index = distances.min(dim=1)
            improved = local_distance < best_distance
            best_distance[improved] = local_distance[improved]
            best_index[improved] = local_index[improved] + reference_start

        all_distances.append(best_distance.cpu().numpy())
        all_indices.append(best_index.cpu().numpy())

    return np.concatenate(all_distances), np.concatenate(all_indices)


def compute_memorization_metrics(
    model_name: str,
    generated_images: torch.Tensor,
    classifier: FashionClassifier,
    train_reference_images: torch.Tensor,
    train_reference_features: np.ndarray,
    baseline_real_images: torch.Tensor,
    baseline_real_features: np.ndarray,
) -> dict:
    generated_subset = generated_images[:CFG.nn_generated_samples]
    _, generated_features = classifier_outputs(classifier, generated_subset)
    generated_distances, generated_indices = chunked_nearest_neighbors(
        generated_features, train_reference_features
    )
    baseline_distances, _ = chunked_nearest_neighbors(
        baseline_real_features, train_reference_features
    )

    threshold = float(np.percentile(baseline_distances, 1))
    memorization_rate = float((generated_distances <= threshold).mean())

    nearest_images = train_reference_images[generated_indices]
    ordering = np.argsort(generated_distances)
    ordering_tensor = torch.from_numpy(ordering).long()
    save_pair_grid(
        generated_subset[ordering_tensor],
        nearest_images[ordering_tensor],
        FIG_DIR / f"{model_name.lower()}_nearest_neighbors.png",
        f"{model_name} generated",
        "nearest training image",
        max_pairs=16,
    )

    table = pd.DataFrame({
        "generated_index": np.arange(len(generated_distances)),
        "nearest_reference_index": generated_indices,
        "feature_distance": generated_distances,
        "below_1pct_real_baseline_threshold": generated_distances <= threshold,
    }).sort_values("feature_distance")
    table.to_csv(TABLE_DIR / f"{model_name.lower()}_nearest_neighbor_distances.csv", index=False)

    return {
        "nearest_neighbor_mean_distance": float(generated_distances.mean()),
        "nearest_neighbor_median_distance": float(np.median(generated_distances)),
        "memorization_rate_below_real_1pct_threshold": memorization_rate,
        "memorization_threshold": threshold,
    }


def loss_stability_statistics(losses: Sequence[float]) -> dict:
    values = np.asarray(losses, dtype=np.float64)
    if len(values) == 0:
        return {
            "final_loss": float("nan"),
            "tail_coefficient_of_variation": float("nan"),
            "loss_slope": float("nan"),
            "nonfinite_count": 0,
        }
    tail = values[max(0, int(0.8 * len(values))):]
    coefficient = float(tail.std() / max(abs(tail.mean()), 1e-12))
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
    return {
        "final_loss": float(values[-1]),
        "tail_coefficient_of_variation": coefficient,
        "loss_slope": slope,
        "nonfinite_count": int((~np.isfinite(values)).sum()),
    }


def save_low_confidence_grid(
    model_name: str,
    generated_images: torch.Tensor,
    stats: dict,
) -> None:
    ordering = torch.from_numpy(np.argsort(stats["confidence"])[:64]).long()
    save_tensor_grid(
        generated_images[ordering],
        FIG_DIR / f"{model_name.lower()}_lowest_confidence_samples.png",
        f"{model_name}: lowest classifier-confidence samples (failure-case inspection)",
        nrow=8,
    )

def plot_class_distribution(model_stats: Dict[str, dict]) -> None:
    positions = np.arange(10)
    width = 0.36

    figure = plt.figure(figsize=(11, 5.5))
    plt.bar(
        positions - width / 2,
        model_stats["VAE"]["class_distribution"],
        width=width,
        label="VAE",
    )
    plt.bar(
        positions + width / 2,
        model_stats["DDPM"]["class_distribution"],
        width=width,
        label="DDPM",
    )
    plt.axhline(0.1, linestyle="--", linewidth=1.2, label="Balanced target (0.1)")
    plt.xticks(positions, FASHION_MNIST_CLASSES, rotation=35, ha="right")
    plt.ylabel("Predicted class proportion")
    plt.title("Generated class distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    figure.savefig(FIG_DIR / "class_distribution_comparison.png")
    plt.close(figure)


def plot_distribution_metrics(metrics_frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["custom_fid"])
    plt.ylabel("Classifier-feature FID (lower is better)")
    plt.title("Distribution similarity: custom FID")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "custom_fid_comparison.png")
    plt.close(figure)

    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["custom_kid_mean"])
    plt.errorbar(
        metrics_frame["model"],
        metrics_frame["custom_kid_mean"],
        yerr=metrics_frame["custom_kid_std"],
        fmt="none",
        capsize=5,
    )
    plt.ylabel("Classifier-feature KID (lower is better)")
    plt.title("Distribution similarity: custom KID")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "custom_kid_comparison.png")
    plt.close(figure)


def plot_recognizability_metrics(metrics_frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["mean_classifier_confidence"] * 100)
    plt.ylabel("Mean classifier confidence (%)")
    plt.title("Generated-image recognizability")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "classifier_confidence_comparison.png")
    plt.close(figure)

    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["class_distribution_entropy"] * 100)
    plt.ylabel("Normalized class entropy (%)")
    plt.title("Generated class-distribution diversity")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "class_entropy_comparison.png")
    plt.close(figure)


def plot_compute_metrics(metrics_frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["training_minutes"])
    plt.ylabel("Training time (minutes)")
    plt.title("Measured training cost")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "training_time_comparison.png")
    plt.close(figure)

    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["sampling_ms_per_image"])
    plt.ylabel("Sampling time (ms per image)")
    plt.title("Measured generation speed")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "sampling_time_comparison.png")
    plt.close(figure)


def plot_sharpness(metrics_frame: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7, 5))
    plt.bar(metrics_frame["model"], metrics_frame["sobel_sharpness"])
    plt.ylabel("Mean Sobel gradient magnitude")
    plt.title("Edge sharpness proxy")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    figure.savefig(FIG_DIR / "sharpness_comparison.png")
    plt.close(figure)


def winner(metrics_frame: pd.DataFrame, column: str, lower_is_better: bool) -> str:
    row = (
        metrics_frame.loc[metrics_frame[column].idxmin()]
        if lower_is_better
        else metrics_frame.loc[metrics_frame[column].idxmax()]
    )
    return str(row["model"])


def format_float(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def create_results_markdown(
    metrics_frame: pd.DataFrame,
    classifier_accuracy: float,
    classifier_test_loss: float,
    vae_validation: dict,
    config: ExperimentConfig,
) -> None:
    fid_winner = winner(metrics_frame, "custom_fid", lower_is_better=True)
    kid_winner = winner(metrics_frame, "custom_kid_mean", lower_is_better=True)
    confidence_winner = winner(
        metrics_frame, "mean_classifier_confidence", lower_is_better=False
    )
    entropy_winner = winner(
        metrics_frame, "class_distribution_entropy", lower_is_better=False
    )
    training_winner = winner(metrics_frame, "training_minutes", lower_is_better=True)
    sampling_winner = winner(
        metrics_frame, "sampling_ms_per_image", lower_is_better=True
    )

    table_markdown = metrics_frame[
        [
            "model",
            "custom_fid",
            "custom_kid_mean",
            "mean_classifier_confidence",
            "class_coverage",
            "class_distribution_entropy",
            "nearest_neighbor_mean_distance",
            "memorization_rate_below_real_1pct_threshold",
            "training_minutes",
            "sampling_ms_per_image",
        ]
    ].to_markdown(index=False, floatfmt=".4f")

    text = f"""# VAE vs DDPM on Fashion-MNIST — Generated Results

## Experimental setup

- Profile: `{config.profile}`
- Device: `{DEVICE}`
- Classifier test accuracy: **{classifier_accuracy:.2%}**
- Classifier test loss: **{classifier_test_loss:.4f}**
- Evaluation samples per model: **{config.eval_samples:,}**
- DDPM diffusion steps: **{config.diffusion_steps}**
- VAE latent dimension: **{config.latent_dim}**

## Main quantitative results

{table_markdown}

## Automatically derived comparisons

- Lowest classifier-feature FID: **{fid_winner}**
- Lowest classifier-feature KID: **{kid_winner}**
- Highest mean classifier confidence: **{confidence_winner}**
- Highest class-distribution entropy: **{entropy_winner}**
- Shortest measured training time: **{training_winner}**
- Fastest measured sampling: **{sampling_winner}**
- VAE test negative-ELBO components: total **{vae_validation['total']:.3f}**, reconstruction **{vae_validation['reconstruction']:.3f}**, KL **{vae_validation['kl']:.3f}**

## How to use these results in the report

Use the FID and KID results together when discussing distribution similarity. Use classifier confidence and the low-confidence sample grids for recognizability and failure cases. Use class coverage, normalized class entropy, and the class-distribution chart for diversity. Use the nearest-neighbor figures and calibrated memorization rate for novelty. Use the VAE interpolation and DDPM denoising trajectory for generation structure. Use measured training and sampling times, parameter counts, and peak GPU memory for computational cost.

## Important limitations

1. The reported FID and KID are **custom classifier-feature metrics**, not standard ImageNet Inception metrics. This is intentional because Fashion-MNIST is 28×28 grayscale and is not well matched to ImageNet features.
2. Classifier confidence is a recognizability proxy, not a direct human-quality score. Interpret it with sample grids and diversity statistics.
3. Nearest-neighbor analysis uses a fixed subset of training references for tractability. It is a memorization screen, not a mathematical proof that memorization is absent.
4. Results vary with random seed, training duration, model capacity, and hardware. For a stronger final report, repeat the report profile with several seeds and report mean ± standard deviation.

## Output map

- `figures/`: report-ready PNG charts and qualitative figures
- `tables/model_comparison_metrics.csv`: main results table
- `tables/*nearest_neighbor_distances.csv`: detailed memorization distances
- `logs/`: training histories
- `models/`: trained classifier, VAE, DDPM, and EMA DDPM checkpoints
- `samples/`: generated tensor files
- `results.json`: machine-readable metrics and configuration
"""
    (OUTPUT_DIR / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")


def save_environment_info() -> None:
    packages = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(packages, OUTPUT_DIR / "environment.json")

def run_complete_experiment() -> pd.DataFrame:
    save_json(asdict(CFG), OUTPUT_DIR / "config.json")
    save_environment_info()

    # 1) Train and validate the feature classifier.
    classifier = FashionClassifier(feature_dim=128).to(DEVICE)
    classifier_history, classifier_seconds, classifier_peak_gpu = train_classifier(classifier)
    classifier_test_loss, classifier_accuracy, confusion = evaluate_classifier(
        classifier, classifier_test_loader
    )
    plot_classifier_history(classifier_history)
    plot_confusion_matrix(confusion)
    pd.DataFrame(classifier_history).to_csv(
        LOG_DIR / "classifier_history.csv", index=False
    )

    if classifier_accuracy < 0.88:
        warnings.warn(
            f"Classifier test accuracy is only {classifier_accuracy:.2%}. "
            "Custom FID/KID and recognizability metrics may be unreliable. "
            "Use the standard or report profile."
        )

    # 2) Train VAE.
    vae = ConvolutionalVAE(latent_dim=CFG.latent_dim).to(DEVICE)
    vae_history, vae_training_seconds, vae_peak_gpu = train_vae(vae)
    vae_validation = evaluate_vae(vae)
    plot_vae_history(vae_history)
    create_vae_reconstruction_figure(vae)
    create_vae_latent_interpolation(vae)
    pd.DataFrame(vae_history).to_csv(LOG_DIR / "vae_history.csv", index=False)

    # 3) Train DDPM with EMA sampling model.
    diffusion = DiffusionProcess(CFG.diffusion_steps, DEVICE)
    ddpm = SmallUNet(base_channels=64, time_dim=256).to(DEVICE)
    ddpm_ema, ddpm_history, ddpm_training_seconds, ddpm_peak_gpu = train_ddpm(
        ddpm, diffusion
    )
    plot_ddpm_history(ddpm_history)
    create_ddpm_trajectory(ddpm_ema, diffusion)
    pd.DataFrame(ddpm_history).to_csv(LOG_DIR / "ddpm_history.csv", index=False)

    # 4) Generate the same number of samples for both models.
    vae_samples, vae_sampling_seconds = sample_vae(
        vae, CFG.eval_samples, batch_size=512
    )
    ddpm_samples, ddpm_sampling_seconds = sample_ddpm(
        ddpm_ema, diffusion, CFG.eval_samples, batch_size=CFG.ddpm_batch_size
    )
    torch.save(vae_samples.half(), SAMPLE_DIR / "vae_generated_samples.pt")
    torch.save(ddpm_samples.half(), SAMPLE_DIR / "ddpm_generated_samples.pt")

    save_tensor_grid(
        vae_samples[:64],
        FIG_DIR / "vae_sample_grid.png",
        "VAE generated samples",
        nrow=8,
    )
    save_tensor_grid(
        ddpm_samples[:64],
        FIG_DIR / "ddpm_sample_grid.png",
        "DDPM generated samples",
        nrow=8,
    )

    # 5) Use one shared real reference set for fair FID/KID comparisons.
    real_images = collect_dataset_images(test_dataset, CFG.eval_samples)
    save_tensor_grid(
        real_images[:64],
        FIG_DIR / "real_sample_grid.png",
        "Real Fashion-MNIST reference images",
        nrow=8,
    )
    comparison = torch.cat([real_images[:8], vae_samples[:8], ddpm_samples[:8]], dim=0)
    save_tensor_grid(
        comparison,
        FIG_DIR / "real_vae_ddpm_comparison.png",
        "Rows: real reference, VAE samples, DDPM samples",
        nrow=8,
    )
    real_probabilities, real_features = classifier_outputs(classifier, real_images)
    vae_probabilities, vae_features = classifier_outputs(classifier, vae_samples)
    ddpm_probabilities, ddpm_features = classifier_outputs(classifier, ddpm_samples)

    np.save(TABLE_DIR / "real_classifier_features.npy", real_features)
    np.save(TABLE_DIR / "vae_classifier_features.npy", vae_features)
    np.save(TABLE_DIR / "ddpm_classifier_features.npy", ddpm_features)

    vae_stats = classifier_based_statistics(vae_probabilities)
    ddpm_stats = classifier_based_statistics(ddpm_probabilities)
    model_stats = {"VAE": vae_stats, "DDPM": ddpm_stats}

    save_low_confidence_grid("VAE", vae_samples, vae_stats)
    save_low_confidence_grid("DDPM", ddpm_samples, ddpm_stats)
    plot_class_distribution(model_stats)

    # 6) Calibrated nearest-neighbor memorization analysis.
    train_reference_images = collect_dataset_images(
        train_dataset, CFG.nn_reference_samples
    )
    _, train_reference_features = classifier_outputs(
        classifier, train_reference_images
    )
    baseline_real_images = collect_dataset_images(
        test_dataset, CFG.nn_generated_samples
    )
    _, baseline_real_features = classifier_outputs(
        classifier, baseline_real_images
    )

    vae_memorization = compute_memorization_metrics(
        "VAE",
        vae_samples,
        classifier,
        train_reference_images,
        train_reference_features,
        baseline_real_images,
        baseline_real_features,
    )
    ddpm_memorization = compute_memorization_metrics(
        "DDPM",
        ddpm_samples,
        classifier,
        train_reference_images,
        train_reference_features,
        baseline_real_images,
        baseline_real_features,
    )

    # 7) Distribution-level and supporting quality metrics.
    vae_kid_mean, vae_kid_std = custom_kid(
        real_features,
        vae_features,
        CFG.kid_subset_size,
        CFG.kid_repeats,
        CFG.seed,
    )
    ddpm_kid_mean, ddpm_kid_std = custom_kid(
        real_features,
        ddpm_features,
        CFG.kid_subset_size,
        CFG.kid_repeats,
        CFG.seed + 1,
    )

    vae_stability = loss_stability_statistics(vae_history["total"])
    ddpm_stability = loss_stability_statistics(ddpm_history["noise_mse"])

    records = []
    for model_name, samples, features, stats, memorization, training_seconds, sampling_seconds, params, peak_gpu, fid_score, kid_mean, kid_std, stability in [
        (
            "VAE",
            vae_samples,
            vae_features,
            vae_stats,
            vae_memorization,
            vae_training_seconds,
            vae_sampling_seconds,
            count_parameters(vae),
            vae_peak_gpu,
            custom_fid(real_features, vae_features),
            vae_kid_mean,
            vae_kid_std,
            vae_stability,
        ),
        (
            "DDPM",
            ddpm_samples,
            ddpm_features,
            ddpm_stats,
            ddpm_memorization,
            ddpm_training_seconds,
            ddpm_sampling_seconds,
            count_parameters(ddpm),
            ddpm_peak_gpu,
            custom_fid(real_features, ddpm_features),
            ddpm_kid_mean,
            ddpm_kid_std,
            ddpm_stability,
        ),
    ]:
        record = {
            "model": model_name,
            "custom_fid": fid_score,
            "custom_kid_mean": kid_mean,
            "custom_kid_std": kid_std,
            "mean_classifier_confidence": stats["mean_classifier_confidence"],
            "median_classifier_confidence": stats["median_classifier_confidence"],
            "recognizability_rate_at_0.8": stats["recognizability_rate_at_0.8"],
            "class_coverage": stats["class_coverage"],
            "class_distribution_entropy": stats["class_distribution_entropy"],
            "inception_score": stats["inception_score"],
            "inception_score_std": stats["inception_score_std"],
            "feature_pairwise_distance": random_feature_pair_distance(
                features, seed=CFG.seed
            ),
            "sobel_sharpness": sharpness_score(samples),
            "nearest_neighbor_mean_distance": memorization["nearest_neighbor_mean_distance"],
            "nearest_neighbor_median_distance": memorization["nearest_neighbor_median_distance"],
            "memorization_rate_below_real_1pct_threshold": memorization[
                "memorization_rate_below_real_1pct_threshold"
            ],
            "training_minutes": training_seconds / 60.0,
            "sampling_total_seconds": sampling_seconds,
            "sampling_ms_per_image": sampling_seconds / len(samples) * 1000.0,
            "trainable_parameters": params,
            "peak_gpu_memory_mb": peak_gpu,
            "final_training_loss": stability["final_loss"],
            "training_tail_cv": stability["tail_coefficient_of_variation"],
            "training_loss_slope": stability["loss_slope"],
            "nonfinite_loss_count": stability["nonfinite_count"],
        }
        records.append(record)

    metrics_frame = pd.DataFrame(records)
    metrics_frame.to_csv(TABLE_DIR / "model_comparison_metrics.csv", index=False)

    class_distribution_frame = pd.DataFrame({
        "class_index": np.arange(10),
        "class_name": FASHION_MNIST_CLASSES,
        "real_test_distribution": np.bincount(
            real_probabilities.argmax(axis=1), minlength=10
        ) / len(real_probabilities),
        "vae_distribution": vae_stats["class_distribution"],
        "ddpm_distribution": ddpm_stats["class_distribution"],
    })
    class_distribution_frame.to_csv(
        TABLE_DIR / "class_distribution.csv", index=False
    )

    plot_distribution_metrics(metrics_frame)
    plot_recognizability_metrics(metrics_frame)
    plot_compute_metrics(metrics_frame)
    plot_sharpness(metrics_frame)

    results_payload = {
        "config": asdict(CFG),
        "classifier": {
            "test_loss": classifier_test_loss,
            "test_accuracy": classifier_accuracy,
            "training_seconds": classifier_seconds,
            "peak_gpu_memory_mb": classifier_peak_gpu,
            "trainable_parameters": count_parameters(classifier),
        },
        "vae_test_loss": vae_validation,
        "models": metrics_frame.to_dict(orient="records"),
        "class_distributions": class_distribution_frame.to_dict(orient="records"),
        "metric_note": (
            "FID and KID use 128-dimensional features from a Fashion-MNIST "
            "classifier trained on real data. They are not standard Inception metrics."
        ),
    }
    save_json(results_payload, OUTPUT_DIR / "results.json")
    create_results_markdown(
        metrics_frame,
        classifier_accuracy,
        classifier_test_loss,
        vae_validation,
        CFG,
    )

    print("\nMain results:")
    print(metrics_frame.round(5).to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    return metrics_frame

def package_results() -> Path:
    zip_path = Path("./results.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=OUTPUT_DIR.parent,
        base_dir=OUTPUT_DIR.name,
    )
    print(f"Created: {zip_path.resolve()}")
    return zip_path


if __name__ == "__main__":
    run_complete_experiment()
    package_results()
