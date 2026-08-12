"""Three-model, two-dataset, five-seed RGB generative benchmark.

Models: convolutional VAE, pixel-space DDPM, and latent diffusion (LDM).
Datasets: Fashion-MNIST and Fashion Product Images Small (FPIS).

The FPIS images are restricted to Apparel, stratified over the 15 most common
article types, center-padded, and resized to 32x32 RGB. Fashion-MNIST is
replicated to RGB so model architecture and evaluator capacity remain matched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = Path(os.environ.get("BENCHMARK_OUTPUT_DIR", ROOT / "results_final_3models_2datasets_5seeds"))
DATA = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if AMP and torch.cuda.is_bf16_supported() else torch.float16
USE_GRAD_SCALER = AMP and AMP_DTYPE == torch.float16


@dataclass
class Config:
    pipeline_version: str = "2026-08-12-final-v3"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs_classifier_fmnist: int = 40
    epochs_classifier_fpis: int = 40
    classifier_batch_size: int = 256
    generative_batch_size: int = 256
    sampling_batch_size: int = 4096
    vae_steps: int = 3000
    ddpm_steps: int = 5000
    ldm_steps: int = 5000
    eval_samples: int = 5000
    diffusion_steps: int = 100
    latent_channels: int = 4
    vae_lr: float = 1e-3
    diffusion_lr: float = 2e-4
    ddpm_ema_decay: float = 0.995
    ldm_ema_decay: float = 0.999
    ddpm_clean_clip: float = 1.0
    ldm_clean_clip: float = 3.0
    classifier_lr_fmnist: float = 3e-4
    classifier_lr_fpis: float = 1e-3
    classifier_weight_decay: float = 1e-4
    generative_weight_decay: float = 1e-4
    beta: float = 1.0
    grad_clip: float = 1.0
    warmup_steps: int = 200
    kid_subset: int = 1000
    kid_repeats: int = 20
    fpis_test_fraction: float = 0.15
    fpis_classes: int = 15


CFG = Config()
if os.environ.get("BENCHMARK_QUICK", "0") == "1":
    CFG.epochs_classifier_fmnist = CFG.epochs_classifier_fpis = 1
    CFG.vae_steps = CFG.ddpm_steps = CFG.ldm_steps = 20
    CFG.eval_samples, CFG.diffusion_steps, CFG.kid_subset, CFG.kid_repeats = 256, 20, 128, 3
    CFG.sampling_batch_size = 256
    CFG.seeds = (0,)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def amp_context():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP else nullcontext()


class FPISDataset(Dataset):
    def __init__(self, records, indices, classes):
        self.records, self.indices, self.classes = records, list(indices), classes
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        row = self.records[int(self.indices[i])]
        image = row["image"].convert("RGB")
        w, h = image.size
        side = max(w, h)
        left, top = (side - w) // 2, (side - h) // 2
        image = TF.pad(image, [left, top, side - w - left, side - h - top], fill=255)
        image = TF.resize(image, [32, 32], antialias=True)
        return self.to_tensor(image), self.classes.index(row["articleType"])


def load_datasets(name: str):
    if name == "fashion_mnist":
        transform = transforms.Compose([transforms.Resize((32,32)), transforms.Grayscale(3), transforms.ToTensor()])
        train = tv_datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
        test = tv_datasets.FashionMNIST(DATA, train=False, download=True, transform=transform)
        return train, test, list(train.classes)
    from datasets import load_dataset
    records = load_dataset("Transformersx/fashion-product-images-small", split="train")
    labels = np.asarray(records["articleType"], dtype=object)
    apparel = np.asarray(records["masterCategory"], dtype=object) == "Apparel"
    values, counts = np.unique(labels[apparel], return_counts=True)
    classes = values[np.argsort(counts)[-CFG.fpis_classes:][::-1]].tolist()
    rng = np.random.default_rng(1508)
    train_idx, test_idx = [], []
    for cls in classes:
        idx = np.flatnonzero(apparel & (labels == cls)); rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - CFG.fpis_test_fraction)))
        train_idx.extend(idx[:cut]); test_idx.extend(idx[cut:])
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return FPISDataset(records, train_idx, classes), FPISDataset(records, test_idx, classes), classes


class ClassifierResidual(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__(); self.c1=nn.Conv2d(channels,channels,3,stride,padding=1,bias=False); self.b1=nn.BatchNorm2d(channels); self.c2=nn.Conv2d(channels,channels,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(channels)
    def forward(self,x): return F.silu(self.b2(self.c2(F.silu(self.b1(self.c1(x)))))+x)


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),ClassifierResidual(64),ClassifierResidual(64),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.SiLU(),ClassifierResidual(128),ClassifierResidual(128),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1,bias=False),nn.BatchNorm2d(256),nn.SiLU(),ClassifierResidual(256),ClassifierResidual(256),nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x, features=False):
        z = self.body(x).flatten(1)
        return (self.fc(z), z) if features else self.fc(z)


class ConvResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)


class VAE(nn.Module):
    """Spatial VAE whose 4x8x8 latents are also used by the LDM."""
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), ConvResidual(64),
            nn.Conv2d(64, 128, 4, 2, 1), ConvResidual(128),
            nn.Conv2d(128, 128, 4, 2, 1), ConvResidual(128),
        )
        self.moments = nn.Conv2d(128, 2 * latent_channels, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1), ConvResidual(128),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, padding=1), ConvResidual(128),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 64, 3, padding=1), ConvResidual(64),
            nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv2d(64, 3, 3, padding=1),
        )
    def encode(self, x): return self.moments(self.encoder(x)).chunk(2, dim=1)
    def decode_logits(self, z): return self.decoder(z)
    def decode(self, z): return self.decode_logits(z).sigmoid()
    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = logvar.clamp(-12, 12)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode_logits(z), mu, logvar


def tembed(t, dim=128):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
    x=t.float()[:,None]*f[None]; return torch.cat([x.sin(),x.cos()],1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(embedding))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__(); self.heads = heads; self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1); self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        batch, channels, height, width = x.shape; head_dim = channels // self.heads
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        def split(value): return value.reshape(batch, self.heads, head_dim, height * width).transpose(-1, -2)
        attended = F.scaled_dot_product_attention(split(q), split(k), split(v))
        attended = attended.transpose(-1, -2).reshape(batch, channels, height, width)
        return x + self.proj(attended)


class UNetDenoiser(nn.Module):
    """Two-level time-conditioned U-Net with bottleneck attention."""
    def __init__(self, channels, base=64):
        super().__init__(); time_dim = 256
        self.time = nn.Sequential(nn.Linear(128, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(channels, base, 3, padding=1)
        self.enc1a, self.enc1b = ResBlock(base, base), ResBlock(base, base)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.enc2a, self.enc2b = ResBlock(base * 2, base * 2), ResBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.mid1 = ResBlock(base * 4, base * 4); self.attn = SpatialAttention(base * 4); self.mid2 = ResBlock(base * 4, base * 4)
        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2a, self.dec2b = ResBlock(base * 4, base * 2), ResBlock(base * 2, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1a, self.dec1b = ResBlock(base * 2, base), ResBlock(base, base)
        self.output = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, channels, 3, padding=1))
    def forward(self, x, timesteps):
        embedding = self.time(tembed(timesteps))
        e1 = self.enc1b(self.enc1a(self.input(x), embedding), embedding)
        e2 = self.enc2b(self.enc2a(self.down1(e1), embedding), embedding)
        h = self.mid2(self.attn(self.mid1(self.down2(e2), embedding)), embedding)
        h = F.interpolate(h, scale_factor=2, mode="nearest"); h = self.up2(h)
        h = self.dec2b(self.dec2a(torch.cat([h, e2], dim=1), embedding), embedding)
        h = F.interpolate(h, scale_factor=2, mode="nearest"); h = self.up1(h)
        h = self.dec1b(self.dec1a(torch.cat([h, e1], dim=1), embedding), embedding)
        return self.output(h)


class PixelUNet(UNetDenoiser):
    def __init__(self): super().__init__(channels=3, base=64)


class LatentDenoiser(UNetDenoiser):
    def __init__(self, latent_channels=4): super().__init__(channels=latent_channels, base=64)


class Diffusion:
    def __init__(self, steps):
        self.steps=steps; s=.008; x=torch.linspace(0,steps,steps+1,device=DEVICE); ac=torch.cos(((x/steps+s)/(1+s))*math.pi*.5)**2; ac=ac/ac[0]; self.betas=torch.clip(1-ac[1:]/ac[:-1],1e-4,.999); self.alphas=1-self.betas; self.ac=torch.cumprod(self.alphas,0)
    def noise(self,x,t):
        eps=torch.randn_like(x); shape=(-1,)+(1,)*(x.ndim-1); return self.ac[t].sqrt().view(shape)*x+(1-self.ac[t]).sqrt().view(shape)*eps,eps
    @torch.no_grad()
    def sample(self,model,shape,clean_clip=None):
        x=torch.randn(shape,device=DEVICE)
        for i in range(self.steps-1,-1,-1):
            t=torch.full((shape[0],),i,device=DEVICE,dtype=torch.long); b=self.betas[i]; a=self.alphas[i]; ac=self.ac[i]
            with amp_context(): predicted_noise=model(x,t)
            predicted_noise=predicted_noise.float()
            if clean_clip is None:
                mean=(x-(b/torch.sqrt(1-ac))*predicted_noise)/torch.sqrt(a)
                x=mean+(b.sqrt()*torch.randn_like(x) if i else 0)
                continue
            previous=torch.tensor(1.0,device=DEVICE) if i==0 else self.ac[i-1]
            predicted_clean=(x-torch.sqrt(1-ac)*predicted_noise)/torch.sqrt(ac)
            predicted_clean=predicted_clean.clamp(-clean_clip,clean_clip)
            coefficient_clean=b*torch.sqrt(previous)/(1-ac)
            coefficient_noisy=(1-previous)*torch.sqrt(a)/(1-ac)
            mean=coefficient_clean*predicted_clean+coefficient_noisy*x
            posterior_variance=b*(1-previous)/(1-ac)
            x=mean+(posterior_variance.clamp_min(0).sqrt()*torch.randn_like(x) if i else 0)
        return x


def loader(ds, shuffle, batch_size, drop=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop,
                      num_workers=4, pin_memory=AMP, persistent_workers=True, prefetch_factor=4)


def cosine_with_warmup(optimizer, total_steps):
    def schedule(step):
        if step < CFG.warmup_steps:
            return (step + 1) / max(1, CFG.warmup_steps)
        progress = (step - CFG.warmup_steps) / max(1, total_steps - CFG.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def reset_peak_memory():
    if AMP:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024**2 if AMP else 0.0


def synchronize():
    if AMP: torch.cuda.synchronize()


def train_classifier(model, train, test, dataset_name):
    epochs = CFG.epochs_classifier_fmnist if dataset_name == "fashion_mnist" else CFG.epochs_classifier_fpis
    learning_rate=CFG.classifier_lr_fmnist if dataset_name=="fashion_mnist" else CFG.classifier_lr_fpis
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=CFG.classifier_weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=USE_GRAD_SCALER); reset_peak_memory(); start=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x,y in train:
            x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            # Tensor-space augmentation works for both dataset wrappers and is
            # deliberately limited so product orientation remains meaningful.
            if torch.rand(()) < .5: x=torch.flip(x,(-1,))
            with amp_context(): loss=F.cross_entropy(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
    model.eval(); correct=n=0; per_correct=torch.zeros(model.fc.out_features); per_total=torch.zeros(model.fc.out_features); losses=[]
    with torch.no_grad():
        for x,y in test:
            logits=model(x.to(DEVICE)); pred=logits.argmax(1).cpu(); correct+=(pred==y).sum().item(); n+=len(y); losses.append(F.cross_entropy(logits,y.to(DEVICE)).item()*len(y))
            for c in range(model.fc.out_features): per_correct[c]+=((pred==c)&(y==c)).sum(); per_total[c]+=(y==c).sum()
    metrics={"accuracy":correct/n,"balanced_accuracy":float((per_correct/per_total.clamp_min(1)).mean()),"test_loss":sum(losses)/n,
             "training_seconds":time.perf_counter()-start,"peak_gpu_memory_mb":peak_memory_mb()}
    return metrics


@torch.no_grad()
def classifier_diagnostics(model, test, n_classes):
    """Return a row-normalized confusion matrix and per-class recall."""
    confusion=torch.zeros((n_classes,n_classes),dtype=torch.int64)
    model.eval()
    for images,targets in test:
        predictions=model(images.to(DEVICE,non_blocking=True)).argmax(1).cpu()
        pairs=targets.to(torch.int64)*n_classes+predictions.to(torch.int64)
        confusion += torch.bincount(pairs,minlength=n_classes*n_classes).reshape(n_classes,n_classes)
    recall=confusion.diag()/confusion.sum(1).clamp_min(1)
    normalized=confusion/confusion.sum(1,keepdim=True).clamp_min(1)
    return confusion.numpy(),normalized.numpy(),recall.numpy()


def train_vae(model, train):
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG.vae_lr,weight_decay=CFG.generative_weight_decay)
    scheduler=cosine_with_warmup(optimizer,CFG.vae_steps); scaler=torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)
    iterator=iter(train); history=[]; reset_peak_memory(); start=None; model.train()
    for step in range(CFG.vae_steps):
        try: x,_=next(iterator)
        except StopIteration: iterator=iter(train); x,_=next(iterator)
        x=x.to(DEVICE,non_blocking=True); optimizer.zero_grad(set_to_none=True)
        with amp_context(): rec,mu,lv=model(x)
        rec_loss=F.binary_cross_entropy_with_logits(rec.float(),x.float(),reduction="sum")/len(x)
        kl=-.5*(1+lv.float()-mu.float().square()-lv.float().exp()).sum()/len(x)
        beta=CFG.beta*min(1.0,(step+1)/max(1,CFG.warmup_steps)); loss=rec_loss+beta*kl
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
        previous_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()>=previous_scale: scheduler.step()
        if step==0: synchronize(); start=time.perf_counter()
        if step % 50 == 0 or step == CFG.vae_steps-1:
            history.append({"step":step+1,"loss":float(loss.detach()),"reconstruction":float(rec_loss.detach()),"kl":float(kl.detach()),"beta":beta})
    return time.perf_counter()-start,peak_memory_mb(),history


def train_diffusion(model, train, diffusion, total_steps, ema_decay, encoder=None, latent_stats=None):
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG.diffusion_lr,weight_decay=CFG.generative_weight_decay)
    scheduler=cosine_with_warmup(optimizer,total_steps); scaler=torch.amp.GradScaler("cuda",enabled=USE_GRAD_SCALER)
    ema=copy.deepcopy(model).eval(); iterator=iter(train); history=[]; reset_peak_memory(); start=None
    for step in range(total_steps):
        try: x,_=next(iterator)
        except StopIteration: iterator=iter(train); x,_=next(iterator)
        x=x.to(DEVICE,non_blocking=True)
        if encoder:
            with torch.no_grad(): x=encoder.encode(x)[0]; x=(x-latent_stats[0])/latent_stats[1]
        else: x=x*2-1
        t=torch.randint(0,diffusion.steps,(len(x),),device=DEVICE); noisy,eps=diffusion.noise(x,t)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(): loss=F.mse_loss(model(noisy,t),eps)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip)
        previous_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()>=previous_scale: scheduler.step()
        with torch.no_grad():
            for ema_parameter,parameter in zip(ema.parameters(),model.parameters()):
                ema_parameter.lerp_(parameter,1-ema_decay)
        if step==0: synchronize(); start=time.perf_counter()
        if step % 50 == 0 or step == total_steps-1:
            history.append({"step":step+1,"noise_mse":float(loss.detach())})
    return ema,time.perf_counter()-start,peak_memory_mb(),history


@torch.no_grad()
def latent_stats(vae, train):
    values=[]
    for x,_ in train: values.append(vae.encode(x.to(DEVICE))[0].cpu())
    z=torch.cat(values); return z.mean((0,2,3),keepdim=True).to(DEVICE),z.std((0,2,3),keepdim=True).clamp_min(1e-4).to(DEVICE)


@torch.no_grad()
def outputs(classifier, images):
    ps,fs=[],[]
    for x in images.split(CFG.classifier_batch_size):
        with amp_context(): logits,f=classifier(x.to(DEVICE),True)
        ps.append(logits.float().softmax(1).cpu()); fs.append(f.float().cpu())
    return torch.cat(ps).numpy(),torch.cat(fs).numpy()


def fid(a,b):
    ma,mb=a.mean(0),b.mean(0); ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False); root=sqrtm(ca@cb)
    if np.iscomplexobj(root): root=root.real
    return float((ma-mb)@(ma-mb)+np.trace(ca+cb-2*root))


def kid(a,b,seed):
    rng=np.random.default_rng(seed); vals=[]; n=min(CFG.kid_subset,len(a),len(b)); d=a.shape[1]
    for _ in range(CFG.kid_repeats):
        x=a[rng.choice(len(a),n,False)]; y=b[rng.choice(len(b),n,False)]; kxx=(x@x.T/d+1)**3; kyy=(y@y.T/d+1)**3; kxy=(x@y.T/d+1)**3
        vals.append((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(n*(n-1))-2*kxy.mean())
    return float(np.mean(vals)),float(np.std(vals,ddof=1))


def save_grid(images,path,title):
    path.parent.mkdir(parents=True,exist_ok=True); grid=make_grid(images[:64],8,padding=2); plt.figure(figsize=(8,8)); plt.imshow(grid.permute(1,2,0).squeeze(),cmap="gray"); plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()


def evaluate(name, samples, classifier, real_features, real_distribution, train_s, sample_s, params,
             training_peak_gpu, sampling_peak_gpu, history, seed):
    p,f=outputs(classifier,samples); km,ks=kid(real_features,f,seed); counts=np.bincount(p.argmax(1),minlength=p.shape[1])/len(p); ent=float(-(counts[counts>0]*np.log(counts[counts>0])).sum()/np.log(len(counts)))
    gray=(.299*samples[:,0:1]+.587*samples[:,1:2]+.114*samples[:,2:3]).float()
    sobel_x=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)
    sobel_y=sobel_x.transpose(-1,-2)
    gx=F.conv2d(gray,sobel_x,padding=1); gy=F.conv2d(gray,sobel_y,padding=1)
    sobel_score=torch.sqrt(gx.square()+gy.square()+1e-12).mean()
    py=p.mean(0); inception=float(np.exp(np.mean(np.sum(p*(np.log(p+1e-12)-np.log(py+1e-12)),axis=1))))
    rng=np.random.default_rng(seed); ix=rng.integers(0,len(f),size=(min(10000,len(f)*2),2)); pair=float(np.linalg.norm(f[ix[:,0]]-f[ix[:,1]],axis=1).mean())
    midpoint=.5*(counts+real_distribution)
    jsd=.5*np.sum(counts*np.log((counts+1e-12)/(midpoint+1e-12)))+.5*np.sum(real_distribution*np.log((real_distribution+1e-12)/(midpoint+1e-12)))
    loss_key="loss" if "loss" in history[-1] else "noise_mse"
    losses=np.asarray([entry[loss_key] for entry in history],dtype=float)
    parameter_total=params if isinstance(params,int) else sum(x.numel() for x in params if x.requires_grad)
    return {"model":name,"custom_fid":fid(real_features,f),"custom_kid_mean":km,"custom_kid_std":ks,"mean_classifier_confidence":float(p.max(1).mean()),"median_classifier_confidence":float(np.median(p.max(1))),"recognizability_rate_at_0.8":float((p.max(1)>=.8).mean()),"class_coverage":int((counts>0).sum()),"class_distribution_entropy":ent,"class_distribution_jsd":float(jsd),"inception_score":inception,"feature_pairwise_distance":pair,"sobel_sharpness":float(sobel_score.item()),"training_minutes":train_s/60,"sampling_total_seconds":sample_s,"sampling_ms_per_image":sample_s/len(samples)*1000,"trainable_parameters":parameter_total,"training_peak_gpu_memory_mb":training_peak_gpu,"sampling_peak_gpu_memory_mb":sampling_peak_gpu,"peak_gpu_memory_mb":max(training_peak_gpu,sampling_peak_gpu),"final_training_loss":float(losses[-1]),"training_tail_cv":float(losses[-min(10,len(losses)):].std()/(abs(losses[-min(10,len(losses)):].mean())+1e-12)),"nonfinite_loss_count":int((~np.isfinite(losses)).sum()),"class_distribution":counts.tolist()}


def prepare_evaluator(dataset_name):
    train_ds,test_ds,classes=load_datasets(dataset_name); seed_all(1508)
    epochs=CFG.epochs_classifier_fmnist if dataset_name=="fashion_mnist" else CFG.epochs_classifier_fpis
    learning_rate=CFG.classifier_lr_fmnist if dataset_name=="fashion_mnist" else CFG.classifier_lr_fpis
    evaluator_config={"architecture":"residual_64_128_256_v1","seed":1508,"epochs":epochs,
                      "batch_size":CFG.classifier_batch_size,"learning_rate":learning_rate,
                      "weight_decay":CFG.classifier_weight_decay,"classes":classes}
    evaluator_dir=OUT/"evaluators"; evaluator_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=evaluator_dir/f"{dataset_name}_classifier.pt"; metrics_path=evaluator_dir/f"{dataset_name}_metrics.json"
    if checkpoint.exists() and metrics_path.exists():
        stored=json.loads(metrics_path.read_text())
        if stored.get("evaluator_config")==evaluator_config:
            classifier=Classifier(len(classes)).to(DEVICE); classifier.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True))
            keys=("accuracy","balanced_accuracy","test_loss","training_seconds","peak_gpu_memory_mb")
            return train_ds,test_ds,classes,classifier.eval(),{key:stored[key] for key in keys}
    train_loader=loader(train_ds,True,CFG.classifier_batch_size)
    test_loader=loader(test_ds,False,CFG.classifier_batch_size)
    classifier=Classifier(len(classes)).to(DEVICE)
    metrics=train_classifier(classifier,train_loader,test_loader,dataset_name)
    confusion,normalized_confusion,recall=classifier_diagnostics(classifier,test_loader,len(classes))
    torch.save(classifier.state_dict(),checkpoint)
    metrics_path.write_text(json.dumps({"classes":classes,"evaluator_config":evaluator_config,**metrics},indent=2))
    pd.DataFrame(confusion,index=classes,columns=classes).to_csv(evaluator_dir/f"{dataset_name}_confusion_matrix.csv")
    pd.DataFrame(normalized_confusion,index=classes,columns=classes).to_csv(evaluator_dir/f"{dataset_name}_confusion_matrix_normalized.csv")
    pd.DataFrame({"class":classes,"recall":recall}).to_csv(evaluator_dir/f"{dataset_name}_per_class_recall.csv",index=False)
    return train_ds,test_ds,classes,classifier.eval(),metrics


@torch.no_grad()
def real_reference(test_ds,classifier,n_classes):
    n=min(CFG.eval_samples,len(test_ds)); images=torch.stack([test_ds[i][0] for i in range(n)])
    probabilities,features=outputs(classifier,images)
    distribution=np.bincount(probabilities.argmax(1),minlength=n_classes)/len(probabilities)
    return images,features,distribution


def run_one(dataset_name,seed,train_ds,test_size,classes,classifier,classifier_metrics,real,real_features,real_distribution):
    run=OUT/dataset_name/f"seed_{seed:02d}"; result=run/"results.json"
    config_snapshot={**json.loads(json.dumps(asdict(CFG))),"source_sha256":SOURCE_SHA256}
    if result.exists():
        cached=json.loads(result.read_text())
        if cached.get("config")==config_snapshot: return cached
        raise RuntimeError(f"Refusing to mix configurations in existing run directory: {run}")
    run.mkdir(parents=True,exist_ok=True); seed_all(seed)
    tr=loader(train_ds,True,CFG.generative_batch_size,drop=True)
    n=CFG.eval_samples

    print("Training spatial VAE...",flush=True)
    vae=VAE(CFG.latent_channels).to(DEVICE); vt,vmem,vhistory=train_vae(vae,tr)
    vae_parameters=sum(parameter.numel() for parameter in vae.parameters() if parameter.requires_grad)
    vae.eval()
    with torch.inference_mode(),amp_context():
        warmup=vae.decode(torch.randn(n,CFG.latent_channels,8,8,device=DEVICE))
    del warmup; synchronize(); torch.cuda.empty_cache(); reset_peak_memory(); start=time.perf_counter()
    with torch.inference_mode(),amp_context():
        vs=vae.decode(torch.randn(n,CFG.latent_channels,8,8,device=DEVICE)).float().cpu()
    synchronize(); vst=time.perf_counter()-start; vsmem=peak_memory_mb()
    vae.to("cpu"); torch.cuda.empty_cache()

    print(f"VAE complete ({vt/60:.2f} min); training pixel DDPM...",flush=True)
    diffusion=Diffusion(CFG.diffusion_steps); ddpm=PixelUNet().to(DEVICE)
    ddpm_ema,dt,dmem,dhistory=train_diffusion(ddpm,tr,diffusion,CFG.ddpm_steps,CFG.ddpm_ema_decay)
    ddpm_parameters=sum(parameter.numel() for parameter in ddpm.parameters() if parameter.requires_grad)
    del ddpm
    with torch.inference_mode():
        for size in sorted({CFG.sampling_batch_size,n%CFG.sampling_batch_size} - {0}):
            with amp_context(): warmup=ddpm_ema(torch.randn(size,3,32,32,device=DEVICE),torch.zeros(size,dtype=torch.long,device=DEVICE))
            del warmup
    synchronize(); torch.cuda.empty_cache(); reset_peak_memory(); start=time.perf_counter(); all_dd=[]
    for i in range(0,n,CFG.sampling_batch_size):
        size=min(CFG.sampling_batch_size,n-i)
        all_dd.append(((diffusion.sample(ddpm_ema,(size,3,32,32),CFG.ddpm_clean_clip)+1)/2).clamp(0,1).cpu())
    synchronize(); ds=torch.cat(all_dd); dst=time.perf_counter()-start; dsmem=peak_memory_mb()
    del all_dd,ddpm_ema; torch.cuda.empty_cache()

    print(f"DDPM complete ({dt/60:.2f} min); training latent diffusion...",flush=True)
    vae.to(DEVICE)
    zs=latent_stats(vae,tr); ldm=LatentDenoiser(CFG.latent_channels).to(DEVICE)
    ldm_ema,lt,lmem,lhistory=train_diffusion(ldm,tr,diffusion,CFG.ldm_steps,CFG.ldm_ema_decay,vae,zs)
    ldm_parameters=vae_parameters+sum(parameter.numel() for parameter in ldm.parameters() if parameter.requires_grad)
    del ldm; vae.eval()
    with torch.inference_mode():
        for size in sorted({CFG.sampling_batch_size,n%CFG.sampling_batch_size} - {0}):
            latent=torch.randn(size,CFG.latent_channels,8,8,device=DEVICE)
            with amp_context(): warmup_noise=ldm_ema(latent,torch.zeros(size,dtype=torch.long,device=DEVICE)); warmup_decode=vae.decode(latent*zs[1]+zs[0])
            del latent,warmup_noise,warmup_decode
    synchronize(); torch.cuda.empty_cache(); reset_peak_memory(); start=time.perf_counter(); all_ld=[]
    for i in range(0,n,CFG.sampling_batch_size):
        z=diffusion.sample(ldm_ema,(min(CFG.sampling_batch_size,n-i),CFG.latent_channels,8,8),CFG.ldm_clean_clip)
        with torch.inference_mode(),amp_context(): decoded=vae.decode(z*zs[1]+zs[0])
        all_ld.append(decoded.float().cpu())
    synchronize(); ls=torch.cat(all_ld); lst=time.perf_counter()-start; lsmem=peak_memory_mb()
    del all_ld,ldm_ema; vae.to("cpu"); torch.cuda.empty_cache()

    print(f"LDM complete ({lt/60:.2f} min); evaluating 3 model distributions...",flush=True)
    rows=[evaluate("VAE",vs,classifier,real_features,real_distribution,vt,vst,vae_parameters,vmem,vsmem,vhistory,seed),
          evaluate("DDPM",ds,classifier,real_features,real_distribution,dt,dst,ddpm_parameters,dmem,dsmem,dhistory,seed+10),
          evaluate("LDM",ls,classifier,real_features,real_distribution,vt+lt,lst,ldm_parameters,max(vmem,lmem),lsmem,lhistory,seed+20)]
    for name,x in [("real",real),("vae",vs),("ddpm",ds),("ldm",ls)]: save_grid(x,run/"figures"/f"{name}_samples.png",f"{dataset_name}: {name.upper()} (seed {seed})")
    logs=run/"logs"; logs.mkdir(exist_ok=True)
    pd.DataFrame(vhistory).to_csv(logs/"vae_history.csv",index=False); pd.DataFrame(dhistory).to_csv(logs/"ddpm_history.csv",index=False); pd.DataFrame(lhistory).to_csv(logs/"ldm_history.csv",index=False)
    flat_rows=[{key:value for key,value in row.items() if key!="class_distribution"} for row in rows]
    pd.DataFrame(flat_rows).to_csv(run/"model_metrics.csv",index=False)
    class_frame=pd.DataFrame({"class":classes,"real":real_distribution,**{row["model"].lower():row["class_distribution"] for row in rows}})
    class_frame.to_csv(run/"class_distribution.csv",index=False)
    payload={"dataset":dataset_name,"seed":seed,"classes":classes,"train_size":len(train_ds),"test_size":test_size,
             "real_reference_samples":len(real),"generated_samples_per_model":n,"config":config_snapshot,
             "classifier":classifier_metrics,"models":rows}; result.write_text(json.dumps(payload,indent=2)); return payload


def aggregate(results):
    records=[]
    for result in results:
        for model_metrics in result["models"]:
            flat={key:value for key,value in model_metrics.items() if key!="class_distribution"}
            records.append({"dataset":result["dataset"],"seed":result["seed"],
                            **{f"classifier_{key}":value for key,value in result["classifier"].items()},**flat})
    raw=pd.DataFrame(records)
    numeric=[column for column in raw.columns if column not in ("dataset","model")]
    aggregate_frame=raw.groupby(["dataset","model"])[numeric].agg(["mean","std"])
    figure_dir=OUT/"aggregate"/"figures"; figure_dir.mkdir(parents=True,exist_ok=True)
    raw.to_csv(OUT/"aggregate"/"all_runs.csv",index=False); aggregate_frame.to_csv(OUT/"aggregate"/"mean_std.csv")
    metrics=["custom_fid","custom_kid_mean","mean_classifier_confidence","recognizability_rate_at_0.8",
             "class_distribution_entropy","class_distribution_jsd","inception_score","feature_pairwise_distance",
             "sobel_sharpness","training_minutes","sampling_ms_per_image","peak_gpu_memory_mb"]
    for metric in metrics:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for axis,(dataset,group) in zip(axes,raw.groupby("dataset")):
            order=["VAE","DDPM","LDM"]
            means=group.groupby("model")[metric].mean().reindex(order); std=group.groupby("model")[metric].std().reindex(order)
            axis.bar(order,means,yerr=std,capsize=5,color=["#4472C4","#ED7D31","#70AD47"])
            axis.set_title(dataset.replace("_"," ").title()); axis.set_ylabel(metric.replace("_"," "))
        fig.suptitle(f"Five-seed mean ± SD: {metric.replace('_',' ')}")
        fig.tight_layout(); fig.savefig(figure_dir/f"{metric}.png",dpi=200); plt.close(fig)
    summary=["# Three-model benchmark: five-seed results","",
             "Models: VAE, pixel-space DDPM, and latent diffusion. Datasets: Fashion-MNIST and Fashion Product Images Small.","",
             "Values are mean ± sample standard deviation across seeds 0–4.",""]
    for dataset in raw.dataset.unique():
        summary += [f"## {dataset.replace('_',' ').title()}",""]
        for metric in metrics:
            values=[]
            for model in ["VAE","DDPM","LDM"]:
                series=raw[(raw.dataset==dataset)&(raw.model==model)][metric]
                values.append(f"{model}: {series.mean():.4f} ± {series.std():.4f}")
            summary.append(f"- {metric.replace('_',' ')} — "+"; ".join(values))
        summary.append("")
    (OUT/"aggregate"/"RESULTS_SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config_path=OUT/"config.json"; config_snapshot={**json.loads(json.dumps(asdict(CFG))),"source_sha256":SOURCE_SHA256}
    if config_path.exists() and json.loads(config_path.read_text())!=config_snapshot:
        raise RuntimeError(f"Output directory already contains a different configuration: {OUT}")
    config_path.write_text(json.dumps(config_snapshot,indent=2)); print(f"Device: {DEVICE}")
    results=[]
    for dataset in ("fashion_mnist","fashion_product_images"):
        train_ds,test_ds,classes,classifier,classifier_metrics=prepare_evaluator(dataset)
        real,real_features,real_distribution=real_reference(test_ds,classifier,len(classes))
        for seed in CFG.seeds:
            print(f"\n=== {dataset} / seed {seed} ===",flush=True)
            results.append(run_one(dataset,seed,train_ds,len(test_ds),classes,classifier,classifier_metrics,real,real_features,real_distribution))
        del classifier,real,real_features,real_distribution,train_ds,test_ds
        if AMP: torch.cuda.empty_cache()
    aggregate(results); print(f"\nComplete: {OUT.resolve()}")


if __name__ == "__main__": main()
