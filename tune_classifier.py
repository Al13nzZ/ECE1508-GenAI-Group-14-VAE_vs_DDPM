"""Validation-only search for the benchmark feature classifier.

The script deliberately never selects on the official test split.  It writes a
CSV and plot that document the bounded search used by the final report.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import run_three_model_color_benchmark as bench


OUT = Path("hyperparameter_search/classifier")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = bench.DEVICE

CANDIDATES = [
    {"name": "large_batch", "batch": 1024, "epochs": 20, "lr": 1e-3, "wd": 1e-5, "weighted": False},
    {"name": "balanced_adamw", "batch": 256, "epochs": 40, "lr": 3e-4, "wd": 1e-4, "weighted": False},
    {"name": "higher_lr", "batch": 256, "epochs": 40, "lr": 1e-3, "wd": 1e-4, "weighted": False},
    {"name": "class_weighted", "batch": 256, "epochs": 50, "lr": 3e-4, "wd": 1e-4, "weighted": True},
]


def labels_for(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    return np.asarray([
        dataset.classes.index(dataset.records[int(index)]["articleType"])
        for index in dataset.indices
    ])


def stratified_indices(labels, val_fraction=0.1, seed=1508):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * val_fraction))
        val.extend(indices[:n_val]); train.extend(indices[n_val:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def evaluate(model, loader, n_classes):
    model.eval(); correct = 0; total = 0
    per_correct = torch.zeros(n_classes); per_total = torch.zeros(n_classes)
    with torch.no_grad():
        for images, labels in loader:
            pred = model(images.to(DEVICE, non_blocking=True)).argmax(1).cpu()
            correct += (pred == labels).sum().item(); total += len(labels)
            for cls in range(n_classes):
                mask = labels == cls
                per_correct[cls] += ((pred == cls) & mask).sum()
                per_total[cls] += mask.sum()
    recalls = per_correct / per_total.clamp_min(1)
    return correct / total, float(recalls.mean()), float(recalls.min())


def train_candidate(dataset, labels, train_idx, val_idx, n_classes, candidate, seed=1508):
    bench.seed_all(seed)
    class_labels=np.asarray(labels,dtype=np.int64).copy()
    kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=candidate["batch"], shuffle=True, **kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1024, shuffle=False, **kwargs)
    model = bench.Classifier(n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["lr"], weight_decay=candidate["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, candidate["epochs"])
    weights = None
    if candidate["weighted"]:
        selected=class_labels[np.asarray(train_idx,dtype=np.int64)]
        if selected.min()<0 or selected.max()>=n_classes:
            raise ValueError(f"Labels outside [0,{n_classes-1}]: {selected.min()}..{selected.max()}")
        counts = np.bincount(selected, minlength=n_classes)
        # Square-root inverse frequency is less aggressive than raw inverse
        # frequency and was included specifically for the imbalanced RGB data.
        weights = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float32, device=DEVICE)
        weights /= weights.mean()
    history = []; start = time.perf_counter()
    for epoch in range(candidate["epochs"]):
        model.train()
        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True); targets = targets.to(DEVICE, non_blocking=True)
            if torch.rand((), device=DEVICE) < 0.5:
                images = torch.flip(images, (-1,))
            optimizer.zero_grad(set_to_none=True)
            with bench.amp_context():
                loss = F.cross_entropy(model(images), targets, weight=weights)
            loss.backward(); optimizer.step()
        scheduler.step()
        if epoch == candidate["epochs"] - 1 or (epoch + 1) % 5 == 0:
            accuracy, balanced, minimum = evaluate(model, val_loader, n_classes)
            history.append({"epoch": epoch + 1, "accuracy": accuracy, "balanced_accuracy": balanced, "minimum_recall": minimum})
    accuracy, balanced, minimum = evaluate(model, val_loader, n_classes)
    return {**candidate, "val_accuracy": accuracy, "val_balanced_accuracy": balanced,
            "val_minimum_recall": minimum, "selection_score": (accuracy + balanced) / 2,
            "training_seconds": time.perf_counter() - start, "history": history}


def main():
    rows = []
    for dataset_name in ("fashion_mnist", "fashion_product_images"):
        dataset, _, classes = bench.load_datasets(dataset_name)
        labels = labels_for(dataset)
        train_idx, val_idx = stratified_indices(labels)
        for candidate in CANDIDATES:
            history_path=OUT/f"{dataset_name}_{candidate['name']}_history.json"
            if history_path.exists():
                history=json.loads(history_path.read_text()); final=history[-1]
                rows.append({"dataset":dataset_name,**candidate,"val_accuracy":final["accuracy"],
                             "val_balanced_accuracy":final["balanced_accuracy"],"val_minimum_recall":final["minimum_recall"],
                             "selection_score":(final["accuracy"]+final["balanced_accuracy"])/2,"training_seconds":np.nan})
                print(f"{dataset_name}: {candidate['name']} (cached)",flush=True); continue
            print(f"{dataset_name}: {candidate['name']}", flush=True)
            result = train_candidate(dataset, labels, train_idx, val_idx, len(classes), candidate)
            history = result.pop("history")
            rows.append({"dataset": dataset_name, **result})
            history_path.write_text(json.dumps(history, indent=2))
            pd.DataFrame(rows).to_csv(OUT/"classifier_search_partial.csv",index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "classifier_search.csv", index=False)
    winners=frame.loc[frame.groupby("dataset").selection_score.idxmax()].sort_values("dataset")
    (OUT/"selected_hyperparameters.json").write_text(json.dumps(
        winners[["dataset","name","batch","epochs","lr","wd","weighted","val_accuracy",
                 "val_balanced_accuracy","val_minimum_recall","selection_score"]].to_dict("records"),indent=2))
    (OUT/"search_protocol.json").write_text(json.dumps({
        "split":"fixed stratified 90/10 split of each training set",
        "selection_seed":1508,
        "official_test_images_used_for_selection":0,
        "selection_metric":"mean of validation accuracy and validation balanced accuracy",
        "candidate_count_per_dataset":len(CANDIDATES),
    },indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (dataset, group) in zip(axes, frame.groupby("dataset")):
        positions = np.arange(len(group)); width = 0.36
        axis.bar(positions - width/2, group.val_accuracy * 100, width, label="Accuracy")
        axis.bar(positions + width/2, group.val_balanced_accuracy * 100, width, label="Balanced accuracy")
        axis.set_xticks(positions, group.name, rotation=20, ha="right")
        axis.set_ylim(max(0, min(group.val_balanced_accuracy.min(), group.val_accuracy.min())*100-5), 100)
        axis.set_title(dataset.replace("_", " ").title()); axis.set_ylabel("Validation score (%)"); axis.legend()
    fig.suptitle("Classifier hyperparameter search (held-out validation data)")
    fig.tight_layout(); fig.savefig(OUT / "classifier_search.png", dpi=220); plt.close(fig)
    print(frame.sort_values(["dataset", "selection_score"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
