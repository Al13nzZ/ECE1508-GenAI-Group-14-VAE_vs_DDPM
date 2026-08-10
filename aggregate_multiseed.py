from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "vae_ddpm_fashionmnist_multiseed_results"
AGG_DIR = RUNS_ROOT / "aggregate"
FIG_DIR = AGG_DIR / "figures"
TABLE_DIR = AGG_DIR / "tables"
PRESENTATION_DIR = ROOT / "report_presentation_package"


METRICS = [
    ("custom_fid", "Classifier-feature FID", "lower"),
    ("custom_kid_mean", "Classifier-feature KID", "lower"),
    ("mean_classifier_confidence", "Mean classifier confidence", "higher"),
    ("recognizability_rate_at_0.8", "Recognizability at 0.8", "higher"),
    ("class_distribution_entropy", "Normalized class entropy", "higher"),
    ("sobel_sharpness", "Sobel sharpness", "higher"),
    ("training_minutes", "Training time (minutes)", "lower"),
    ("sampling_ms_per_image", "Sampling time (ms/image)", "lower"),
    ("peak_gpu_memory_mb", "Peak allocated GPU memory (MB)", "lower"),
]


def save_metric_plot(frame: pd.DataFrame, metric: str, label: str) -> None:
    summary = frame.groupby("model")[metric].agg(["mean", "std"]).reindex(["VAE", "DDPM"])
    colors = ["#4C78A8", "#F58518"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=7,
                  color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel(label)
    ax.set_title(f"{label} across 10 random seeds")
    ax.grid(axis="y", alpha=0.25)
    for bar, mean, std in zip(bars, summary["mean"], summary["std"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mean:.4g} ± {std:.2g}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{metric}_multiseed.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_metric_dashboard(
    frame: pd.DataFrame,
    specifications: list[tuple[str, str]],
    title: str,
    filename: str,
    shape: tuple[int, int],
) -> None:
    fig, axes = plt.subplots(*shape, figsize=(12, 7 if shape[0] == 2 else 4.5))
    axes_array = np.asarray(axes).reshape(-1)
    colors = ["#4C78A8", "#F58518"]
    for ax, (metric, label) in zip(axes_array, specifications):
        summary = frame.groupby("model")[metric].agg(["mean", "std"]).reindex(["VAE", "DDPM"])
        bars = ax.bar(summary.index, summary["mean"], yerr=summary["std"],
                      capsize=5, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.22)
        for bar, mean, std in zip(bars, summary["mean"], summary["std"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mean:.3g} ± {std:.2g}", ha="center", va="bottom", fontsize=8)
    for ax in axes_array[len(specifications):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    classifiers = []
    class_frames = []
    for seed in range(10):
        seed_dir = RUNS_ROOT / f"seed_{seed:02d}"
        metrics = pd.read_csv(seed_dir / "tables" / "model_comparison_metrics.csv")
        metrics.insert(0, "seed", seed)
        frames.append(metrics)
        payload = json.loads((seed_dir / "results.json").read_text(encoding="utf-8"))
        classifiers.append({"seed": seed, **payload["classifier"]})
        classes = pd.read_csv(seed_dir / "tables" / "class_distribution.csv")
        classes.insert(0, "seed", seed)
        class_frames.append(classes)

    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(TABLE_DIR / "all_seed_model_metrics.csv", index=False)
    pd.DataFrame(classifiers).to_csv(TABLE_DIR / "all_seed_classifier_metrics.csv", index=False)

    numeric = [c for c in all_metrics.select_dtypes(include=[np.number]).columns if c != "seed"]
    grouped = all_metrics.groupby("model")[numeric].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.reset_index().to_csv(TABLE_DIR / "model_metrics_mean_std_min_max.csv", index=False)

    for metric, label, _ in METRICS:
        save_metric_plot(all_metrics, metric, label)

    save_metric_dashboard(
        all_metrics,
        [
            ("custom_fid", "Classifier-feature FID ↓"),
            ("custom_kid_mean", "Classifier-feature KID ↓"),
            ("mean_classifier_confidence", "Mean confidence ↑"),
            ("recognizability_rate_at_0.8", "Recognizability at 0.8 ↑"),
            ("class_distribution_entropy", "Class entropy ↑"),
            ("sobel_sharpness", "Sobel sharpness ↑"),
        ],
        "Generation quality across 10 random seeds (mean ± SD)",
        "quality_metrics_multiseed.png",
        (2, 3),
    )
    save_metric_dashboard(
        all_metrics,
        [
            ("training_minutes", "Training time (minutes) ↓"),
            ("sampling_ms_per_image", "Sampling latency (ms/image) ↓"),
            ("peak_gpu_memory_mb", "Peak allocated memory (MB) ↓"),
        ],
        "Computational cost across 10 random seeds (mean ± SD)",
        "compute_metrics_multiseed.png",
        (1, 3),
    )

    classes = pd.concat(class_frames, ignore_index=True)
    class_summary = classes.groupby(["class_index", "class_name"])[
        ["real_test_distribution", "vae_distribution", "ddpm_distribution"]
    ].agg(["mean", "std"])
    class_summary.columns = [f"{name}_{stat}" for name, stat in class_summary.columns]
    class_summary = class_summary.reset_index()
    class_summary.to_csv(TABLE_DIR / "class_distribution_mean_std.csv", index=False)

    x = np.arange(10)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - width, class_summary["real_test_distribution_mean"], width,
           yerr=class_summary["real_test_distribution_std"], capsize=2, label="Real test")
    ax.bar(x, class_summary["vae_distribution_mean"], width,
           yerr=class_summary["vae_distribution_std"], capsize=2, label="VAE")
    ax.bar(x + width, class_summary["ddpm_distribution_mean"], width,
           yerr=class_summary["ddpm_distribution_std"], capsize=2, label="DDPM")
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1, label="Uniform")
    ax.set_xticks(x, class_summary["class_name"], rotation=35, ha="right")
    ax.set_ylabel("Predicted class proportion")
    ax.set_title("Class distribution across 10 random seeds (mean ± SD)")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution_multiseed.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    classifier_frame = pd.DataFrame(classifiers)
    classifier_summary = {
        "test_accuracy_mean": float(classifier_frame["test_accuracy"].mean()),
        "test_accuracy_std": float(classifier_frame["test_accuracy"].std(ddof=1)),
    }
    payload = {
        "seeds": list(range(10)),
        "n_seeds": 10,
        "classifier": classifier_summary,
        "models": grouped.reset_index().to_dict(orient="records"),
    }
    (AGG_DIR / "aggregate_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selected = [
        "custom_fid_multiseed.png", "custom_kid_mean_multiseed.png",
        "mean_classifier_confidence_multiseed.png",
        "recognizability_rate_at_0.8_multiseed.png",
        "class_distribution_entropy_multiseed.png", "sobel_sharpness_multiseed.png",
        "training_minutes_multiseed.png", "sampling_ms_per_image_multiseed.png",
        "peak_gpu_memory_mb_multiseed.png", "class_distribution_multiseed.png",
        "quality_metrics_multiseed.png", "compute_metrics_multiseed.png",
    ]
    target = PRESENTATION_DIR / "figures" / "multiseed"
    target.mkdir(parents=True, exist_ok=True)
    for name in selected:
        shutil.copy2(FIG_DIR / name, target / name)
    shutil.copy2(TABLE_DIR / "model_metrics_mean_std_min_max.csv",
                 PRESENTATION_DIR / "data" / "multiseed_model_metrics.csv")
    shutil.copy2(TABLE_DIR / "all_seed_model_metrics.csv",
                 PRESENTATION_DIR / "data" / "all_seed_model_metrics.csv")
    print(f"Aggregate outputs written to {AGG_DIR}")


if __name__ == "__main__":
    main()
