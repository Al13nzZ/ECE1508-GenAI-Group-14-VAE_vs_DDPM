# VAE vs DDPM on Fashion-MNIST

An end-to-end comparison of a convolutional Variational Autoencoder (VAE) and
a Denoising Diffusion Probabilistic Model (DDPM) trained under the same
Fashion-MNIST experimental setting. The project covers training, sampling,
quantitative evaluation, qualitative analysis, checkpointing, and automatic
report generation.

The original project proposal and literature references are available in
[PROJECT_ABSTRACT.md](PROJECT_ABSTRACT.md).

## Results

The included report-profile run used 60,000 training images, 10,000 test
images, and 5,000 generated evaluation samples per model. It was executed on
an NVIDIA GeForce RTX 5070 Ti using PyTorch 2.11.0 with CUDA 12.8.

| Metric | VAE | DDPM |
| --- | ---: | ---: |
| Custom classifier-feature FID (lower is better) | 51.1207 | **7.6580** |
| Custom classifier-feature KID (lower is better) | 1.8737 | **0.3257** |
| Mean classifier confidence | 0.7040 | **0.8322** |
| Recognizability rate at 0.8 confidence | 41.0% | **65.6%** |
| Class coverage | 10/10 | 10/10 |
| Normalized class-distribution entropy | 0.9310 | **0.9911** |
| Training time | **0.37 min** | 4.76 min |
| Sampling time per image | **0.0075 ms** | 8.0151 ms |

The Fashion-MNIST evaluation classifier reached **91.34% test accuracy**.
FID and KID are computed from its 128-dimensional features rather than from an
ImageNet Inception network, making them more appropriate for 28x28 grayscale
fashion images. They should therefore be interpreted as dataset-specific
metrics, not directly compared with standard ImageNet FID/KID scores.

See [RESULTS_SUMMARY.md](vae_ddpm_fashionmnist_results/RESULTS_SUMMARY.md) for
the generated interpretation and
[model_comparison_metrics.csv](vae_ddpm_fashionmnist_results/tables/model_comparison_metrics.csv)
for all recorded metrics.

## Repository contents

- `complete_project.py` - complete command-line experiment
- `VAE_vs_DDPM_FashionMNIST_Complete.ipynb` - notebook version
- `run.ps1` - Windows launcher with profile selection
- `vae_ddpm_fashionmnist_results/` - complete generated results, figures,
  histories, samples, feature arrays, and trained checkpoints
- `PROPOSAL_METRIC_MAPPING.md` - mapping between proposal goals and metrics
- `PROJECT_ABSTRACT.md` - original project abstract and references

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For RTX 50-series GPUs, install a Blackwell-compatible CUDA wheel before the
remaining requirements:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scipy matplotlib tqdm tabulate
```

Fashion-MNIST is downloaded automatically on the first run.

## Running the experiment

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Profile report
```

Available profiles are:

- `quick` - end-to-end smoke test
- `standard` - balanced runtime and quality
- `report` - full 35-epoch DDPM run with 5,000 evaluation samples

Alternatively:

```powershell
$env:EXPERIMENT_PROFILE = "report"
python complete_project.py
```

On GPUs with at least 14 GB VRAM, the code automatically enables larger batch
sizes, FP16 autocast, TF32, pinned-memory prefetching, and cuDNN autotuning.
Batch sizes can be overridden with `CLASSIFIER_BATCH_SIZE`, `VAE_BATCH_SIZE`,
and `DDPM_BATCH_SIZE`.

Existing profile-specific checkpoints are reused automatically. Delete or move
the relevant checkpoint only when a fresh training run is required.

## Evaluation outputs

The pipeline generates:

- real/VAE/DDPM image grids and failure-case figures
- classifier-feature FID and KID
- recognizability, confidence, coverage, and class-entropy statistics
- VAE interpolation and DDPM denoising trajectories
- nearest-neighbor memorization analysis
- sharpness and feature-diversity proxies
- training curves, timing, parameter counts, and peak GPU memory
- CSV/JSON results and trained model checkpoints

