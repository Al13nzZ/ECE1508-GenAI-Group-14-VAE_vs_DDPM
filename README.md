# VAE vs DDPM vs Latent Diffusion

Code-only repository for a five-seed generative-model benchmark across two
apparel datasets.

## Benchmark

- Models: convolutional VAE, pixel-space DDPM, and latent diffusion
- Datasets: Fashion-MNIST and Fashion Product Images Small
- Seeds: 0, 1, 2, 3, and 4
- Input: 32×32 RGB
- Evaluator: residual CNN with 256-dimensional features
- Metrics: classifier accuracy and balanced accuracy, FID, KID, confidence,
  recognizability, class coverage, normalized entropy, Inception-style score,
  feature diversity, sharpness, parameters, training time, and sampling latency

The Fashion Product Images dataset is restricted to the 15 most frequent
apparel article types and uses a deterministic stratified 85/15 split.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe run_three_model_color_benchmark.py
```

The default output directory is:

```text
results_three_models_two_datasets_5seeds/
```

Override it with:

```powershell
$env:BENCHMARK_OUTPUT_DIR = "D:\benchmark_results"
.\.venv\Scripts\python.exe run_three_model_color_benchmark.py
```

Run a one-seed, one-epoch end-to-end check with:

```powershell
$env:BENCHMARK_QUICK = "1"
.\.venv\Scripts\python.exe run_three_model_color_benchmark.py
```

The runner skips completed dataset/seed pairs, allowing interrupted experiments
to resume. Downloaded datasets, virtual environments, and generated outputs are
excluded from Git.

## RTX 5070 Ti execution

When CUDA is available, the benchmark uses FP16 autocast, pinned-memory loaders,
four persistent data workers, cuDNN benchmarking, and batch size 1024. If another
application consumes substantial VRAM, reduce `Config.batch_size` in the runner.

## Repository contents

- `run_three_model_color_benchmark.py` — complete training, sampling,
  evaluation, aggregation, and plotting pipeline
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes datasets, environments, and generated artifacts
