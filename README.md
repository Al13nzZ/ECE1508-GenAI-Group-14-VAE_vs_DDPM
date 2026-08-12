# VAE, Pixel DDPM, and Latent Diffusion for Apparel Generation

This project compares three generative systems across two apparel image
domains: a spatial convolutional variational autoencoder (VAE), a pixel-space
denoising diffusion probabilistic model (DDPM), and a latent diffusion model
(LDM). Fashion-MNIST supplies the controlled grayscale benchmark, while the 15
most frequent apparel classes from Fashion Product Images Small provide a more
imbalanced, real-world RGB setting. Five independent seeds are evaluated for
every model--dataset pair (30 trained generative models in total).

The study is designed around the practical quality--efficiency trade-off. A
frozen residual CNN evaluator supplies 256-dimensional task-specific features
for feature FID and KID, as well as recognizability and class-distribution
diagnostics. The runner also records training stability, parameters, wall-clock
training and batched sampling time, and peak allocated GPU memory. Results are always
reported as the five-seed mean and sample standard deviation.

## Experiment design

- Models: spatial VAE, pixel DDPM, and spatial LDM
- Datasets: Fashion-MNIST and Fashion Product Images Small
- Image representation: 32 x 32 RGB for all systems
- Seeds: 0, 1, 2, 3, and 4
- Samples evaluated per trained model: 5,000
- Diffusion sampler: 100 ancestral reverse steps with a cosine noise schedule
- Reverse stabilization: exact posterior variance, clean-image clipping for
  DDPM, and validation-selected standardized-latent clipping for LDM
- Optimizer: AdamW with gradient clipping, warm-up, and cosine decay
- Diffusion stabilization: exponential moving average (EMA) weights
- Evaluator: residual CNN with 256-dimensional features

Fashion-MNIST is resized and replicated across three channels so the three
models have the same input interface. The color dataset is restricted to
Apparel, center-padded without geometric distortion, resized to 32 x 32, and
split deterministically using a stratified 85/15 split.

## Installation

Python 3.10 or newer is required. A CUDA build of PyTorch is strongly
recommended for the full benchmark.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Validate the pipeline

Run the unit-level smoke suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run a short end-to-end pass through both datasets and all three models:

```powershell
$env:BENCHMARK_QUICK = "1"
$env:BENCHMARK_OUTPUT_DIR = "results_smoke_validation"
.\.venv\Scripts\python.exe run_three_model_color_benchmark.py
Remove-Item Env:BENCHMARK_QUICK
Remove-Item Env:BENCHMARK_OUTPUT_DIR
```

## Reproduce hyperparameter selection

Classifier candidates are selected on fixed, stratified validation splits from
the training data. The official test sets are never used for selection.

```powershell
.\.venv\Scripts\python.exe tune_classifier.py
```

The bounded generator pilot uses a fixed, class-balanced 1,000-image validation
subset from the Fashion-MNIST training split. It compares VAE learning-rate and
KL-weight settings and diffusion learning-rate/EMA settings. The selected
configuration is transferred unchanged to the second dataset to avoid
dataset-specific test tuning.

```powershell
.\.venv\Scripts\python.exe tune_generators.py
```

Search tables, histories, plots, and machine-readable selections are written to
`hyperparameter_search/`.

## Run the definitive benchmark

```powershell
.\.venv\Scripts\python.exe run_three_model_color_benchmark.py
```

The default output is `results_final_3models_2datasets_5seeds/`. Every completed
dataset/seed pair is saved immediately, so an interrupted run can resume. The
runner refuses to mix incompatible configurations in one output directory.
Each result also embeds the pipeline version and SHA-256 hash of the exact
runner source, preventing silent aggregation across code revisions.
Set `BENCHMARK_OUTPUT_DIR` to use another location.

## Generate report figures

After the definitive run completes:

```powershell
.\.venv\Scripts\python.exe generate_report_assets.py
```

This produces the curated, publication-resolution figure set in
`project_final_report/figures/`. The report source in
`project_final_report/report.tex` retains the original NeurIPS-style project
template. `project_final_report/TECHNICAL_REFERENCES.md` provides the
same technical references in readable IEEE format with URLs.
The validated compiled paper is included as `project_final_report/report.pdf`.

Validate the completed result tree before using it in the report:

```powershell
.\.venv\Scripts\python.exe validate_results.py
.\.venv\Scripts\python.exe validate_report.py
```

## RTX 5070 Ti execution

On the tested RTX 5070 Ti, the runner uses native BF16 autocast, batch size 256,
pinned-memory transfers, four persistent data-loader workers, prefetching, and
cuDNN kernel benchmarking. Diffusion inference also uses BF16 and a larger
4,096-image sampling batch, selected by a local throughput/VRAM check; it used
about 8.0 GB allocated memory in the 4,096-image pixel-DDPM probe. The definitive environment is recorded in
`project_final_report/EXPERIMENT_ENVIRONMENT.md`. GPU utilization can still drop temporarily
during dataset decoding, metric calculation, file output, or SciPy covariance
matrix operations; those stages are not CUDA-bound.

Timing excludes one untimed warm-up forward at every exact inference batch
shape and starts after the first fully executed optimizer update. CUDA is
synchronized at each boundary, preventing one seed from absorbing compilation
and cuDNN algorithm-selection overhead that later seeds inherit from caches.

## Metric interpretation

FID and KID are computed from the frozen apparel classifier's feature space,
not from an ImageNet Inception network. They are therefore useful for controlled
within-dataset comparisons in this repository, but their absolute values should
not be compared with standard ImageNet-FID numbers from unrelated work. Visual
grids, confidence, diversity, class-distribution JSD, and computational metrics
are reported alongside them to avoid relying on one score.

## Main files

- `run_three_model_color_benchmark.py`: training, sampling, evaluation, resume,
  aggregation, and base plotting
- `tune_classifier.py`: validation-only classifier hyperparameter comparison
- `tune_generators.py`: validation-only bounded generator pilot
- `generate_report_assets.py`: report figure generation
- `validate_results.py`: completeness, configuration, and finite-value checks
- `validate_report.py`: report citation, reference, table, and figure checks
- `tests/test_pipeline.py`: fast model and metric smoke tests
- `results_final_3models_2datasets_5seeds/`: complete audited output for all
  30 runs, including per-seed metrics, training histories, sample grids,
  evaluator diagnostics/checkpoints, and aggregate mean/SD results
- `project_final_report/`: self-contained final report source, PDF, selected
  figures, result tables, original formatting template, environment record,
  and IEEE references
