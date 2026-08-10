# VAE vs. DDPM: Ten-Seed Results

The full report-profile experiment was repeated independently for random seeds
0 through 9. Each run trained a new evaluation classifier, VAE, DDPM, and EMA
model, then generated 5,000 evaluation samples per generator. The aggregate
therefore summarizes 10 trained instances and 50,000 generated samples per
model. Values below are mean ± sample standard deviation across seeds.

| Metric | VAE | DDPM |
| --- | ---: | ---: |
| Classifier-feature FID ↓ | 43.32 ± 4.51 | **11.14 ± 5.79** |
| Classifier-feature KID ↓ | 1.807 ± 0.268 | **0.530 ± 0.342** |
| Mean classifier confidence ↑ | 0.717 ± 0.009 | **0.841 ± 0.014** |
| Recognizability at confidence 0.8 ↑ | 43.37% ± 1.59% | **67.11% ± 2.93%** |
| Class coverage | 10/10 | 10/10 |
| Normalized class entropy ↑ | 0.941 ± 0.013 | **0.971 ± 0.028** |
| Sobel sharpness ↑ | 0.549 ± 0.006 | **0.646 ± 0.059** |
| Training time ↓ | **0.411 ± 0.020 min** | 4.811 ± 0.059 min |
| Sampling latency ↓ | **0.00784 ± 0.00062 ms/image** | 8.173 ± 0.110 ms/image |
| Peak allocated GPU memory ↓ | **558 MB** | 6,275 MB |
| Parameters ↓ | **1.70M** | 5.22M |

Evaluation classifiers reached 91.37% ± 0.20% test accuracy. Every model
covered all ten predicted classes in every run, and no run contained a
non-finite training loss.

## Interpretation

The direction of the original single-seed conclusion is stable across ten
seeds: DDPM has better mean distribution similarity, recognizability,
sharpness, and class balance, while VAE has dramatically lower compute cost.
However, the DDPM FID standard deviation is substantial, so the exact size of
its quality advantage depends more strongly on initialization and training
order than the single-seed report suggested.

The DDPM averages approximately 11.7 times the VAE training time, 1,043 times
the VAE sampling latency, 11.2 times the peak allocated memory, and 3.1 times
the parameter count under the fixed RTX 5070 Ti configuration.

These are descriptive statistics across ten seeds. Formal paired tests,
confidence intervals, and parameter-matched experiments remain future work.

## Files

- `tables/all_seed_model_metrics.csv`: every model metric for every seed.
- `tables/all_seed_classifier_metrics.csv`: evaluator results for every seed.
- `tables/model_metrics_mean_std_min_max.csv`: aggregate model statistics.
- `tables/class_distribution_mean_std.csv`: aggregate per-class proportions.
- `figures/*_multiseed.png`: mean plots with sample-standard-deviation bars.
- `../seed_00` through `../seed_09`: complete auditable per-seed outputs.

