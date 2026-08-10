# VAE vs DDPM on Fashion-MNIST — Generated Results

## Experimental setup

- Profile: `report`
- Device: `cuda`
- Classifier test accuracy: **91.17%**
- Classifier test loss: **0.2460**
- Evaluation samples per model: **5,000**
- DDPM diffusion steps: **100**
- VAE latent dimension: **32**

## Main quantitative results

| model   |   custom_fid |   custom_kid_mean |   mean_classifier_confidence |   class_coverage |   class_distribution_entropy |   nearest_neighbor_mean_distance |   memorization_rate_below_real_1pct_threshold |   training_minutes |   sampling_ms_per_image |
|:--------|-------------:|------------------:|-----------------------------:|-----------------:|-----------------------------:|---------------------------------:|----------------------------------------------:|-------------------:|------------------------:|
| VAE     |      39.8975 |            1.7070 |                       0.7230 |               10 |                       0.9527 |                           1.9985 |                                        0.0000 |             0.4224 |                  0.0078 |
| DDPM    |       8.5679 |            0.3828 |                       0.8436 |               10 |                       0.9817 |                           1.4514 |                                        0.0000 |             4.9058 |                  8.4029 |

## Automatically derived comparisons

- Lowest classifier-feature FID: **DDPM**
- Lowest classifier-feature KID: **DDPM**
- Highest mean classifier confidence: **DDPM**
- Highest class-distribution entropy: **DDPM**
- Shortest measured training time: **VAE**
- Fastest measured sampling: **VAE**
- VAE test negative-ELBO components: total **238.423**, reconstruction **222.196**, KL **16.227**

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
