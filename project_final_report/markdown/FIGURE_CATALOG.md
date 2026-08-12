# Report figure catalog

All plots are generated at publication resolution by
`generate_report_assets.py`. Unless a caption explicitly says “seed 0,” result
figures report the mean and sample standard deviation across seeds 0–4.

## Core figures used in the main report

| Figure | Purpose | Interpretation guardrail |
|---|---|---|
| `model_architectures.png` | Shows VAE, pixel-DDPM, and LDM sampling paths and tensor sizes | Conceptual diagram; not a measured result |
| `classifier_confusion_matrices.png` | Discloses frozen evaluator errors on both datasets | Generator scores inherit these errors |
| `classifier_hyperparameter_search.png` | Documents validation accuracy/balanced-accuracy search | Uses training-derived validation data only |
| `generator_hyperparameter_search.png` | Documents bounded generator pilot | Pilot is under-trained and used only for selection |
| `ldm_capacity_sampler_diagnostic.png` | Diagnoses LDM width, sampler variance, and exploding latent scale | Validation-only diagnostic |
| `ldm_clipping_search.png` | Selects the clean-latent clipping threshold | Validation-only; FID is the selection metric |
| `ddpm_sampler_search.png` | Compares pixel-DDPM reverse updates | Validation-only; chosen independently of LDM |
| `distribution_quality.png` | Five-seed feature FID and KID | Values are task-feature metrics, not ImageNet FID/KID |
| `qualitative_fashion_mnist.png` | Real/VAE/DDPM/LDM grids on Fashion-MNIST | Representative seed 0; use with multi-seed metrics |
| `qualitative_fashion_product_images.png` | Equivalent color product grids | Representative seed 0; use with multi-seed metrics |
| `class_distributions.png` | Average real/generated predicted class proportions | Evaluator-predicted rather than ground-truth generated labels |
| `compute_cost.png` | Training time, batched sampling time, and system peak memory | Hardware-specific RTX 5070 Ti throughput measurements |
| `quality_compute_tradeoff.png` | Feature FID against amortized batched sampling time | Compare within each dataset only; not single-image latency |
| `training_curves.png` | Mean and one-SD convergence curves | VAE ELBO and diffusion MSE are not cross-family comparable |
| `seed_variability.png` | Every seed's feature FID | Exposes instability hidden by an average |

## Supporting and appendix figures

| Figure | Purpose |
|---|---|
| `classifier_performance.png` | Official test accuracy and balanced accuracy |
| `classifier_per_class_recall.png` | Rare/difficult class behavior |
| `dataset_class_balance.png` | Ground-truth test class proportions |
| `diversity.png` | Normalized entropy and class-distribution JSD |
| `supporting_quality.png` | Confidence, high-confidence rate, and Sobel sharpness |
| `memory_breakdown.png` | Training and sampling memory peaks separately |
| `model_size.png` | Complete-system trainable parameter counts |

## Machine-readable companions

- `metrics_means.csv`: mean of each numeric metric by dataset/model
- `classifier_summary.csv`: final evaluator scores and metadata
- `../tables/all_seed_metrics.csv`: every raw metric for all 30 model runs
- `../tables/all_metrics_mean_std.csv`: mean and sample SD for every numeric metric
- `../tables/RESULTS_SUMMARY.md`: readable five-seed summary of principal metrics
- `../tables/quality_results.tex`: five-seed quality table used by LaTeX
- `../tables/compute_results.tex`: five-seed compute table used by LaTeX
- `../tables/classifier_results.tex`: final evaluator table used by LaTeX
- `../tables/metric_leaders.json`: model leading each major metric per dataset
