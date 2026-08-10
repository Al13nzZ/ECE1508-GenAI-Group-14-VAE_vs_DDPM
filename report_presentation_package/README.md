# Report and Presentation Package

This package contains the evidence worth using in the final report and a
five-slide presentation. It deliberately excludes checkpoints, generated
tensor files, and redundant plots.

## Recommended main-report figures

1. `figures/main/real_vae_ddpm_comparison.png` — representative qualitative
   result; label it explicitly because images cannot be averaged across seeds.
2. `figures/main/vae_latent_interpolation.png` and
   `figures/main/ddpm_denoising_trajectory.png` — generation mechanisms.
3. `figures/multiseed/quality_metrics_multiseed.png` — primary quantitative
   quality figure; all six panels report ten-seed mean ± sample SD.
4. `figures/multiseed/class_distribution_multiseed.png` — averaged diversity
   evidence with per-class uncertainty bars.
5. `figures/multiseed/compute_metrics_multiseed.png` — primary efficiency
   figure with training, sampling, and memory results across ten seeds.

Use the dashboards in the main report and slides. Individual multi-seed plots
remain available when a larger single-metric figure is required.

## Appendix-only figures

- training curves: stability/convergence evidence;
- nearest-neighbor grids: memorization screen;
- lowest-confidence samples: transparent failure-case analysis;
- classifier confusion matrix: evaluator limitation/context.

## Key numbers

- Evaluation classifier accuracy: 91.37% ± 0.20%.
- DDPM classifier-feature FID: 11.14 ± 5.79; VAE: 43.32 ± 4.51.
- DDPM classifier-feature KID: 0.530 ± 0.342; VAE: 1.807 ± 0.268.
- Confidence >= 0.8: DDPM 67.11% ± 2.93%; VAE 43.37% ± 1.59%.
- Class entropy: DDPM 0.971 ± 0.028; VAE 0.941 ± 0.013; both cover 10/10 classes.
- Training: DDPM 4.811 ± 0.059 min; VAE 0.411 ± 0.020 min.
- Batched sampling: DDPM 8.173 ± 0.110 ms/image; VAE 0.00784 ± 0.00062 ms/image.
- Peak allocated GPU memory: DDPM 6275 MB; VAE 558 MB.
- Parameters: DDPM 5.22M; VAE 1.70M.

Use the exact CSV values for tables; rounded values above are for prose and
slides.
