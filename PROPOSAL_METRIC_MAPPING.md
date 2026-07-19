# Proposal-to-Code Metric Mapping

| Proposal area | Implementation and generated output |
|---|---|
| Sample quality | Equal-size real, VAE, and DDPM sample grids; low-confidence failure-case grids |
| Distribution similarity | Classifier-feature FID and KID computed from the same real reference set and the same number of generated samples |
| Classifier recognizability | Fashion-MNIST classifier validation accuracy, generated mean/median confidence, and confidence ≥ 0.8 rate |
| Diversity | Class coverage, normalized class-distribution entropy, predicted class histogram, feature pairwise distance |
| Generation structure | VAE reconstruction figure, VAE latent interpolation, DDPM reverse-denoising trajectory |
| Memorization | Feature-space nearest-neighbor pairs, average/median distance, and generated rate below a calibrated real-image 1st-percentile threshold |
| Training stability | Classifier/VAE/DDPM loss curves, final loss, last-20% coefficient of variation, loss slope, non-finite loss count |
| Compute cost | Training minutes, total sampling seconds, milliseconds per image, trainable parameters, peak GPU memory |
| Supporting evidence | Sobel edge-sharpness proxy and Fashion-classifier inception-score proxy |

The main report table is saved as `tables/model_comparison_metrics.csv`. Every figure is saved at high resolution in `figures/`.
