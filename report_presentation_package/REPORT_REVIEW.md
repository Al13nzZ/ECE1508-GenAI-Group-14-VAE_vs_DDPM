# Draft Review and Recommended Edits

## Fix before submission

1. Rename `Assentation of Teamwork` to `Statement of Team Contributions`.
2. Fix `shared evaluatioGenAIn pipeline` to `shared evaluation pipeline`.
3. In the Introduction, cite Fashion-MNIST as `[3]`, not `[2]`.
4. Ensure the figure directory in the LaTeX project matches the packaged
   paths, including `figures/main/`, `figures/multiseed/`, and
   `figures/appendix/`.
5. State that timing values are from one hardware/software run and depend on
   batching and implementation. This is especially important for the very low
   VAE per-image sampling time.
6. Keep saying `classifier-feature FID/KID`; do not shorten these to standard
   `FID/KID` in conclusions or slide labels.
7. Add the class-distribution plot to the main quantitative section. It is the
   evidence for the claim that DDPM output is more balanced.
8. Use `quality_metrics_multiseed.png`, `class_distribution_multiseed.png`, and
   `compute_metrics_multiseed.png` as the main quantitative figures. They show
   ten-seed mean ± sample SD and directly support the central conclusion.

## Suggested concise replacement for the teamwork section

> Jenny led the VAE implementation, training pipeline, and latent-space
> experiments. Yuye led the DDPM implementation, diffusion training and
> sampling pipeline, and denoising visualization. Jasmine led the shared
> evaluation pipeline, metric analysis, comparison, and report integration.
> All members contributed to experimental design, result interpretation, and
> final review.

Only use this wording if it accurately reflects the team's work.

## What not to add

- Do not add every generated bar chart to the paper; the main numerical table
  already communicates FID, KID, confidence, and entropy efficiently.
- Do not claim that DDPM is universally superior. It wins on this
  implementation's quality/diversity metrics and loses decisively on cost.
- Do not claim that the nearest-neighbor screen proves absence of memorization.
- Do not compare these classifier-feature FID/KID numbers with standard
  ImageNet Inception scores from other papers.
- Do not compare VAE ELBO and DDPM noise MSE numerically; they are different
  objectives.

## Optional stronger limitations sentence

> Because the comparison uses ten seeds but non-parameter-matched models, the
> reported variability strengthens the system comparison without establishing
> a universal ranking of VAEs and diffusion models.
