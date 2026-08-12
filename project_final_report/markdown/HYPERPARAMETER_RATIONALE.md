# Hyperparameter rationale and search evidence

## What “best” means here

The selected settings are the best candidates found by the repository's
predeclared, bounded searches under one RTX 5070 Ti compute budget. They are not
claimed to be global optima. Both searches use validation images derived from
the training split; zero official test images are used for selection. Final
claims come only from the independent five-seed test benchmark.

Random/bounded search is appropriate when exhaustive joint tuning is too
expensive and only a small number of dimensions are expected to dominate
performance [Bergstra and Bengio](https://www.jmlr.org/papers/v13/bergstra12a.html).

## Classifier selection

Each dataset uses a fixed, stratified 90/10 training/validation split with seed
1508. The selection score is the arithmetic mean of validation accuracy and
validation balanced accuracy.

| Dataset | Selected candidate | Batch | Epochs | LR | Weight decay | Validation accuracy | Balanced accuracy | Minimum recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `balanced_adamw` | 256 | 40 | 3e-4 | 1e-4 | 94.57% | 94.57% | 82.00% |
| Color products | `higher_lr` | 256 | 40 | 1e-3 | 1e-4 | 88.76% | 80.85% | 25.00% |

The color search included a square-root inverse-frequency loss. It increased
minimum validation recall from 25.00% to 41.67% and slightly increased balanced
accuracy, but reduced overall accuracy enough that its predefined combined
score (84.60%) was below the selected unweighted candidate (84.81%). This
trade-off is retained in the search table rather than hidden. The final report
therefore discloses test balanced accuracy, class recall, and confusion matrices
alongside raw accuracy. Class weighting was motivated by work on long-tailed
recognition, including [Cui et al.](https://arxiv.org/abs/1901.05555).

After selection, the evaluators were retrained on their complete training
splits and frozen. The final Fashion-MNIST evaluator reaches 94.23% official
test accuracy and 94.23% balanced accuracy; its lowest class recall is 79.90%
for Shirt. The color-product evaluator reaches 88.05% test accuracy but 77.40%
balanced accuracy, with 31.82% recall for the rare Sweaters class. Generator
metrics derived from this evaluator are therefore interpreted together with
its confusion matrix and per-class recall, especially on the imbalanced set.

The residual classifier uses 64/128/256-channel stages and a 256-dimensional
feature vector. Residual connections support optimization of deeper feature
extractors [He et al.](https://arxiv.org/abs/1512.03385). AdamW separates weight
decay from the adaptive gradient update
[Loshchilov and Hutter](https://arxiv.org/abs/1711.05101), and cosine decay is
based on SGDR's cosine schedule
[Loshchilov and Hutter](https://arxiv.org/abs/1608.03983).

## Generator selection

The generator pilot uses Fashion-MNIST seed 1508, 59,000 generator-training
images, and a balanced 1,000-image validation subset (100 images per class)
held out from the training split. VAE candidates receive 1,500 optimizer
updates; each diffusion candidate receives 2,000. Every candidate generates
1,000 samples and is ranked within its model family by validation feature FID.

| Family | Selected candidate | Learning rate | Beta | EMA | Pilot feature FID |
|---|---|---:|---:|---:|---:|
| VAE | `vae_lr1e-3_beta1` | 1e-3 | 1.0 | — | 137.20 |
| Pixel DDPM | `ddpm_lr2e-4_ema0.995` | 2e-4 | — | 0.995 | 557.46 |
| LDM | `ldm_lr2e-4_ema0.999` | 2e-4 | — | 0.999 | 270.63 |

The short diffusion pilots are intentionally under-trained, so their absolute
FID values are selection diagnostics, not final model-quality estimates. The
selected optimizer settings are frozen and transferred to both datasets. This
tests cross-domain transfer and avoids tuning on the color test set.

### Full-budget diffusion sampling diagnostics

The short pilot selected optimizer hyperparameters but did not expose an LDM
sampling pathology visible after 5,000 updates: without stabilization, sampled
standardized latents had SD 13.3 (rather than approximately 1) and maximum
absolute value 327.3. A validation-only follow-up compared base width 32/64,
fixed-large/exact-posterior/DDIM reverse updates, and clean-latent clipping.
Base width 64, exact posterior variance, and clipping the predicted clean latent
to ±3 produced the lowest validation FID (58.78), restored 10/10 class coverage,
and brought latent SD to 0.842. Thresholds 1.5, 2, 2.5, 3, 4, and 6 were tested;
3 was selected by the declared FID criterion.

An equivalent pixel-DDPM sampler diagnostic compared fixed-large variance,
exact posterior variance, exact posterior with predicted-clean image clipping,
and deterministic DDIM with clipping. Exact posterior sampling with the clean
image clipped to `[-1,1]` reduced validation FID from 68.86 to 34.04 and was
selected. These diagnostics use only the training-derived validation split.

## Structural and budget choices

- **32 × 32 RGB interface.** Fashion-MNIST is resized and channel-replicated;
  the color photographs are padded to square before resizing. A shared interface
  keeps architecture and evaluator capacity comparable without distorting the
  photographs' aspect ratios.
- **Spatial 4 × 8 × 8 latent.** The latent preserves spatial layout while
  reducing the denoiser's spatial positions by 16× and scalar representation by
  12× relative to a 3 × 32 × 32 image. Spatial latents follow the motivation of
  latent diffusion [Rombach et al.](https://arxiv.org/abs/2112.10752).
- **Per-channel latent standardization.** This fixes the latent scale seen by
  the diffusion process, directly controlling effective signal-to-noise ratio;
  latent scaling is an explicit component of LDM implementations.
- **Cosine diffusion schedule.** The schedule allocates noise more gradually
  than a simple linear schedule and is supported by Improved DDPM
  [Nichol and Dhariwal](https://arxiv.org/abs/2102.09672).
- **100 reverse steps.** This is a deliberate low-resolution quality/latency
  budget shared by pixel and latent diffusion. It is not claimed to be optimal;
  a step-count ablation or DDIM/solver sampler is the highest-value follow-up.
- **Predicted-clean clipping.** Both diffusion models use exact posterior
  variance. Pixel predictions are constrained to their normalized support
  `[-1,1]`; LDM predictions are constrained to ±3 standardized latent units,
  selected from the validation-only stabilization search.
- **3,000 VAE and 5,000 denoiser updates.** Fixed update counts are used instead
  of equal epochs because the datasets contain 60,000 versus 8,007 training
  images. Equal epochs would give the smaller dataset far fewer optimizer
  updates.
- **Batch 256 for training.** This produced stable validation searches and high
  training utilization without changing the optimization regime between the
  pilot and final run.
- **BF16 and sampling batch 4,096.** The RTX 5070 Ti reports native BF16 support.
  A local inference probe found that pixel-DDPM sampling improved from about
  31 ms/image in the accidental FP32/256-batch path to about 11.7 ms/image with
  BF16 and batch 4,096, using roughly 8.0 GB allocated VRAM. This is a throughput
  optimization; the reverse equations and weights are unchanged.
- **Timing warm-up.** Each measured sampler is preceded by one untimed forward
  at every exact batch shape. Training performs every configured update, but
  wall-clock timing begins after the first update. CUDA synchronization brackets
  every timer. This prevents seed 0 alone from paying cuDNN/kernel-selection
  startup costs and makes the five timing replicates comparable.
- **KL and learning-rate warm-up.** The VAE's beta and all learning rates warm
  up for 200 steps to avoid abrupt early regularization and large initial
  updates. Cosine decay then reduces the rate to 10% of its initial value.
- **Gradient clipping at 1.0.** This bounds rare unstable updates across all
  three systems; non-finite loss counts and late-training coefficient of
  variation are saved to verify stability rather than assuming it.
- **Five seeds.** Seeds 0–4 expose initialization and order sensitivity. The
  report uses mean and sample standard deviation, and also includes every
  seed's FID curve.

## Why the models are not parameter matched

The study compares complete systems suitable for use: the VAE needs its encoder
and decoder, the DDPM needs its pixel U-Net, and the LDM needs both its VAE and
latent U-Net. Matching parameter counts would require changing widths and could
still leave training FLOPs and sequential sampling cost unmatched. The report
therefore avoids causal claims about objectives and explicitly reports
parameters, wall-clock time, training peak memory, sampling peak memory, and
latency.

## Most valuable next search dimensions

1. Diffusion sampling steps and a DDIM or solver-based sampler.
2. U-Net base width under a fixed training-FLOP budget.
3. Latent channels (4, 8, 16) and downsampling factor (2×, 4×, 8×).
4. VAE perceptual reconstruction loss versus the current BCE likelihood.
5. EMA decay jointly tuned with the longer definitive training budget.
6. A parameter-matched and an equal-FLOP comparison reported separately from
   the present system-level benchmark.
