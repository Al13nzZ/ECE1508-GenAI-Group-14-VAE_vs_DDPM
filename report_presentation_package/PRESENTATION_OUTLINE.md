# Academic Five-Slide Presentation Outline

Target length: 6–7 minutes. Each slide should make one defensible claim and
show the evidence supporting it. Use 26–30 pt body text, short technical
phrases instead of paragraphs, and citations in a small footer.

## Slide 1 — Title and Research Question

### Title

**VAE vs. DDPM for Image Generation on Fashion-MNIST**

### Subtitle

**A controlled comparison of distribution quality, diversity, and
computational efficiency**

### Content shown on the slide

- Team member names, course, institution, and date.
- Research question in a highlighted box:

  > Under a shared dataset and evaluation pipeline, how much generation
  > quality does iterative diffusion gain over latent-variable decoding, and
  > what computational cost accompanies that gain?

- One-line contribution:

  > We train and evaluate a convolutional VAE and a 100-step DDPM using 5,000
  > generated samples per model and a shared Fashion-MNIST feature evaluator.

### Figure and layout

No graph. Use a restrained academic cover. A faint crop of generated samples
may be used as a background strip, but it must not compete with the research
question.

### Footer citation

Kingma and Welling (2013); Ho et al. (2020); Xiao et al. (2017).

---

## Slide 2 — Models, Experimental Control, and Evaluation

### Claim

**The models use fundamentally different generation mechanisms but are tested
with the same data, sample count, reference set, and evaluator.**

### Left column — Model formulation

**VAE**

- Encoder: $q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))$.
- Latent dimension: 32.
- Objective: binary reconstruction loss plus KL divergence, $\beta=1$.
- Generation: $z\sim\mathcal{N}(0,I)$ followed by one decoder pass.
- 1.70M parameters; trained for 25 epochs.

**DDPM**

- Forward process:
  $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$.
- Time-conditioned U-Net, cosine schedule, bottleneck attention, EMA sampling.
- Objective: noise-prediction MSE.
- Generation: 100 sequential reverse-denoising steps.
- 5.22M parameters; trained for 35 epochs.

### Right column — Controlled experimental setup

| Control | Setting |
| --- | --- |
| Dataset | Fashion-MNIST: 60k train / 10k test |
| Image format | 28x28 grayscale, 10 classes |
| Generated evaluation set | 5,000 images per model |
| Evaluator | CNN, 91.37% ± 0.20% test accuracy |
| Feature representation | 128-dimensional classifier features |
| Optimizer | AdamW + cosine LR decay + gradient clipping |
| Hardware | RTX 5070 Ti, PyTorch 2.11, CUDA 12.8 |
| Random seeds | 0–9; independent full runs |

### Figures and layout

- Place `figures/main/vae_latent_interpolation.png` below the VAE description.
- Place `figures/main/ddpm_denoising_trajectory.png` below the DDPM description.
- Crop excess whitespace so each mechanism figure remains readable.

### Bottom metric strip

**Quality:** classifier-feature FID/KID and confidence  
**Diversity:** class coverage and normalized class entropy  
**Cost:** training time, sampling latency, memory, and parameter count

### Academic qualification

The evaluator is shared, but the architectures are not parameter matched.
Therefore, this is a controlled system comparison, not a capacity-matched
causal ablation.

---

## Slide 3 — Generation Quality and Class Diversity

### Claim

**The DDPM produces a substantially closer feature distribution, more
recognizable images, and a more balanced predicted class distribution.**

### Figures and layout

- Top strip: `figures/main/real_vae_ddpm_comparison.png`, labeled
  **representative qualitative samples; quantitative results use 10 seeds**.
- Main quantitative panel: `figures/multiseed/quality_metrics_multiseed.png`.
- Right or bottom inset: `figures/multiseed/class_distribution_multiseed.png`.
- If space is tight, remove the repeated numerical table from the rendered
  slide and keep only three callouts; the dashboard already contains all six
  mean ± SD comparisons.

### Quantitative table shown on the slide

| Metric | VAE | DDPM | Interpretation |
| --- | ---: | ---: | --- |
| Classifier-feature FID down | 43.32 ± 4.51 | **11.14 ± 5.79** | DDPM features closer to real data |
| Classifier-feature KID down | 1.807 ± 0.268 | **0.530 ± 0.342** | Same conclusion without Gaussian-only summary |
| Mean confidence up | 0.717 ± 0.009 | **0.841 ± 0.014** | DDPM samples more recognizable |
| Confidence >= 0.8 up | 43.37% ± 1.59% | **67.11% ± 2.93%** | More high-confidence DDPM samples |
| Class coverage | 10/10 | 10/10 | Both reach every predicted class in every run |
| Normalized entropy up | 0.941 ± 0.013 | **0.971 ± 0.028** | DDPM distribution closer to uniform on average |
| Sobel sharpness up | 0.549 ± 0.006 | **0.646 ± 0.059** | Stronger mean edge response for DDPM |

### Interpretation box

- The visual grids and classifier-derived measures agree: DDPM samples are
  sharper and more recognizable.
- Coverage alone is insufficient: both models reach 10 classes, but the VAE
  overproduces several categories.
- Entropy describes balance across predicted classes, not diversity within a
  class.
- FID/KID use Fashion-MNIST classifier features; they are not standard
  ImageNet Inception scores.

### Optional verbal result, not another slide element

The mean nearest-neighbor memorization-screen rate was 1.17% ± 1.12% for DDPM
and 0.31% ± 0.40% for VAE under a strict real-data-calibrated threshold. This
does not establish copying or prove that memorization is absent.

---

## Slide 4 — Computational Cost and Quality–Efficiency Trade-off

### Claim

**The DDPM's quality improvement requires substantially greater sequential
computation, memory, and model capacity.**

### Main figure

`figures/multiseed/compute_metrics_multiseed.png`

Caption on slide: **Ten-seed computational cost, mean ± sample SD**

### Main quantitative table

| Resource metric | VAE | DDPM | DDPM/VAE |
| --- | ---: | ---: | ---: |
| Training time | 0.411 ± 0.020 min | 4.811 ± 0.059 min | **11.7x** |
| Sampling latency | 0.00784 ± 0.00062 ms/image | 8.173 ± 0.110 ms/image | **~1,043x** |
| Peak allocated GPU memory | 558 MB | 6,275 MB | **11.2x** |
| Trainable parameters | 1.70M | 5.22M | **3.1x** |
| Generation network evaluations | 1 decoder pass | 100 denoising passes | **100x steps** |

### Technical explanation box

- The approximately 3.1x parameter increase does **not** explain the roughly
  1,043x mean sampling-latency increase.
- The dominant sampling cost is sequential depth: this DDPM evaluates its U-Net
  100 times, while the VAE evaluates its decoder once.
- DDPM peak memory also includes wider feature maps and training activations at
  a batch size of 1024.
- VAE speed is especially useful for interactive or latency-constrained
  generation; DDPM is preferable when fidelity outweighs inference cost.

### Quality-versus-cost summary band

| Model | Main strength | Main limitation |
| --- | --- | --- |
| VAE | Very fast, compact, smooth latent interpolation | Blurrier samples and weaker feature match |
| DDPM | Better FID/KID, confidence, sharpness, and balance | Sequential sampling and much higher memory cost |

### Measurement qualification

> Times are measurements from one RTX 5070 Ti run using mixed precision and
> large batches. Ratios support the within-system comparison; absolute latency
> is hardware- and implementation-dependent.

This qualification should appear in small text at the bottom rather than being
left only to the speech.

---

## Slide 5 — Conclusions, Validity, and Next Experiments

### Main conclusion

**Neither model dominates every criterion: diffusion buys quality and balance
with compute, while the VAE buys speed and compactness with softer samples.**

### Evidence-based decision table

| Requirement | Preferred model | Evidence |
| --- | --- | --- |
| Distribution match and recognizability | DDPM | Lower FID/KID; higher confidence |
| Balanced predicted class coverage | DDPM | Mean entropy 0.971 vs. 0.941 |
| Low-latency generation | VAE | ~1,043x lower mean measured ms/image |
| Lower training/memory budget | VAE | 11.7x less mean training time; 11.2x less peak allocation |
| Smooth latent manipulation | VAE | Direct 32-D interpolation |

### Threats to validity

- Ten random seeds: sample SD is reported, but formal confidence intervals and hypothesis tests are not.
- Unequal capacity: 1.70M versus 5.22M parameters.
- Evaluator dependence: feature metrics rely on classifiers averaging 91.37% ± 0.20% accuracy.
- Dataset scope: 28x28 grayscale Fashion-MNIST is much simpler than natural
  images.
- Memorization screen uses a finite reference subset and feature representation.

### Next experiments

1. Add formal paired statistical tests and confidence intervals across seeds.
2. Match parameter count or compute budget to separate architecture from scale.
3. Test 10, 25, 50, and 100 DDPM sampling steps to estimate a quality–latency
   Pareto curve.
4. Add per-class precision/recall or intra-class diversity measurements.
5. Evaluate on a higher-resolution dataset to test whether the trade-off
   persists.

### Layout

- Left 55%: decision table.
- Right 45%: threats to validity and next experiments.
- Bottom highlighted takeaway:

  > In this controlled implementation, DDPM is the quality-oriented choice;
  > VAE is the efficiency-oriented choice.

No additional figure is needed. The slide should end with synthesis rather
than introduce another plot.

---

## Final five-slide narrative

The complete academic argument must remain inside these five slides:

1. Slide 1 defines the research question and contribution.
2. Slide 2 establishes the models, controls, and evaluation validity.
3. Slide 3 presents all essential quality and diversity evidence.
4. Slide 4 presents all essential computational evidence and explains its
   cause.
5. Slide 5 makes the decision, states limitations, and defines the next
   experiments.

Do not append backup slides. If asked about training stability, memorization,
or evaluator error, answer verbally from the following facts already supported
by the report:

- both training runs completed without non-finite losses;
- final DDPM noise MSE was 0.0696;
- VAE test negative ELBO was 238.20, but the objectives are not comparable;
- the nearest-neighbor screen found no strong evidence of direct memorization;
- evaluator accuracy averaged 91.37% ± 0.20%, so classifier-derived metrics remain imperfect.
