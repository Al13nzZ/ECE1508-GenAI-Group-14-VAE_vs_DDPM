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
| Evaluator | CNN, 91.34% test accuracy |
| Feature representation | 128-dimensional classifier features |
| Optimizer | AdamW + cosine LR decay + gradient clipping |
| Hardware | RTX 5070 Ti, PyTorch 2.11, CUDA 12.8 |
| Random seed | 42 |


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


- Top 48%: `figures/main/real_vae_ddpm_comparison.png`, full width.
- Bottom-left 32%: `figures/main/class_distribution_comparison.png`.
- Bottom-right: compact quantitative table.


### Quantitative table shown on the slide


| Metric | VAE | DDPM | Interpretation |
| --- | ---: | ---: | --- |
| Classifier-feature FID down | 51.12 | **7.66** | DDPM features closer to real data |
| Classifier-feature KID down | 1.874 | **0.326** | Same conclusion without Gaussian-only summary |
| Mean confidence up | 0.704 | **0.832** | DDPM samples more recognizable |
| Confidence >= 0.8 up | 41.0% | **65.6%** | More high-confidence DDPM samples |
| Class coverage | 10/10 | 10/10 | Both reach every predicted class |
| Normalized entropy up | 0.931 | **0.991** | DDPM distribution closer to uniform |
| Sobel sharpness up | 0.542 | **0.680** | Stronger edge response for DDPM |


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


The DDPM nearest-neighbor memorization-screen rate was 0.78% under a strict
real-data-calibrated threshold; VAE was 0%. This does not establish copying or
prove that memorization is absent.


---


## Slide 4 — Computational Cost and Quality–Efficiency Trade-off


### Claim


**The DDPM's quality improvement requires substantially greater sequential
computation, memory, and model capacity.**


### Top-left figure


`figures/main/training_time_comparison.png`


Caption on slide: **End-to-end measured training time on the same GPU**


### Top-right figure


`figures/main/sampling_time_comparison.png`


Caption on slide: **Batched generation latency per image**


### Main quantitative table


| Resource metric | VAE | DDPM | DDPM/VAE |
| --- | ---: | ---: | ---: |
| Training time | 0.37 min | 4.76 min | **12.8x** |
| Sampling latency | 0.0075 ms/image | 8.02 ms/image | **~1,068x** |
| Peak allocated GPU memory | 558 MB | 6,275 MB | **11.2x** |
| Trainable parameters | 1.70M | 5.22M | **3.1x** |
| Generation network evaluations | 1 decoder pass | 100 denoising passes | **100x steps** |


### Technical explanation box


- The approximately 3.1x parameter increase does **not** explain the roughly
  1,068x sampling-latency increase.
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
| Balanced predicted class coverage | DDPM | Entropy 0.991 vs. 0.931 |
| Low-latency generation | VAE | ~1,068x lower measured ms/image |
| Lower training/memory budget | VAE | 12.8x less training time; 11.2x less peak allocation |
| Smooth latent manipulation | VAE | Direct 32-D interpolation |


### Threats to validity


- One random seed: no run-to-run confidence intervals.
- Unequal capacity: 1.70M versus 5.22M parameters.
- Evaluator dependence: feature metrics rely on a 91.34%-accurate classifier.
- Dataset scope: 28x28 grayscale Fashion-MNIST is much simpler than natural
  images.
- Memorization screen uses a finite reference subset and feature representation.


### Next experiments


1. Repeat at least three seeds and report mean plus standard deviation.
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
- evaluator accuracy was 91.34%, so classifier-derived metrics remain imperfect.


