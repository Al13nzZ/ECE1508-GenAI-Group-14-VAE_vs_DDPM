# Speaker Script for the Final Five-Slide Presentation

Estimated length: 6–7 minutes at a natural academic speaking pace. The script
follows the latest `PRESENTATION_OUTLINE.md` exactly.

## Slide 1 — Title and Research Question

**Target time: 40–50 seconds**

“Our project compares a Variational Autoencoder, or VAE, with a Denoising
Diffusion Probabilistic Model, or DDPM, for image generation on Fashion-MNIST.

Rather than asking which model is universally better, we study a more practical
question: under the same dataset and evaluation pipeline, how much generation
quality does iterative diffusion gain over direct latent-variable decoding,
and what computational cost accompanies that improvement?

To answer this, we repeated the complete experiment across ten random seeds,
generating 5,000 evaluation images per model per seed, or 50,000 per model in
total. We compared distribution similarity, recognizability, class diversity,
training behavior, memory use, and generation speed. Our central
result is a clear quality–efficiency trade-off: the DDPM generated stronger and
more balanced samples, while the VAE was dramatically faster and smaller.”

**Transition:**

“I’ll first explain how the two systems differ and how we kept their evaluation
consistent.”

---

## Slide 2 — Models, Experimental Control, and Evaluation

**Target time: 1 minute 30 seconds**

“The VAE and DDPM represent two fundamentally different approaches to
generation.

On the left, the VAE encoder maps an image to the mean and variance of a
32-dimensional Gaussian latent distribution. We sample a latent variable using
the reparameterization trick and reconstruct the image with a convolutional
decoder. Its objective combines binary reconstruction loss with KL divergence,
using beta equal to one. The KL term makes the latent space smooth and
sampleable, but that compression and regularization can remove fine detail. At
generation time, the VAE samples from a standard Gaussian and requires only one
decoder pass. It contains 1.70 million trainable parameters and was trained for
25 epochs.

On the right, the DDPM gradually adds Gaussian noise during its forward process.
A time-conditioned U-Net learns to predict that noise at randomly sampled
timesteps. Our implementation uses a cosine noise schedule, residual blocks,
bottleneck self-attention, and an exponential moving average of the parameters
for sampling. Generation begins with random noise and applies 100 sequential
reverse-denoising steps. The DDPM contains 5.22 million parameters and was
trained for 35 epochs.

For experimental control, both models used the same 60,000 Fashion-MNIST
training images in ten independent runs using seeds zero through nine. Each
run produced 5,000 evaluation samples per model, for 50,000 per model overall.
A separate CNN classifier, trained only on real Fashion-MNIST in each run, achieved
91.37 percent mean test accuracy, with a standard deviation of 0.20 percentage
points across the ten runs. We used its predictions and 128-dimensional
features to evaluate both generators.

Our quality metrics are classifier-feature FID and KID, plus classifier
confidence. Diversity is measured through class coverage and normalized class
entropy. We also record training time, batched sampling latency, trainable
parameters, and peak allocated GPU memory.

The evaluator and data are shared, but the models are not parameter matched.
This is therefore a controlled comparison of our two complete systems—not a
causal claim that every performance difference comes only from the generative
objective.”

**Transition:**

“With that evaluation framework established, we can examine the generated
images and their distribution-level results.”

---

## Slide 3 — Generation Quality and Class Diversity

**Target time: 1 minute 35 seconds**

“The sample strip at the top is a representative qualitative example because
images themselves cannot be averaged across seeds. All quantitative graphs on
this slide show the mean and sample standard deviation from ten independent
runs. Both models learn recognizable clothing structure, but the VAE outputs
are generally softer, with less distinct boundaries between the object and
background. DDPM samples have clearer edges and more recognizable object
shapes.

The six-panel quality figure shows that the numerical results agree with this
visual pattern. Classifier-feature FID
is 43.32 plus or minus 4.51 for the VAE and 11.14 plus or minus 5.79 for the
DDPM. Lower is better because this
metric compares the means and covariances of the real and generated feature
distributions. Mean classifier-feature KID also falls from 1.807 plus or minus
0.268 to 0.530 plus or minus 0.342. KID is a
kernel-based two-sample statistic, so agreement between FID and KID gives
stronger evidence than relying on one score alone.

The mean classifier confidence rises from 0.717 to 0.841. More importantly,
the proportion of generated images receiving at least 0.8 confidence increases
from 43.37 percent for the VAE to 67.11 percent for the DDPM. Sobel edge
sharpness also increases from 0.549 to 0.646 on average, supporting the qualitative
observation of clearer boundaries.

Both models cover all ten predicted Fashion-MNIST classes, but coverage alone
hides imbalance. In the distribution chart, the VAE overproduces categories
such as T-shirt or top and pullover while underrepresenting others. Its
mean normalized class entropy is 0.941. The DDPM distribution is closer to the
uniform reference line and reaches a mean entropy of 0.971, where one would mean a
perfectly uniform ten-class distribution.

These metrics must still be interpreted carefully. Entropy measures balance
between predicted classes, not variation within each class. FID and KID use
our Fashion-MNIST classifier features, so they are dataset-specific and cannot
be compared directly with standard ImageNet Inception scores. However, the
sample grid, FID, KID, confidence, sharpness, and class distribution all point
in the same direction: the DDPM gives the stronger generation-quality result
in this experiment.”

**Transition:**

“That improvement is substantial, but it is not free. The largest contrast
appears in computational cost.”

---

## Slide 4 — Computational Cost and Quality–Efficiency Trade-off

**Target time: 1 minute 35 seconds**

“The computational dashboard reports ten-seed means with sample-standard-
deviation error bars and quantifies the price of the DDPM’s better samples.
All measurements come from the same RTX 5070 Ti system using PyTorch
2.11, CUDA 12.8, mixed precision, and a batch size of 1024.

VAE training averaged 0.411 plus or minus 0.020 minutes, compared with 4.811
plus or minus 0.059 minutes for the DDPM. That makes the DDPM approximately
11.7 times slower to train on average.

The difference is much larger during generation. Batched VAE sampling required
about 0.00784 plus or minus 0.00062 milliseconds per image. DDPM sampling
required 8.173 plus or minus 0.110 milliseconds per image, approximately 1,043
times longer. Peak allocated GPU memory rose
from 558 megabytes to 6,275 megabytes, an increase of about 11.2 times. The
parameter count increased from 1.70 million to 5.22 million, or about 3.1
times.

The parameter increase alone cannot explain a thousand-fold sampling gap. The
main reason is sequential computation. A VAE generates an image with one
decoder evaluation. This DDPM performs 100 U-Net evaluations, and each reverse
step depends on the output of the previous step, limiting parallelization
across timesteps. Its wider activation maps and training batch also contribute
to the higher peak memory allocation.

This gives us the central trade-off. The VAE is compact, extremely fast to
sample from, and supports smooth latent interpolation, but its generated
feature distribution is weaker and its images are blurrier. The DDPM improves
FID, KID, recognizability, sharpness, and class balance, but requires much more
training time, memory, and sequential inference.

The absolute timing values depend on hardware, batching, and implementation.
We therefore use them as a controlled within-system comparison, not as
universal latency benchmarks.”

**Transition:**

“These results lead to a conditional conclusion rather than a single winner.”

---

## Slide 5 — Conclusions, Validity, and Next Experiments

**Target time: 1 minute 20 seconds**

“The preferred model depends on the application requirement.

If distribution similarity, recognizability, sharpness, and balanced class
coverage are the priorities, our DDPM is the stronger choice. It has much lower
classifier-feature FID and KID, higher confidence, and entropy close to one.

If low latency, low memory consumption, rapid iteration, or smooth latent
manipulation matters more, the VAE is the stronger choice. It uses about one
third of the parameters, roughly one eleventh of the peak allocated memory,
and generates images over one thousand times faster in the recorded batched
experiment.

There are still important threats to validity. First, ten seeds quantify
run-to-run variation, but we have not added formal hypothesis tests. Second,
the models are not capacity matched. Third, the generated-image metrics depend
on classifiers averaging 91.37 percent—not perfect—test accuracy. Fourth,
Fashion-MNIST is a simple 28 by 28
grayscale dataset, so the result may not transfer directly to natural images.
Finally, our nearest-neighbor analysis found no strong evidence of direct
memorization, but a finite feature-space screen cannot prove its absence.

The next experiments should add paired statistical tests, compare parameter-
or compute-matched architectures, and evaluate DDPM sampling with
10, 25, 50, and 100 steps. That would produce a quality–latency Pareto curve
and show whether much of the DDPM quality can be retained with fewer reverse
evaluations. Per-class diversity metrics and a higher-resolution dataset would
further test the generality of the conclusions.

Our final takeaway is: in this controlled implementation, diffusion bought
better quality and class balance with substantially more compute, while the
VAE bought speed and compactness with softer samples.”

## Timing Summary

| Slide | Target time |
| --- | ---: |
| 1. Research question | 0:45 |
| 2. Models and evaluation | 1:30 |
| 3. Quality and diversity | 1:35 |
| 4. Computational trade-off | 1:35 |
| 5. Conclusion and validity | 1:20 |
| **Total** | **6:45** |
