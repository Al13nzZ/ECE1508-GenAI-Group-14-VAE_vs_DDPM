# ECE1508-GenAI-Group-14-VAE_vs_DDPM
# Abstract

## Summary of the Problem

This project proposes a comparative study of Variational Autoencoders (VAEs) and Denoising Diffusion Probabilistic Models (DDPMs) for image generation on Fashion-MNIST. The main idea of the project is to understand how two different generative learning approaches perform under the same dataset and experimental setting.

VAEs generate images by compressing data into a latent space and sampling from it [1], while DDPMs generate images by starting with random noise and gradually denoising it into a realistic image [2]. By comparing these two methods, the project aims to explain the practical trade-offs between latent-variable generation and diffusion-based generation.

## Key Outputs

The key outputs of this project will include:

* Trained VAE and DDPM models
* Generated image samples
* Visual comparison grids
* Training loss curves
* Evaluation results for each model

The models will be evaluated using both qualitative and quantitative differentiation metrics, including sample quality, distribution similarity, diversity, generation structure, memorization, training stability, and computational cost.

The details of the evaluation areas are shown below:

| Evaluation Area         | Metric / Method                                        | Purpose                                                                                           |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Sample Quality          | Visual sample grids                                    | Check whether generated images are clear, sharp, and recognizable                                 |
| Distribution Similarity | FID or custom classifier-based FID                     | Measure how close generated images are to real images [3]                                         |
| Distribution Similarity | KID                                                    | Provide another distribution-level quality metric, especially useful for smaller sample sizes [4] |
| Diversity               | Class coverage and class distribution entropy          | Check whether the model generates all clothing categories or only a few                           |
| Generation Structure    | VAE latent interpolation and DDPM denoising trajectory | Compare how each model represents and generates images                                            |
| Memorization Check      | Nearest-neighbor comparison                            | Check whether generated images are new or copied from training images                             |
| Training Stability      | Loss curves and failure cases                          | Compare whether training is smooth, unstable, or prone to failure                                 |
| Compute Cost            | Training time and sampling time                        | Compare practical efficiency and resource requirements                                            |

## Course Components

This project covers two major course components: VAEs and Diffusion Models.

The VAE part connects to variational inference, latent-variable models, and data generation by sampling from latent space. The DDPM part connects to probabilistic diffusion models, including the forward noising process, reverse denoising process, and sampling from learned distributions.

## Required Resources

The main required resource is the Fashion-MNIST dataset, which contains 60,000 training images and 10,000 test images of 28 × 28 grayscale fashion items across 10 classes [5].

We will load and preprocess the dataset using Torchvision, implement both models in Python with PyTorch, and use NumPy, Matplotlib, TorchMetrics, and related packages for training, visualization, and evaluation.

Because DDPM training and sampling are more computationally expensive than VAE experiments, we plan to use Google Colab as the main training environment and run smaller tests locally when feasible.

The final repository will include:

* A documented codebase
* A `README.md` file
* A reproducible environment file
* A demo notebook showing model training, sampling, and evaluation

## References

[1] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *arXiv preprint arXiv:1312.6114*, 2013. [Online]. Available: https://arxiv.org/abs/1312.6114

[2] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” *arXiv preprint arXiv:2006.11239*, 2020. [Online]. Available: https://arxiv.org/abs/2006.11239

[3] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,” *Advances in Neural Information Processing Systems*, 2017. [Online]. Available: https://arxiv.org/abs/1706.08500

[4] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “Demystifying MMD GANs,” *arXiv preprint arXiv:1801.01401*, 2018. [Online]. Available: https://arxiv.org/abs/1801.01401

[5] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms,” *arXiv preprint arXiv:1708.07747*, 2017. [Online]. Available: https://arxiv.org/abs/1708.07747
