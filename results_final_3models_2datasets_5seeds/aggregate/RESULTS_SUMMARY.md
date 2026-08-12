# Three-model benchmark: five-seed results

Models: VAE, pixel-space DDPM, and latent diffusion. Datasets: Fashion-MNIST and Fashion Product Images Small.

Values are mean ± sample standard deviation across seeds 0–4.

## Fashion Mnist

- custom fid — VAE: 119.0966 ± 2.8724; DDPM: 29.6439 ± 1.4938; LDM: 52.0127 ± 0.7082
- custom kid mean — VAE: 1.1133 ± 0.0211; DDPM: 0.3487 ± 0.0187; LDM: 0.6870 ± 0.0154
- mean classifier confidence — VAE: 0.7939 ± 0.0044; DDPM: 0.9279 ± 0.0016; LDM: 0.9513 ± 0.0025
- recognizability rate at 0.8 — VAE: 0.5726 ± 0.0100; DDPM: 0.8531 ± 0.0070; LDM: 0.8993 ± 0.0057
- class distribution entropy — VAE: 0.8684 ± 0.0077; DDPM: 0.9733 ± 0.0045; LDM: 0.9603 ± 0.0032
- class distribution jsd — VAE: 0.0769 ± 0.0047; DDPM: 0.0152 ± 0.0023; LDM: 0.0251 ± 0.0022
- inception score — VAE: 4.2930 ± 0.1004; DDPM: 7.7850 ± 0.0885; LDM: 8.0789 ± 0.0492
- feature pairwise distance — VAE: 13.0054 ± 0.1430; DDPM: 19.5947 ± 0.0423; LDM: 20.4805 ± 0.1116
- sobel sharpness — VAE: 0.5462 ± 0.0085; DDPM: 0.6081 ± 0.0090; LDM: 0.5067 ± 0.0106
- training minutes — VAE: 1.6122 ± 0.0022; DDPM: 5.5522 ± 0.0022; LDM: 3.4841 ± 0.0519
- sampling ms per image — VAE: 0.0362 ± 0.0002; DDPM: 10.6112 ± 0.0079; LDM: 0.7628 ± 0.0012
- peak gpu memory mb — VAE: 6300.8589 ± 3.6336; DDPM: 7682.5322 ± 0.0000; LDM: 5195.8330 ± 0.0000

## Fashion Product Images

- custom fid — VAE: 83.8805 ± 3.7983; DDPM: 16.6743 ± 0.8536; LDM: 33.6526 ± 2.2124
- custom kid mean — VAE: 0.9403 ± 0.0490; DDPM: 0.2080 ± 0.0163; LDM: 0.4287 ± 0.0447
- mean classifier confidence — VAE: 0.7899 ± 0.0064; DDPM: 0.8790 ± 0.0032; LDM: 0.8963 ± 0.0068
- recognizability rate at 0.8 — VAE: 0.5720 ± 0.0119; DDPM: 0.7556 ± 0.0066; LDM: 0.7901 ± 0.0154
- class distribution entropy — VAE: 0.6437 ± 0.0193; DDPM: 0.7278 ± 0.0102; LDM: 0.7337 ± 0.0082
- class distribution jsd — VAE: 0.0608 ± 0.0067; DDPM: 0.0062 ± 0.0012; LDM: 0.0268 ± 0.0029
- inception score — VAE: 3.4052 ± 0.1040; DDPM: 5.2830 ± 0.0913; LDM: 5.6369 ± 0.0699
- feature pairwise distance — VAE: 13.8118 ± 0.1305; DDPM: 18.3564 ± 0.1029; LDM: 18.9835 ± 0.1809
- sobel sharpness — VAE: 0.7671 ± 0.0029; DDPM: 0.8267 ± 0.0021; LDM: 0.7821 ± 0.0018
- training minutes — VAE: 1.6645 ± 0.0018; DDPM: 5.6226 ± 0.0056; LDM: 3.7824 ± 0.0106
- sampling ms per image — VAE: 0.0364 ± 0.0003; DDPM: 10.6240 ± 0.0112; LDM: 0.7649 ± 0.0032
- peak gpu memory mb — VAE: 6317.7334 ± 0.0000; DDPM: 7698.4692 ± 0.0000; LDM: 5211.1450 ± 0.0000
