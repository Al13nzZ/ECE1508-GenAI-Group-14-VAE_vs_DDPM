# Experiment environment

The definitive benchmark was configured for the following local environment.

- GPU: NVIDIA GeForce RTX 5070 Ti (16,303 MiB reported VRAM)
- NVIDIA driver: 596.49
- Operating system: Microsoft Windows NT 10.0.26200.0
- Python: 3.12.10
- PyTorch: 2.11.0+cu128
- CUDA runtime used by PyTorch: 12.8
- Torchvision: 0.26.0+cu128
- NumPy: 2.4.4
- pandas: 3.0.5
- SciPy: 1.18.0
- Matplotlib: 3.11.1

CUDA training uses BF16 autocast because this GPU reports native BF16 support.
Diffusion inference also uses BF16, with a 4,096-image sampling batch selected
from local throughput/VRAM probes. The pipeline also uses pinned-memory data
loaders, four persistent workers, cuDNN kernel benchmarking, AdamW, gradient
clipping, linear warm-up followed by cosine decay, and exponential moving
averages for both diffusion denoisers.
Exact experiment settings are serialized in the final results directory as
`config.json` and are also embedded in every per-seed `results.json` file.
Timed inference uses an untimed warm-up forward at every exact batch shape;
training timing begins after the first of the fully executed optimizer updates.
CUDA synchronization brackets all reported wall-clock intervals.
