import unittest

import numpy as np
import torch

import run_three_model_color_benchmark as benchmark


class PipelineSmokeTests(unittest.TestCase):
    def test_model_shapes_and_finite_values(self):
        device=benchmark.DEVICE
        images=torch.rand(2,3,32,32,device=device); timesteps=torch.tensor([1,2],device=device)
        vae=benchmark.VAE().to(device); ddpm=benchmark.PixelUNet().to(device); ldm=benchmark.LatentDenoiser().to(device)
        logits,mean,logvar=vae(images)
        self.assertEqual(logits.shape,images.shape); self.assertEqual(mean.shape,(2,4,8,8))
        self.assertEqual(ddpm(images,timesteps).shape,images.shape); self.assertEqual(ldm(mean,timesteps).shape,mean.shape)
        for tensor in (logits,mean,logvar): self.assertTrue(torch.isfinite(tensor).all())

    def test_diffusion_noise_and_short_sampling(self):
        device=benchmark.DEVICE; diffusion=benchmark.Diffusion(4)
        model=benchmark.LatentDenoiser().to(device); clean=torch.randn(2,4,8,8,device=device); t=torch.tensor([0,3],device=device)
        noisy,noise=diffusion.noise(clean,t)
        self.assertEqual(noisy.shape,clean.shape); self.assertEqual(noise.shape,clean.shape)
        sampled=diffusion.sample(model,(2,4,8,8)); self.assertTrue(torch.isfinite(sampled).all())

    def test_feature_metrics(self):
        rng=np.random.default_rng(1508); features=rng.normal(size=(256,32))
        self.assertAlmostEqual(benchmark.fid(features,features),0.0,places=5)
        mean,std=benchmark.kid(features,features,1508)
        self.assertTrue(np.isfinite(mean)); self.assertTrue(np.isfinite(std))

    def test_classifier_diagnostics(self):
        device=benchmark.DEVICE; model=benchmark.Classifier(3).to(device)
        images=torch.rand(9,3,32,32); labels=torch.tensor([0,1,2]*3)
        counts,normalized,recall=benchmark.classifier_diagnostics(model,[(images,labels)],3)
        self.assertEqual(counts.shape,(3,3)); self.assertEqual(recall.shape,(3,))
        self.assertTrue(np.allclose(normalized.sum(1),1.0))


if __name__=="__main__": unittest.main()
