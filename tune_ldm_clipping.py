"""Validation search for stabilized clean-latent clipping in LDM sampling."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

import run_three_model_color_benchmark as bench


OUT=Path("hyperparameter_search/ldm_clipping"); OUT.mkdir(parents=True,exist_ok=True)


@torch.inference_mode()
def sample_clipped(model,diffusion,shape,threshold):
    x=torch.randn(shape,device=bench.DEVICE)
    for index in range(diffusion.steps-1,-1,-1):
        timestep=torch.full((shape[0],),index,device=bench.DEVICE,dtype=torch.long)
        with bench.amp_context(): predicted_noise=model(x,timestep)
        predicted_noise=predicted_noise.float(); alpha_bar=diffusion.ac[index]
        previous=torch.tensor(1.0,device=bench.DEVICE) if index==0 else diffusion.ac[index-1]
        predicted_clean=(x-torch.sqrt(1-alpha_bar)*predicted_noise)/torch.sqrt(alpha_bar)
        predicted_clean=predicted_clean.clamp(-threshold,threshold)
        beta=diffusion.betas[index]; alpha=diffusion.alphas[index]
        coefficient_clean=beta*torch.sqrt(previous)/(1-alpha_bar)
        coefficient_noisy=(1-previous)*torch.sqrt(alpha)/(1-alpha_bar)
        mean=coefficient_clean*predicted_clean+coefficient_noisy*x
        variance=beta*(1-previous)/(1-alpha_bar)
        x=mean+(variance.clamp_min(0).sqrt()*torch.randn_like(x) if index else 0)
    return x


def main():
    bench.OUT=Path("results_final_3models_2datasets_5seeds")
    bench.CFG.vae_steps=3000; bench.CFG.ldm_steps=5000; bench.CFG.diffusion_lr=2e-4
    bench.CFG.classifier_batch_size=256; bench.CFG.generative_batch_size=256; bench.CFG.sampling_batch_size=1000
    bench.CFG.eval_samples=1000; bench.CFG.kid_subset=500; bench.CFG.kid_repeats=10
    train_ds,_,classes,classifier,_=bench.prepare_evaluator("fashion_mnist")
    labels=np.asarray(train_ds.targets,dtype=np.int64); rng=np.random.default_rng(1508)
    train_indices=[]; validation_indices=[]
    for label in range(len(classes)):
        indices=np.flatnonzero(labels==label); rng.shuffle(indices)
        validation_indices.extend(indices[:100]); train_indices.extend(indices[100:])
    rng.shuffle(train_indices)
    train_loader=bench.loader(Subset(train_ds,train_indices),True,bench.CFG.generative_batch_size,drop=True)
    real=torch.stack([train_ds[int(index)][0] for index in validation_indices])
    probabilities,real_features=bench.outputs(classifier,real)
    real_distribution=np.bincount(probabilities.argmax(1),minlength=len(classes))/len(probabilities)

    bench.seed_all(1508); vae=bench.VAE(bench.CFG.latent_channels).to(bench.DEVICE)
    print("Training VAE",flush=True); bench.train_vae(vae,train_loader); vae.eval()
    stats=bench.latent_stats(vae,train_loader); diffusion=bench.Diffusion(bench.CFG.diffusion_steps)
    bench.seed_all(1508); model=bench.UNetDenoiser(bench.CFG.latent_channels,base=64).to(bench.DEVICE)
    print("Training base-64 LDM",flush=True)
    ema,seconds,memory,history=bench.train_diffusion(model,train_loader,diffusion,bench.CFG.ldm_steps,.999,vae,stats)
    torch.save({"vae":vae.state_dict(),"ldm_ema":ema.state_dict(),"latent_mean":stats[0].cpu(),"latent_std":stats[1].cpu()},OUT/"diagnostic_checkpoint.pt")

    rows=[]
    for threshold in (1.5,2.0,2.5,3.0,4.0,6.0):
        print(f"Evaluating clean-latent clip {threshold}",flush=True); start=time.perf_counter()
        standardized=sample_clipped(ema,diffusion,(1000,bench.CFG.latent_channels,8,8),threshold)
        with torch.inference_mode(),bench.amp_context(): samples=vae.decode(standardized*stats[1]+stats[0]).float().cpu()
        bench.synchronize(); sample_seconds=time.perf_counter()-start; latent=standardized.cpu()
        metrics=bench.evaluate(f"ldm_clip_{threshold}",samples,classifier,real_features,real_distribution,seconds,sample_seconds,
                               list(vae.parameters())+list(model.parameters()),memory,memory,history,3210)
        rows.append({"clip_threshold":threshold,"latent_mean":float(latent.mean()),"latent_std":float(latent.std()),
                     "latent_abs_max":float(latent.abs().max()),**{key:value for key,value in metrics.items() if key!="class_distribution"}})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"ldm_clipping_search.csv",index=False)
    winner=frame.sort_values(["custom_fid","class_distribution_jsd"]).iloc[0]
    record={key:(value.item() if hasattr(value,"item") else value) for key,value in winner.items()}
    (OUT/"selected_clipping.json").write_text(json.dumps(record,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for axis,metric,title in zip(axes,("custom_fid","class_distribution_entropy","latent_std"),("Feature FID ↓","Class entropy ↑","Generated latent SD (target ≈1)")):
        axis.plot(frame.clip_threshold,frame[metric],marker="o",color="#70AD47"); axis.set_xlabel("Clean-latent clipping threshold"); axis.set_ylabel(title); axis.grid(alpha=.25)
    fig.suptitle("LDM clean-latent stabilization search (training-derived validation set)")
    fig.tight_layout(); fig.savefig(OUT/"ldm_clipping_search.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    print(frame[["clip_threshold","custom_fid","custom_kid_mean","class_distribution_entropy","class_distribution_jsd","latent_std","latent_abs_max"]].to_string(index=False))
    print("Selected:",record)


if __name__=="__main__": main()
