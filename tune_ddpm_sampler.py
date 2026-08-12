"""Validation-only comparison of pixel-DDPM reverse samplers."""
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


OUT=Path("hyperparameter_search/ddpm_sampler"); OUT.mkdir(parents=True,exist_ok=True)


@torch.inference_mode()
def sample(model,diffusion,shape,method):
    x=torch.randn(shape,device=bench.DEVICE)
    for index in range(diffusion.steps-1,-1,-1):
        timestep=torch.full((shape[0],),index,device=bench.DEVICE,dtype=torch.long)
        with bench.amp_context(): predicted_noise=model(x,timestep)
        predicted_noise=predicted_noise.float(); alpha_bar=diffusion.ac[index]
        previous=torch.tensor(1.0,device=bench.DEVICE) if index==0 else diffusion.ac[index-1]
        beta=diffusion.betas[index]; alpha=diffusion.alphas[index]
        if method=="fixed_large":
            mean=(x-beta/torch.sqrt(1-alpha_bar)*predicted_noise)/torch.sqrt(alpha)
            x=mean+(beta.sqrt()*torch.randn_like(x) if index else 0)
            continue
        predicted_clean=(x-torch.sqrt(1-alpha_bar)*predicted_noise)/torch.sqrt(alpha_bar)
        if "clip" in method: predicted_clean=predicted_clean.clamp(-1,1)
        if method.startswith("ddim"):
            x=torch.sqrt(previous)*predicted_clean+torch.sqrt(1-previous)*predicted_noise
            continue
        coefficient_clean=beta*torch.sqrt(previous)/(1-alpha_bar)
        coefficient_noisy=(1-previous)*torch.sqrt(alpha)/(1-alpha_bar)
        mean=coefficient_clean*predicted_clean+coefficient_noisy*x
        variance=beta*(1-previous)/(1-alpha_bar)
        x=mean+(variance.clamp_min(0).sqrt()*torch.randn_like(x) if index else 0)
    return x


def main():
    bench.OUT=Path("results_final_3models_2datasets_5seeds")
    bench.CFG.ddpm_steps=5000; bench.CFG.diffusion_lr=2e-4; bench.CFG.classifier_batch_size=256
    bench.CFG.generative_batch_size=256; bench.CFG.sampling_batch_size=1000
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
    diffusion=bench.Diffusion(bench.CFG.diffusion_steps); bench.seed_all(1508)
    model=bench.PixelUNet().to(bench.DEVICE); print("Training pixel DDPM",flush=True)
    ema,seconds,memory,history=bench.train_diffusion(model,train_loader,diffusion,bench.CFG.ddpm_steps,.995)
    rows=[]
    for method in ("fixed_large","posterior","posterior_clip","ddim_clip"):
        print(f"Evaluating {method}",flush=True); start=time.perf_counter()
        generated=sample(ema,diffusion,(1000,3,32,32),method)
        samples=((generated+1)/2).clamp(0,1).cpu(); bench.synchronize(); sample_seconds=time.perf_counter()-start
        metrics=bench.evaluate(f"ddpm_{method}",samples,classifier,real_features,real_distribution,seconds,sample_seconds,
                               model.parameters(),memory,memory,history,4220)
        rows.append({"sampler":method,"raw_mean":float(generated.mean()),"raw_std":float(generated.std()),
                     "raw_abs_max":float(generated.abs().max()),**{key:value for key,value in metrics.items() if key!="class_distribution"}})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"ddpm_sampler_search.csv",index=False)
    winner=frame.sort_values(["custom_fid","class_distribution_jsd"]).iloc[0]
    record={key:(value.item() if hasattr(value,"item") else value) for key,value in winner.items()}
    (OUT/"selected_sampler.json").write_text(json.dumps(record,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for axis,metric,title in zip(axes,("custom_fid","class_distribution_entropy","raw_abs_max"),("Feature FID ↓","Class entropy ↑","Raw sample max |x|")):
        axis.bar(frame.sampler,frame[metric],color="#ED7D31"); axis.tick_params(axis="x",rotation=25); axis.set_ylabel(title); axis.grid(axis="y",alpha=.25)
    fig.suptitle("Pixel-DDPM sampler search (training-derived validation set)")
    fig.tight_layout(); fig.savefig(OUT/"ddpm_sampler_search.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    print(frame[["sampler","custom_fid","custom_kid_mean","class_distribution_entropy","class_distribution_jsd","raw_std","raw_abs_max"]].to_string(index=False))
    print("Selected:",record)


if __name__=="__main__": main()
