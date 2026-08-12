"""Validation-only diagnosis of LDM capacity and reverse-process variance."""
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


OUT=Path("hyperparameter_search/ldm_diagnostic"); OUT.mkdir(parents=True,exist_ok=True)


@torch.inference_mode()
def reverse_sample(model,diffusion,shape,method):
    x=torch.randn(shape,device=bench.DEVICE)
    for index in range(diffusion.steps-1,-1,-1):
        timestep=torch.full((shape[0],),index,device=bench.DEVICE,dtype=torch.long)
        with bench.amp_context(): predicted=model(x,timestep)
        predicted=predicted.float(); alpha_bar=diffusion.ac[index]
        if method=="fixed_large":
            beta=diffusion.betas[index]; alpha=diffusion.alphas[index]
            mean=(x-beta/torch.sqrt(1-alpha_bar)*predicted)/torch.sqrt(alpha)
            x=mean+(beta.sqrt()*torch.randn_like(x) if index else 0)
        elif method=="posterior":
            beta=diffusion.betas[index]; alpha=diffusion.alphas[index]
            mean=(x-beta/torch.sqrt(1-alpha_bar)*predicted)/torch.sqrt(alpha)
            previous=torch.tensor(1.0,device=bench.DEVICE) if index==0 else diffusion.ac[index-1]
            variance=beta*(1-previous)/(1-alpha_bar)
            x=mean+(variance.clamp_min(0).sqrt()*torch.randn_like(x) if index else 0)
        elif method=="ddim":
            predicted_clean=(x-torch.sqrt(1-alpha_bar)*predicted)/torch.sqrt(alpha_bar)
            previous=torch.tensor(1.0,device=bench.DEVICE) if index==0 else diffusion.ac[index-1]
            x=torch.sqrt(previous)*predicted_clean+torch.sqrt(1-previous)*predicted
        else:
            raise ValueError(method)
    return x


@torch.inference_mode()
def evaluate_sampler(model,vae,stats,diffusion,classifier,real_features,real_distribution,history,memory,name,method):
    samples=[]; latent_moments=[]; start=time.perf_counter()
    for index in range(0,1000,bench.CFG.sampling_batch_size):
        batch=min(bench.CFG.sampling_batch_size,1000-index)
        standardized=reverse_sample(model,diffusion,(batch,bench.CFG.latent_channels,8,8),method)
        latent_moments.append(standardized.cpu())
        with bench.amp_context(): decoded=vae.decode(standardized*stats[1]+stats[0])
        samples.append(decoded.float().cpu())
    bench.synchronize(); sample_seconds=time.perf_counter()-start
    samples=torch.cat(samples); latents=torch.cat(latent_moments)
    metrics=bench.evaluate(name,samples,classifier,real_features,real_distribution,0.0,sample_seconds,model.parameters(),memory,memory,history,3108)
    return {"sampler":method,"latent_mean":float(latents.mean()),"latent_std":float(latents.std()),
            "latent_abs_max":float(latents.abs().max()),**{key:value for key,value in metrics.items() if key!="class_distribution"}}


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
    rng.shuffle(train_indices); validation_indices=np.asarray(validation_indices)
    train_loader=bench.loader(Subset(train_ds,train_indices),True,bench.CFG.generative_batch_size,drop=True)
    real=torch.stack([train_ds[int(index)][0] for index in validation_indices])
    probabilities,real_features=bench.outputs(classifier,real)
    real_distribution=np.bincount(probabilities.argmax(1),minlength=len(classes))/len(probabilities)

    bench.seed_all(1508); vae=bench.VAE(bench.CFG.latent_channels).to(bench.DEVICE)
    print("Training diagnostic VAE",flush=True); bench.train_vae(vae,train_loader); vae.eval()
    stats=bench.latent_stats(vae,train_loader); diffusion=bench.Diffusion(bench.CFG.diffusion_steps); rows=[]
    for base in (32,64):
        print(f"Training LDM base={base}",flush=True); bench.seed_all(1508)
        model=bench.UNetDenoiser(bench.CFG.latent_channels,base=base).to(bench.DEVICE)
        ema,seconds,memory,history=bench.train_diffusion(model,train_loader,diffusion,bench.CFG.ldm_steps,.999,vae,stats)
        for method in ("fixed_large","posterior","ddim"):
            print(f"Evaluating base={base}, sampler={method}",flush=True)
            row=evaluate_sampler(ema,vae,stats,diffusion,classifier,real_features,real_distribution,history,memory,
                                 f"ldm_base{base}_{method}",method)
            rows.append({"base":base,"training_seconds":seconds,**row})
        del model,ema; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(OUT/"ldm_diagnostic_partial.csv",index=False)

    frame=pd.DataFrame(rows); frame.to_csv(OUT/"ldm_diagnostic.csv",index=False)
    winner=frame.sort_values(["custom_fid","class_distribution_jsd"]).iloc[0]
    (OUT/"selected_ldm_revision.json").write_text(json.dumps({key:(value.item() if hasattr(value,"item") else value) for key,value in winner.items()},indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,metric in zip(axes,("custom_fid","class_distribution_entropy")):
        labels=[f"b{row.base}\n{row.sampler}" for row in frame.itertuples()]
        axis.bar(range(len(frame)),frame[metric],color=["#70AD47" if base==64 else "#4472C4" for base in frame.base])
        axis.set_xticks(range(len(frame)),labels,rotation=25,ha="right"); axis.set_ylabel(metric.replace("_"," "))
    fig.suptitle("LDM capacity and sampler diagnostic (training-derived validation set)")
    fig.tight_layout(); fig.savefig(OUT/"ldm_diagnostic.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    print(frame[["base","sampler","custom_fid","custom_kid_mean","class_distribution_entropy","class_distribution_jsd","latent_std","latent_abs_max"]].to_string(index=False))
    print("Selected:",winner.to_dict())


if __name__=="__main__": main()
