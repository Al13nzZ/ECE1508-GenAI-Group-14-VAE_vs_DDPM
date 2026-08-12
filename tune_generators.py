"""Bounded pilot search for generator optimization hyperparameters.

Candidates are evaluated on Fashion-MNIST seed 1508 with 1,000 samples.  The
search is intentionally small enough to reproduce on one consumer GPU; final
claims come from the independent five-seed benchmark, not this pilot.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

import run_three_model_color_benchmark as bench


OUT=Path("hyperparameter_search/generators"); OUT.mkdir(parents=True,exist_ok=True)


@torch.no_grad()
def sample_vae(model,n):
    start=time.perf_counter(); samples=[]
    for i in range(0,n,bench.CFG.generative_batch_size):
        size=min(bench.CFG.generative_batch_size,n-i)
        samples.append(model.decode(torch.randn(size,bench.CFG.latent_channels,8,8,device=bench.DEVICE)).cpu())
    return torch.cat(samples),time.perf_counter()-start


@torch.no_grad()
def sample_diffusion(model,diffusion,n,channels,size,decoder=None,stats=None):
    start=time.perf_counter(); samples=[]
    for i in range(0,n,bench.CFG.generative_batch_size):
        batch=min(bench.CFG.generative_batch_size,n-i)
        generated=diffusion.sample(model,(batch,channels,size,size))
        if decoder is None: generated=((generated+1)/2).clamp(0,1)
        else: generated=decoder.decode(generated*stats[1]+stats[0])
        samples.append(generated.cpu())
    return torch.cat(samples),time.perf_counter()-start


def main():
    bench.OUT=OUT/"evaluator"; bench.CFG.eval_samples=1000; bench.CFG.kid_subset=500; bench.CFG.kid_repeats=10
    bench.CFG.classifier_batch_size=256; bench.CFG.generative_batch_size=256
    train_ds,_,classes,classifier,_=bench.prepare_evaluator("fashion_mnist")
    # Hyperparameters must never be selected on the official test split.  Hold
    # out exactly 100 training images per Fashion-MNIST class for this pilot.
    labels=np.asarray(train_ds.targets,dtype=np.int64); rng=np.random.default_rng(1508)
    train_indices=[]; validation_indices=[]
    for label in range(len(classes)):
        indices=np.flatnonzero(labels==label); rng.shuffle(indices)
        validation_indices.extend(indices[:100]); train_indices.extend(indices[100:])
    rng.shuffle(train_indices); rng.shuffle(validation_indices)
    (OUT/"search_protocol.json").write_text(json.dumps({
        "selection_dataset":"Fashion-MNIST training split",
        "selection_seed":1508,
        "generator_training_images":len(train_indices),
        "held_out_validation_images":len(validation_indices),
        "official_test_images_used_for_selection":0,
        "selection_metric":"classifier-feature FID",
    },indent=2))
    train_loader=bench.loader(Subset(train_ds,train_indices),True,bench.CFG.generative_batch_size,drop=True)
    real=torch.stack([train_ds[index][0] for index in validation_indices])
    real_p,real_features=bench.outputs(classifier,real)
    real_distribution=np.bincount(real_p.argmax(1),minlength=len(classes))/len(real_p)
    diffusion=bench.Diffusion(bench.CFG.diffusion_steps); rows=[]; trained_vaes={}

    vae_candidates=[
        {"name":"vae_lr3e-4_beta1","lr":3e-4,"beta":1.0},
        {"name":"vae_lr1e-3_beta1","lr":1e-3,"beta":1.0},
        {"name":"vae_lr1e-3_beta0.5","lr":1e-3,"beta":0.5},
    ]
    for index,candidate in enumerate(vae_candidates):
        print(candidate["name"],flush=True); bench.seed_all(1508); bench.CFG.vae_lr=candidate["lr"]; bench.CFG.beta=candidate["beta"]; bench.CFG.vae_steps=1500
        model=bench.VAE(bench.CFG.latent_channels).to(bench.DEVICE); seconds,memory,history=bench.train_vae(model,train_loader)
        samples,sample_seconds=sample_vae(model,1000)
        metrics=bench.evaluate(candidate["name"],samples,classifier,real_features,real_distribution,seconds,sample_seconds,model.parameters(),memory,memory,history,100+index)
        rows.append({"family":"VAE","candidate":candidate["name"],"lr":candidate["lr"],"beta":candidate["beta"],"ema":np.nan,**{k:v for k,v in metrics.items() if k!="class_distribution"}})
        trained_vaes[candidate["name"]]=copy.deepcopy(model).cpu()

    best_vae=min((row for row in rows if row["family"]=="VAE"),key=lambda row:row["custom_fid"])
    vae=trained_vaes[best_vae["candidate"]].to(bench.DEVICE).eval(); stats=bench.latent_stats(vae,train_loader)
    diffusion_candidates=[
        {"lr":1e-4,"ema":.995},{"lr":2e-4,"ema":.995},
        {"lr":1e-4,"ema":.999},{"lr":2e-4,"ema":.999},
    ]
    for family in ("DDPM","LDM"):
        for index,candidate in enumerate(diffusion_candidates):
            name=f"{family.lower()}_lr{candidate['lr']:.0e}_ema{candidate['ema']}"; print(name,flush=True)
            bench.seed_all(1508); bench.CFG.diffusion_lr=candidate["lr"]
            model=bench.PixelUNet().to(bench.DEVICE) if family=="DDPM" else bench.LatentDenoiser(bench.CFG.latent_channels).to(bench.DEVICE)
            ema,seconds,memory,history=bench.train_diffusion(model,train_loader,diffusion,2000,candidate["ema"],vae if family=="LDM" else None,stats if family=="LDM" else None)
            samples,sample_seconds=sample_diffusion(ema,diffusion,1000,3 if family=="DDPM" else bench.CFG.latent_channels,32 if family=="DDPM" else 8,vae if family=="LDM" else None,stats if family=="LDM" else None)
            metrics=bench.evaluate(name,samples,classifier,real_features,real_distribution,seconds,sample_seconds,model.parameters(),memory,memory,history,200+index+(100 if family=="LDM" else 0))
            rows.append({"family":family,"candidate":name,"lr":candidate["lr"],"beta":np.nan,"ema":candidate["ema"],**{k:v for k,v in metrics.items() if k!="class_distribution"}})

    frame=pd.DataFrame(rows); frame.to_csv(OUT/"generator_search.csv",index=False)
    winners=frame.loc[frame.groupby("family").custom_fid.idxmin()].sort_values("family")
    selected=winners[["family","candidate","lr","beta","ema","custom_fid","custom_kid_mean","class_distribution_entropy"]]
    records=json.loads(selected.to_json(orient="records"))
    (OUT/"selected_hyperparameters.json").write_text(json.dumps(records,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,(family,group) in zip(axes,frame.groupby("family",sort=True)):
        axis.bar(range(len(group)),group.custom_fid,color="#4472C4"); axis.set_xticks(range(len(group)),group.candidate,rotation=30,ha="right"); axis.set_ylabel("Pilot feature FID ↓"); axis.set_title(family)
    fig.suptitle("Bounded generator hyperparameter search: Fashion-MNIST seed 1508")
    fig.tight_layout(); fig.savefig(OUT/"generator_search.png",dpi=220); plt.close(fig)
    print(frame[["family","candidate","custom_fid","custom_kid_mean","class_distribution_entropy"]].to_string(index=False)); print("\nSelected:\n",winners.to_string(index=False))


if __name__=="__main__": main()
