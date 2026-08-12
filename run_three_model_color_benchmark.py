"""Three-model, two-dataset, five-seed RGB generative benchmark.

Models: convolutional VAE, pixel-space DDPM, and latent diffusion (LDM).
Datasets: Fashion-MNIST and Fashion Product Images Small (FPIS).

The FPIS images are restricted to Apparel, stratified over the ten most common
article types, center-padded, and resized to 32x32 RGB. Fashion-MNIST is
replicated to RGB so model architecture and evaluator capacity remain matched.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets as tv_datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BENCHMARK_OUTPUT_DIR", ROOT / "results_three_models_two_datasets_5seeds"))
DATA = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs_classifier: int = 20
    epochs_vae: int = 25
    epochs_ddpm: int = 35
    epochs_ldm: int = 35
    batch_size: int = 1024
    eval_samples: int = 5000
    diffusion_steps: int = 100
    latent_dim: int = 32
    vae_lr: float = 2e-3
    diffusion_lr: float = 2e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 1.0
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    kid_subset: int = 1000
    kid_repeats: int = 20
    fpis_test_fraction: float = 0.15
    fpis_classes: int = 15


CFG = Config()
if os.environ.get("BENCHMARK_QUICK", "0") == "1":
    CFG.epochs_classifier = CFG.epochs_vae = CFG.epochs_ddpm = CFG.epochs_ldm = 1
    CFG.eval_samples, CFG.diffusion_steps, CFG.kid_subset, CFG.kid_repeats = 256, 20, 128, 3
    CFG.seeds = (0,)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def amp_context():
    return torch.autocast("cuda", dtype=torch.float16) if AMP else nullcontext()


class FPISDataset(Dataset):
    def __init__(self, records, indices, classes):
        self.records, self.indices, self.classes = records, list(indices), classes
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        row = self.records[int(self.indices[i])]
        image = row["image"].convert("RGB")
        w, h = image.size
        side = max(w, h)
        left, top = (side - w) // 2, (side - h) // 2
        image = TF.pad(image, [left, top, side - w - left, side - h - top], fill=255)
        image = TF.resize(image, [32, 32], antialias=True)
        return self.to_tensor(image), self.classes.index(row["articleType"])


def load_datasets(name: str):
    if name == "fashion_mnist":
        transform = transforms.Compose([transforms.Resize((32,32)), transforms.Grayscale(3), transforms.ToTensor()])
        train = tv_datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
        test = tv_datasets.FashionMNIST(DATA, train=False, download=True, transform=transform)
        return train, test, list(train.classes)
    from datasets import load_dataset
    records = load_dataset("Transformersx/fashion-product-images-small", split="train")
    labels = np.asarray(records["articleType"], dtype=object)
    apparel = np.asarray(records["masterCategory"], dtype=object) == "Apparel"
    values, counts = np.unique(labels[apparel], return_counts=True)
    classes = values[np.argsort(counts)[-CFG.fpis_classes:][::-1]].tolist()
    rng = np.random.default_rng(1508)
    train_idx, test_idx = [], []
    for cls in classes:
        idx = np.flatnonzero(apparel & (labels == cls)); rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - CFG.fpis_test_fraction)))
        train_idx.extend(idx[:cut]); test_idx.extend(idx[cut:])
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return FPISDataset(records, train_idx, classes), FPISDataset(records, test_idx, classes), classes


class ClassifierResidual(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__(); self.c1=nn.Conv2d(channels,channels,3,stride,padding=1,bias=False); self.b1=nn.BatchNorm2d(channels); self.c2=nn.Conv2d(channels,channels,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(channels)
    def forward(self,x): return F.silu(self.b2(self.c2(F.silu(self.b1(self.c1(x)))))+x)


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),ClassifierResidual(64),ClassifierResidual(64),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.BatchNorm2d(128),nn.SiLU(),ClassifierResidual(128),ClassifierResidual(128),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1,bias=False),nn.BatchNorm2d(256),nn.SiLU(),ClassifierResidual(256),ClassifierResidual(256),nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x, features=False):
        z = self.body(x).flatten(1)
        return (self.fc(z), z) if features else self.fc(z)


class VAE(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.SiLU(),nn.Flatten())
        self.mu, self.logvar = nn.Linear(64*8*8,latent), nn.Linear(64*8*8,latent)
        self.proj = nn.Linear(latent,64*8*8)
        self.dec = nn.Sequential(nn.Unflatten(1,(64,8,8)),nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),nn.ConvTranspose2d(32,3,4,2,1))
    def encode(self,x):
        h=self.enc(x); return self.mu(h), self.logvar(h)
    def decode_logits(self,z): return self.dec(self.proj(z))
    def decode(self,z): return self.decode_logits(z).sigmoid()
    def forward(self,x):
        mu,lv=self.encode(x); z=mu+torch.exp(.5*lv)*torch.randn_like(mu); return self.decode_logits(z),mu,lv


def tembed(t, dim=128):
    half=dim//2; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
    x=t.float()[:,None]*f[None]; return torch.cat([x.sin(),x.cos()],1)


class ResBlock(nn.Module):
    def __init__(self, ci, co, td=128):
        super().__init__(); self.c1=nn.Conv2d(ci,co,3,padding=1); self.c2=nn.Conv2d(co,co,3,padding=1); self.t=nn.Linear(td,co); self.skip=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity(); self.n1=nn.GroupNorm(8,co); self.n2=nn.GroupNorm(8,co)
    def forward(self,x,e):
        h=F.silu(self.n1(self.c1(x))); h=h+self.t(e)[:,:,None,None]; h=self.c2(F.silu(self.n2(h))); return h+self.skip(x)


class PixelUNet(nn.Module):
    def __init__(self):
        super().__init__(); self.t=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128)); self.r1=ResBlock(3,64); self.d=nn.Conv2d(64,128,4,2,1); self.r2=ResBlock(128,128); self.mid=ResBlock(128,128); self.u=nn.ConvTranspose2d(128,64,4,2,1); self.r3=ResBlock(128,64); self.out=nn.Conv2d(64,3,3,padding=1)
    def forward(self,x,t):
        e=self.t(tembed(t)); a=self.r1(x,e); b=self.r2(self.d(a),e); b=self.mid(b,e); return self.out(self.r3(torch.cat([self.u(b),a],1),e))


class LatentDenoiser(nn.Module):
    def __init__(self, latent=32):
        super().__init__(); self.time=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,256)); self.inp=nn.Linear(latent,256); self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512),nn.SiLU(),nn.Linear(512,256)) for _ in range(6)]); self.out=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,latent))
    def forward(self,z,t):
        h=self.inp(z)+self.time(tembed(t))
        for block in self.blocks: h=h+block(h)
        return self.out(h)


class Diffusion:
    def __init__(self, steps):
        self.steps=steps; s=.008; x=torch.linspace(0,steps,steps+1,device=DEVICE); ac=torch.cos(((x/steps+s)/(1+s))*math.pi*.5)**2; ac=ac/ac[0]; self.betas=torch.clip(1-ac[1:]/ac[:-1],1e-4,.999); self.alphas=1-self.betas; self.ac=torch.cumprod(self.alphas,0)
    def noise(self,x,t):
        eps=torch.randn_like(x); shape=(-1,)+(1,)*(x.ndim-1); return self.ac[t].sqrt().view(shape)*x+(1-self.ac[t]).sqrt().view(shape)*eps,eps
    @torch.no_grad()
    def sample(self,model,shape):
        x=torch.randn(shape,device=DEVICE)
        for i in range(self.steps-1,-1,-1):
            t=torch.full((shape[0],),i,device=DEVICE,dtype=torch.long); b=self.betas[i]; a=self.alphas[i]; ac=self.ac[i]
            mean=(x-(b/torch.sqrt(1-ac))*model(x,t))/torch.sqrt(a)
            x=mean+(b.sqrt()*torch.randn_like(x) if i else 0)
        return x


def loader(ds, shuffle, drop=False):
    return DataLoader(ds,batch_size=CFG.batch_size,shuffle=shuffle,drop_last=drop,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=4)


def train_classifier(model, train, test):
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.classifier_lr,weight_decay=CFG.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,CFG.epochs_classifier)
    for _ in range(CFG.epochs_classifier):
        model.train()
        for x,y in train:
            x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            # Tensor-space augmentation works for both dataset wrappers and is
            # deliberately limited so product orientation remains meaningful.
            if torch.rand(()) < .5: x=torch.flip(x,(-1,))
            with amp_context(): loss=F.cross_entropy(model(x),y)
            loss.backward(); opt.step()
        sch.step()
    model.eval(); correct=n=0; per_correct=torch.zeros(model.fc.out_features); per_total=torch.zeros(model.fc.out_features); losses=[]
    with torch.no_grad():
        for x,y in test:
            logits=model(x.to(DEVICE)); pred=logits.argmax(1).cpu(); correct+=(pred==y).sum().item(); n+=len(y); losses.append(F.cross_entropy(logits,y.to(DEVICE)).item()*len(y))
            for c in range(model.fc.out_features): per_correct[c]+=((pred==c)&(y==c)).sum(); per_total[c]+=(y==c).sum()
    return {"accuracy":correct/n,"balanced_accuracy":float((per_correct/per_total.clamp_min(1)).mean()),"test_loss":sum(losses)/n}


def train_vae(model, train):
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.vae_lr,weight_decay=CFG.weight_decay); start=time.perf_counter()
    for _ in range(CFG.epochs_vae):
        model.train()
        for x,_ in train:
            x=x.to(DEVICE,non_blocking=True); opt.zero_grad(set_to_none=True)
            with amp_context(): rec,mu,lv=model(x)
            rec_loss=F.binary_cross_entropy_with_logits(rec.float(),x.float(),reduction="sum")/len(x)
            loss=rec_loss+CFG.beta*(-.5*(1+lv.float()-mu.float().square()-lv.float().exp()).sum()/len(x))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip); opt.step()
    return time.perf_counter()-start


def train_diffusion(model, train, diffusion, epochs, encoder=None, latent_stats=None):
    opt=torch.optim.AdamW(model.parameters(),lr=CFG.diffusion_lr,weight_decay=CFG.weight_decay); ema=copy.deepcopy(model).eval(); start=time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x,_ in train:
            x=x.to(DEVICE,non_blocking=True)
            if encoder:
                with torch.no_grad(): x=encoder.encode(x)[0]; x=(x-latent_stats[0])/latent_stats[1]
            else: x=x*2-1
            t=torch.randint(0,diffusion.steps,(len(x),),device=DEVICE); noisy,eps=diffusion.noise(x,t); opt.zero_grad(set_to_none=True)
            with amp_context(): loss=F.mse_loss(model(noisy,t),eps)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip); opt.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p,1-CFG.ema_decay)
    return ema,time.perf_counter()-start


@torch.no_grad()
def latent_stats(vae, train):
    values=[]
    for x,_ in train: values.append(vae.encode(x.to(DEVICE))[0].cpu())
    z=torch.cat(values); return z.mean(0,keepdim=True).to(DEVICE),z.std(0,keepdim=True).clamp_min(1e-4).to(DEVICE)


@torch.no_grad()
def outputs(classifier, images):
    ps,fs=[],[]
    for x in images.split(CFG.batch_size):
        logits,f=classifier(x.to(DEVICE),True); ps.append(logits.softmax(1).cpu()); fs.append(f.cpu())
    return torch.cat(ps).numpy(),torch.cat(fs).numpy()


def fid(a,b):
    ma,mb=a.mean(0),b.mean(0); ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False); root=sqrtm(ca@cb)
    if np.iscomplexobj(root): root=root.real
    return float((ma-mb)@(ma-mb)+np.trace(ca+cb-2*root))


def kid(a,b,seed):
    rng=np.random.default_rng(seed); vals=[]; n=min(CFG.kid_subset,len(a),len(b)); d=a.shape[1]
    for _ in range(CFG.kid_repeats):
        x=a[rng.choice(len(a),n,False)]; y=b[rng.choice(len(b),n,False)]; kxx=(x@x.T/d+1)**3; kyy=(y@y.T/d+1)**3; kxy=(x@y.T/d+1)**3
        vals.append((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(n*(n-1))-2*kxy.mean())
    return float(np.mean(vals)),float(np.std(vals,ddof=1))


def save_grid(images,path,title):
    path.parent.mkdir(parents=True,exist_ok=True); grid=make_grid(images[:64],8,padding=2); plt.figure(figsize=(8,8)); plt.imshow(grid.permute(1,2,0).squeeze(),cmap="gray"); plt.axis("off"); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()


def evaluate(name, samples, classifier, real_features, train_s, sample_s, params, seed):
    p,f=outputs(classifier,samples); km,ks=kid(real_features,f,seed); counts=np.bincount(p.argmax(1),minlength=p.shape[1])/len(p); ent=float(-(counts[counts>0]*np.log(counts[counts>0])).sum()/np.log(len(counts)))
    dx=torch.abs(samples[:,:,:,1:]-samples[:,:,:,:-1]).mean(); dy=torch.abs(samples[:,:,1:,:]-samples[:,:,:-1,:]).mean()
    py=p.mean(0); inception=float(np.exp(np.mean(np.sum(p*(np.log(p+1e-12)-np.log(py+1e-12)),axis=1))))
    rng=np.random.default_rng(seed); ix=rng.integers(0,len(f),size=(min(10000,len(f)*2),2)); pair=float(np.linalg.norm(f[ix[:,0]]-f[ix[:,1]],axis=1).mean())
    return {"model":name,"custom_fid":fid(real_features,f),"custom_kid_mean":km,"custom_kid_std":ks,"mean_classifier_confidence":float(p.max(1).mean()),"median_classifier_confidence":float(np.median(p.max(1))),"recognizability_rate_at_0.8":float((p.max(1)>=.8).mean()),"class_coverage":int((counts>0).sum()),"class_distribution_entropy":ent,"inception_score":inception,"feature_pairwise_distance":pair,"sobel_sharpness":float((dx+dy).item()),"training_minutes":train_s/60,"sampling_total_seconds":sample_s,"sampling_ms_per_image":sample_s/len(samples)*1000,"trainable_parameters":sum(x.numel() for x in params if x.requires_grad)}


def run_one(dataset_name,seed):
    run=OUT/dataset_name/f"seed_{seed:02d}"; result=run/"results.json"
    if result.exists(): return json.loads(result.read_text())
    run.mkdir(parents=True,exist_ok=True); seed_all(seed); train_ds,test_ds,classes=load_datasets(dataset_name); tr,te=loader(train_ds,True),loader(test_ds,False)
    classifier=Classifier(len(classes)).to(DEVICE); classifier_metrics=train_classifier(classifier,tr,te)
    vae=VAE(CFG.latent_dim).to(DEVICE); vt=train_vae(vae,tr)
    diffusion=Diffusion(CFG.diffusion_steps); ddpm=PixelUNet().to(DEVICE); ddpm_ema,dt=train_diffusion(ddpm,tr,diffusion,CFG.epochs_ddpm)
    zs=latent_stats(vae,tr); ldm=LatentDenoiser(CFG.latent_dim).to(DEVICE); ldm_ema,lt=train_diffusion(ldm,tr,diffusion,CFG.epochs_ldm,vae,zs)
    n=min(CFG.eval_samples,len(test_ds)); real=torch.stack([test_ds[i][0] for i in range(n)]); _,rf=outputs(classifier,real)
    start=time.perf_counter(); vs=vae.decode(torch.randn(n,CFG.latent_dim,device=DEVICE)).cpu(); vst=time.perf_counter()-start
    all_dd=[]; start=time.perf_counter()
    for i in range(0,n,CFG.batch_size): all_dd.append(((diffusion.sample(ddpm_ema,(min(CFG.batch_size,n-i),3,32,32))+1)/2).clamp(0,1).cpu())
    ds=torch.cat(all_dd); dst=time.perf_counter()-start
    all_ld=[]; start=time.perf_counter()
    for i in range(0,n,CFG.batch_size):
        z=diffusion.sample(ldm_ema,(min(CFG.batch_size,n-i),CFG.latent_dim)); all_ld.append(vae.decode(z*zs[1]+zs[0]).cpu())
    ls=torch.cat(all_ld); lst=time.perf_counter()-start
    rows=[evaluate("VAE",vs,classifier,rf,vt,vst,vae.parameters(),seed),evaluate("DDPM",ds,classifier,rf,dt,dst,ddpm.parameters(),seed+10),evaluate("LDM",ls,classifier,rf,vt+lt,lst,list(vae.parameters())+list(ldm.parameters()),seed+20)]
    for name,x in [("real",real),("vae",vs),("ddpm",ds),("ldm",ls)]: save_grid(x,run/"figures"/f"{name}_samples.png",f"{dataset_name}: {name.upper()} (seed {seed})")
    pd.DataFrame(rows).to_csv(run/"model_metrics.csv",index=False); payload={"dataset":dataset_name,"seed":seed,"classes":classes,"train_size":len(train_ds),"test_size":len(test_ds),"classifier":classifier_metrics,"models":rows}; result.write_text(json.dumps(payload,indent=2)); return payload


def aggregate(results):
    records=[]
    for r in results:
        for m in r["models"]: records.append({"dataset":r["dataset"],"seed":r["seed"],**{f"classifier_{k}":v for k,v in r["classifier"].items()},**m})
    raw=pd.DataFrame(records); agg=raw.groupby(["dataset","model"]).agg(["mean","std"]); (OUT/"aggregate"/"figures").mkdir(parents=True,exist_ok=True); raw.to_csv(OUT/"aggregate"/"all_runs.csv",index=False); agg.to_csv(OUT/"aggregate"/"mean_std.csv")
    metrics=["custom_fid","custom_kid_mean","mean_classifier_confidence","recognizability_rate_at_0.8","class_distribution_entropy","training_minutes","sampling_ms_per_image"]
    for metric in metrics:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(ds,g) in zip(axes,raw.groupby("dataset")):
            order=["VAE","DDPM","LDM"]; means=g.groupby("model")[metric].mean().reindex(order); std=g.groupby("model")[metric].std().reindex(order); ax.bar(order,means,yerr=std,capsize=5,color=["#4472C4","#ED7D31","#70AD47"]); ax.set_title(ds.replace("_"," ").title()); ax.set_ylabel(metric.replace("_"," "))
        fig.suptitle(f"Five-seed mean ± SD: {metric.replace('_',' ')}"); fig.tight_layout(); fig.savefig(OUT/"aggregate"/"figures"/f"{metric}.png",dpi=200); plt.close(fig)
    summary=["# Extended benchmark: five-seed results","", "Models: VAE, pixel-space DDPM, and latent diffusion. Datasets: Fashion-MNIST and Fashion Product Images Small.","", "Values are mean ± sample standard deviation across seeds 0–4.",""]
    for ds in raw.dataset.unique():
        summary += [f"## {ds.replace('_',' ').title()}",""]
        for metric in metrics:
            vals=[]
            for model in ["VAE","DDPM","LDM"]:
                x=raw[(raw.dataset==ds)&(raw.model==model)][metric]; vals.append(f"{model}: {x.mean():.4f} ± {x.std():.4f}")
            summary.append(f"- {metric.replace('_',' ')} — "+"; ".join(vals))
        summary.append("")
    (OUT/"aggregate"/"RESULTS_SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"config.json").write_text(json.dumps(asdict(CFG),indent=2)); print(f"Device: {DEVICE}")
    results=[]
    for dataset in ("fashion_mnist","fashion_product_images"):
        for seed in CFG.seeds:
            print(f"\n=== {dataset} / seed {seed} ===",flush=True); results.append(run_one(dataset,seed))
    aggregate(results); print(f"\nComplete: {OUT.resolve()}")


if __name__ == "__main__": main()
