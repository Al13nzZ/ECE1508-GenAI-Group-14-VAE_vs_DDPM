"""Create the curated figure set used by the final LaTeX report."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


RESULTS=Path("results_final_3models_2datasets_5seeds")
OUT=Path("project_final_report/figures"); OUT.mkdir(parents=True,exist_ok=True)
TABLES=Path("project_final_report/tables"); TABLES.mkdir(parents=True,exist_ok=True)
MODELS=["VAE","DDPM","LDM"]; COLORS={"VAE":"#4472C4","DDPM":"#ED7D31","LDM":"#70AD47"}
DATASET_NAMES={"fashion_mnist":"Fashion-MNIST","fashion_product_images":"Fashion Product Images (RGB, 15 classes)"}


def grouped_metric(raw,metrics,filename,titles=None,log_metrics=()):
    fig,axes=plt.subplots(1,len(metrics),figsize=(5.3*len(metrics),4.5),squeeze=False)
    for axis,metric in zip(axes[0],metrics):
        width=.24; datasets=list(DATASET_NAMES)
        for offset,model in enumerate(MODELS):
            means=[]; errors=[]
            for dataset in datasets:
                values=raw[(raw.dataset==dataset)&(raw.model==model)][metric]
                means.append(values.mean()); errors.append(values.std())
            x=np.arange(len(datasets))+(offset-1)*width
            axis.bar(x,means,width,yerr=errors,capsize=4,label=model,color=COLORS[model])
        axis.set_xticks(np.arange(len(datasets)),["Fashion-MNIST","Color products"])
        axis.set_ylabel(titles.get(metric,metric.replace("_"," ")) if titles else metric.replace("_"," "))
        if metric in log_metrics: axis.set_yscale("log")
        axis.grid(axis="y",alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(OUT/filename,dpi=240,bbox_inches="tight"); plt.close(fig)


def architecture_figure():
    fig,axes=plt.subplots(3,1,figsize=(12,5.8)); rows=[
        ("VAE",[("Gaussian prior\n4×8×8","#E2F0D9"),("VAE decoder","#BDD7EE"),("Generated image","#D9EAF7")]),
        ("Pixel DDPM",[("Gaussian noise\n3×32×32","#FCE4D6"),("100-step pixel\ndenoising U-Net","#F4B183"),("Generated image","#D9EAF7")]),
        ("Latent diffusion",[("Gaussian noise\n4×8×8","#E2F0D9"),("100-step latent\ndenoising U-Net","#A9D18E"),("VAE decoder","#BDD7EE"),("Generated image","#D9EAF7")]),
    ]
    for axis,(label,boxes) in zip(axes,rows):
        axis.set_xlim(0,12); axis.set_ylim(0,1); axis.axis("off"); axis.text(.05,.5,label,va="center",weight="bold",fontsize=12)
        start=2.0; gap=(9.5)/len(boxes)
        for index,(text,color) in enumerate(boxes):
            x=start+index*gap; patch=FancyBboxPatch((x,.17),gap*.72,.66,boxstyle="round,pad=0.03",facecolor=color,edgecolor="#44546A")
            axis.add_patch(patch); axis.text(x+gap*.36,.5,text,ha="center",va="center",fontsize=9)
            if index<len(boxes)-1: axis.annotate("",xy=(x+gap,.5),xytext=(x+gap*.73,.5),arrowprops={"arrowstyle":"->","lw":1.4})
    fig.suptitle("Three generative systems and their sampling paths",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"model_architectures.png",dpi=240,bbox_inches="tight"); plt.close(fig)


def qualitative_figure(dataset):
    base=RESULTS/dataset/"seed_00"/"figures"; items=[("Real","real_samples.png"),("VAE","vae_samples.png"),("Pixel DDPM","ddpm_samples.png"),("Latent diffusion","ldm_samples.png")]
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for axis,(title,name) in zip(axes.flat,items):
        axis.imshow(plt.imread(base/name)); axis.axis("off"); axis.set_title(title,weight="bold")
    fig.suptitle(DATASET_NAMES[dataset]+": representative seed-0 grids",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/f"qualitative_{dataset}.png",dpi=200,bbox_inches="tight"); plt.close(fig)


def training_curves():
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,dataset in enumerate(DATASET_NAMES):
        for col,(model,file,column) in enumerate([("VAE","vae_history.csv","loss"),("DDPM","ddpm_history.csv","noise_mse"),("LDM","ldm_history.csv","noise_mse")]):
            series=[]
            for seed in range(5): series.append(pd.read_csv(RESULTS/dataset/f"seed_{seed:02d}"/"logs"/file)[column].to_numpy())
            values=np.vstack(series); steps=pd.read_csv(RESULTS/dataset/"seed_00"/"logs"/file)["step"]
            mean=values.mean(0); sample_sd=values.std(0,ddof=1)
            axes[row,col].plot(steps,mean,color=COLORS[model]); axes[row,col].fill_between(steps,mean-sample_sd,mean+sample_sd,color=COLORS[model],alpha=.2)
            axes[row,col].set_title(f"{DATASET_NAMES[dataset]} — {model}"); axes[row,col].set_xlabel("Optimizer step"); axes[row,col].set_ylabel("ELBO loss" if model=="VAE" else "Noise MSE"); axes[row,col].grid(alpha=.2)
    fig.suptitle("Training convergence: mean ± SD over five seeds",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"training_curves.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def class_distributions():
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    distribution_colors={"real":"#7F7F7F","vae":COLORS["VAE"],"ddpm":COLORS["DDPM"],"ldm":COLORS["LDM"]}
    for axis,dataset in zip(axes,DATASET_NAMES):
        frames=[pd.read_csv(RESULTS/dataset/f"seed_{seed:02d}"/"class_distribution.csv") for seed in range(5)]
        average=pd.concat(frames).groupby("class",sort=False).mean(numeric_only=True)
        x=np.arange(len(average)); width=.2
        for offset,column in enumerate(["real","vae","ddpm","ldm"]):
            axis.bar(x+(offset-1.5)*width,average[column],width,label=column.upper() if column!="real" else "Real",color=distribution_colors[column])
        axis.set_xticks(x,average.index,rotation=35,ha="right"); axis.set_ylabel("Predicted proportion"); axis.set_title(DATASET_NAMES[dataset]); axis.legend(ncol=4); axis.grid(axis="y",alpha=.2)
    fig.suptitle("Class distributions averaged across five seeds",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"class_distributions.png",dpi=220,bbox_inches="tight"); plt.close(fig)


def quality_compute(raw):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,(dataset,group) in zip(axes,raw.groupby("dataset")):
        for model in MODELS:
            values=group[group.model==model]
            axis.errorbar(values.sampling_ms_per_image.mean(),values.custom_fid.mean(),xerr=values.sampling_ms_per_image.std(),yerr=values.custom_fid.std(),fmt="o",ms=9,capsize=4,color=COLORS[model],label=model)
        axis.set_xscale("log"); axis.set_xlabel("Batched sampling (ms/image, log scale)"); axis.set_ylabel("Feature FID ↓"); axis.set_title(DATASET_NAMES[dataset]); axis.grid(alpha=.25); axis.legend()
    fig.suptitle("Quality–compute trade-off: mean ± SD",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"quality_compute_tradeoff.png",dpi=240,bbox_inches="tight"); plt.close(fig)


def seed_variability(raw):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,(dataset,group) in zip(axes,raw.groupby("dataset")):
        for model in MODELS:
            values=group[group.model==model].sort_values("seed"); axis.plot(values.seed,values.custom_fid,marker="o",label=model,color=COLORS[model])
        axis.set_xticks(range(5)); axis.set_xlabel("Seed"); axis.set_ylabel("Feature FID ↓"); axis.set_title(DATASET_NAMES[dataset]); axis.grid(alpha=.25); axis.legend()
    fig.suptitle("Run-to-run variability",weight="bold"); fig.tight_layout(); fig.savefig(OUT/"seed_variability.png",dpi=240,bbox_inches="tight"); plt.close(fig)


def classifier_figure():
    rows=[]
    for dataset in DATASET_NAMES:
        values=json.loads((RESULTS/"evaluators"/f"{dataset}_metrics.json").read_text()); rows.append({"dataset":dataset,**values})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"classifier_summary.csv",index=False)
    x=np.arange(len(frame)); width=.35; fig,axis=plt.subplots(figsize=(7,4.5))
    axis.bar(x-width/2,frame.accuracy*100,width,label="Accuracy",color="#4472C4"); axis.bar(x+width/2,frame.balanced_accuracy*100,width,label="Balanced accuracy",color="#70AD47")
    axis.set_xticks(x,["Fashion-MNIST","Color products"]); axis.set_ylabel("Test score (%)"); axis.set_ylim(0,100); axis.legend(); axis.grid(axis="y",alpha=.2)
    axis.set_title("Frozen evaluator performance"); fig.tight_layout(); fig.savefig(OUT/"classifier_performance.png",dpi=240,bbox_inches="tight"); plt.close(fig)


def pm(series,scale=1.0,decimals=3):
    return f"{series.mean()*scale:.{decimals}f} $\\pm$ {series.std()*scale:.{decimals}f}"


def latex_tables(raw):
    lines=[r"\begin{tabular}{llrrrrrr}",r"\toprule",
           r"Dataset & Model & Feature FID $\downarrow$ & Feature KID $\downarrow$ & Confidence $\uparrow$ & Recog. (\%) $\uparrow$ & Entropy $\uparrow$ & JSD $\downarrow$ \\",r"\midrule"]
    for dataset in DATASET_NAMES:
        for index,model in enumerate(MODELS):
            group=raw[(raw.dataset==dataset)&(raw.model==model)]
            label=DATASET_NAMES[dataset] if index==0 else ""
            lines.append(f"{label} & {model} & {pm(group.custom_fid,decimals=2)} & {pm(group.custom_kid_mean)} & {pm(group.mean_classifier_confidence)} & {pm(group['recognizability_rate_at_0.8'],100,1)} & {pm(group.class_distribution_entropy)} & {pm(group.class_distribution_jsd)} \\\\")
        if dataset!=list(DATASET_NAMES)[-1]: lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TABLES/"quality_results.tex").write_text("\n".join(lines),encoding="utf-8")

    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",
           r"Dataset & Model & Train (min) $\downarrow$ & Batched sample (ms/image) $\downarrow$ & Parameters (M) $\downarrow$ & Peak GPU (MB) $\downarrow$ \\",r"\midrule"]
    for dataset in DATASET_NAMES:
        for index,model in enumerate(MODELS):
            group=raw[(raw.dataset==dataset)&(raw.model==model)]
            label=DATASET_NAMES[dataset] if index==0 else ""
            lines.append(f"{label} & {model} & {pm(group.training_minutes,decimals=2)} & {pm(group.sampling_ms_per_image,decimals=3)} & {pm(group.trainable_parameters/1e6,decimals=3)} & {pm(group.peak_gpu_memory_mb,decimals=0)} \\\\")
        if dataset!=list(DATASET_NAMES)[-1]: lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TABLES/"compute_results.tex").write_text("\n".join(lines),encoding="utf-8")

    classifier_lines=[r"\begin{tabular}{lrrrr}",r"\toprule",
                      r"Dataset & Test images & Classes & Accuracy (\%) $\uparrow$ & Balanced accuracy (\%) $\uparrow$ \\",r"\midrule"]
    for dataset in DATASET_NAMES:
        values=json.loads((RESULTS/"evaluators"/f"{dataset}_metrics.json").read_text())
        seed_result=json.loads((RESULTS/dataset/"seed_00"/"results.json").read_text())
        classifier_lines.append(f"{DATASET_NAMES[dataset]} & {seed_result['test_size']:,} & {len(values['classes'])} & {100*values['accuracy']:.2f} & {100*values['balanced_accuracy']:.2f} \\\\")
    classifier_lines += [r"\bottomrule",r"\end{tabular}"]
    (TABLES/"classifier_results.tex").write_text("\n".join(classifier_lines),encoding="utf-8")

    ranking=[]
    for dataset in DATASET_NAMES:
        group=raw[raw.dataset==dataset].groupby("model").mean(numeric_only=True)
        ranking.append({"dataset":dataset,"lowest_fid":group.custom_fid.idxmin(),"lowest_kid":group.custom_kid_mean.idxmin(),
                        "highest_confidence":group.mean_classifier_confidence.idxmax(),"highest_entropy":group.class_distribution_entropy.idxmax(),
                        "lowest_jsd":group.class_distribution_jsd.idxmin(),"fastest_sampling":group.sampling_ms_per_image.idxmin(),
                        "fastest_training":group.training_minutes.idxmin()})
    (TABLES/"metric_leaders.json").write_text(json.dumps(ranking,indent=2),encoding="utf-8")


def classifier_diagnostics():
    fig,axes=plt.subplots(1,2,figsize=(15,5.8))
    for axis,dataset in zip(axes,DATASET_NAMES):
        path=RESULTS/"evaluators"/f"{dataset}_confusion_matrix_normalized.csv"
        frame=pd.read_csv(path,index_col=0)
        image=axis.imshow(frame.to_numpy(),vmin=0,vmax=1,cmap="Blues",aspect="auto")
        axis.set_xticks(range(len(frame.columns)),frame.columns,rotation=60,ha="right",fontsize=7)
        axis.set_yticks(range(len(frame.index)),frame.index,fontsize=7)
        axis.set_xlabel("Predicted class"); axis.set_ylabel("True class"); axis.set_title(DATASET_NAMES[dataset])
    fig.colorbar(image,ax=axes.ravel().tolist(),label="Row-normalized proportion",shrink=.8)
    fig.suptitle("Frozen evaluator confusion matrices",weight="bold")
    fig.savefig(OUT/"classifier_confusion_matrices.png",dpi=240,bbox_inches="tight"); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,4.8))
    for axis,dataset in zip(axes,DATASET_NAMES):
        frame=pd.read_csv(RESULTS/"evaluators"/f"{dataset}_per_class_recall.csv")
        axis.bar(frame["class"],100*frame.recall,color="#4472C4")
        axis.set_ylim(0,100); axis.set_ylabel("Recall (%)"); axis.set_title(DATASET_NAMES[dataset])
        axis.tick_params(axis="x",rotation=45,labelsize=8); axis.grid(axis="y",alpha=.2)
    fig.suptitle("Evaluator per-class recall",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"classifier_per_class_recall.png",dpi=240,bbox_inches="tight"); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,4.8))
    for axis,dataset in zip(axes,DATASET_NAMES):
        confusion=pd.read_csv(RESULTS/"evaluators"/f"{dataset}_confusion_matrix.csv",index_col=0)
        proportions=confusion.sum(axis=1)/confusion.to_numpy().sum()
        axis.bar(proportions.index,100*proportions.values,color="#70AD47")
        axis.set_ylabel("Share of test split (%)"); axis.set_title(DATASET_NAMES[dataset])
        axis.tick_params(axis="x",rotation=45,labelsize=8); axis.grid(axis="y",alpha=.2)
    fig.suptitle("Ground-truth class balance in the evaluation sets",weight="bold")
    fig.tight_layout(); fig.savefig(OUT/"dataset_class_balance.png",dpi=240,bbox_inches="tight"); plt.close(fig)


def main():
    raw=pd.read_csv(RESULTS/"aggregate"/"all_runs.csv")
    means=raw.groupby(["dataset","model"]).mean(numeric_only=True)
    means.to_csv(OUT/"metrics_means.csv")
    raw.to_csv(TABLES/"all_seed_metrics.csv",index=False)
    full_summary=raw.groupby(["dataset","model"]).agg(["mean","std"])
    full_summary.columns=[f"{metric}_{stat}" for metric,stat in full_summary.columns]
    full_summary.to_csv(TABLES/"all_metrics_mean_std.csv")
    shutil.copy2(RESULTS/"aggregate"/"RESULTS_SUMMARY.md",TABLES/"RESULTS_SUMMARY.md")
    latex_tables(raw); architecture_figure(); classifier_figure(); classifier_diagnostics()
    qualitative_figure("fashion_mnist"); qualitative_figure("fashion_product_images")
    grouped_metric(raw,["custom_fid","custom_kid_mean"],"distribution_quality.png",{"custom_fid":"Feature FID ↓","custom_kid_mean":"Feature KID ↓"})
    grouped_metric(raw,["class_distribution_entropy","class_distribution_jsd"],"diversity.png",{"class_distribution_entropy":"Normalized class entropy ↑","class_distribution_jsd":"Class-distribution JSD ↓"})
    grouped_metric(raw,["training_minutes","sampling_ms_per_image","peak_gpu_memory_mb"],"compute_cost.png",{"training_minutes":"Training time (min) ↓","sampling_ms_per_image":"Batched sampling time (ms/image) ↓","peak_gpu_memory_mb":"Peak allocated GPU memory (MB) ↓"},log_metrics=("sampling_ms_per_image",))
    grouped_metric(raw,["training_peak_gpu_memory_mb","sampling_peak_gpu_memory_mb"],"memory_breakdown.png",{"training_peak_gpu_memory_mb":"Training peak allocated memory (MB) ↓","sampling_peak_gpu_memory_mb":"Sampling peak allocated memory (MB) ↓"})
    grouped_metric(raw,["trainable_parameters"],"model_size.png",{"trainable_parameters":"Trainable parameters ↓"})
    grouped_metric(raw,["mean_classifier_confidence","recognizability_rate_at_0.8","sobel_sharpness"],"supporting_quality.png",{
        "mean_classifier_confidence":"Mean evaluator confidence \u2191",
        "recognizability_rate_at_0.8":"Recognizability at confidence \u2265 0.8 \u2191",
        "sobel_sharpness":"Mean Sobel gradient magnitude \u2191",
    })
    quality_compute(raw); seed_variability(raw); class_distributions(); training_curves()
    for source,target in [
        (Path("hyperparameter_search/classifier/classifier_search.png"),"classifier_hyperparameter_search.png"),
        (Path("hyperparameter_search/generators/generator_search.png"),"generator_hyperparameter_search.png"),
        (Path("hyperparameter_search/ldm_diagnostic/ldm_diagnostic.png"),"ldm_capacity_sampler_diagnostic.png"),
        (Path("hyperparameter_search/ldm_clipping/ldm_clipping_search.png"),"ldm_clipping_search.png"),
        (Path("hyperparameter_search/ddpm_sampler/ddpm_sampler_search.png"),"ddpm_sampler_search.png"),
    ]:
        if source.exists(): shutil.copy2(source,OUT/target)
    print(f"Report assets written to {OUT.resolve()}")


if __name__=="__main__": main()
