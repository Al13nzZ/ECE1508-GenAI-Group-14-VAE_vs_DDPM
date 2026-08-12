"""Validate completeness and numerical integrity of the definitive benchmark."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parent
RESULTS=Path(os.environ.get("BENCHMARK_OUTPUT_DIR",ROOT/"results_final_3models_2datasets_5seeds"))
DATASETS=("fashion_mnist","fashion_product_images")
MODELS={"VAE","DDPM","LDM"}


def main():
    errors=[]
    config_path=RESULTS/"config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing configuration: {config_path}")
    config=json.loads(config_path.read_text()); seeds=config["seeds"]
    current_hash=hashlib.sha256((ROOT/"run_three_model_color_benchmark.py").read_bytes()).hexdigest()
    if config.get("source_sha256")!=current_hash:
        errors.append("config source_sha256 does not match the current benchmark runner")
    rows=[]; snapshots=[]
    for dataset in DATASETS:
        evaluator=RESULTS/"evaluators"/f"{dataset}_metrics.json"
        if not evaluator.exists(): errors.append(f"missing evaluator metrics: {evaluator}")
        for seed in seeds:
            run=RESULTS/dataset/f"seed_{seed:02d}"; result_path=run/"results.json"
            if not result_path.exists():
                errors.append(f"missing result: {result_path}"); continue
            result=json.loads(result_path.read_text()); snapshots.append(result.get("config"))
            names={entry.get("model") for entry in result.get("models",[])}
            if names!=MODELS: errors.append(f"{run}: model set is {sorted(names)}")
            for entry in result.get("models",[]):
                for key,value in entry.items():
                    if key=="class_distribution":
                        if not math.isclose(sum(value),1.0,rel_tol=1e-6,abs_tol=1e-6):
                            errors.append(f"{run}/{entry['model']}: class distribution does not sum to one")
                    elif isinstance(value,(int,float)) and not math.isfinite(value):
                        errors.append(f"{run}/{entry['model']}: non-finite {key}")
                if entry.get("nonfinite_loss_count")!=0:
                    errors.append(f"{run}/{entry['model']}: non-finite training losses")
                rows.append({"dataset":dataset,"seed":seed,**{k:v for k,v in entry.items() if k!="class_distribution"}})
            required=[run/"model_metrics.csv",run/"class_distribution.csv"]
            required += [run/"logs"/f"{name}_history.csv" for name in ("vae","ddpm","ldm")]
            required += [run/"figures"/f"{name}_samples.png" for name in ("real","vae","ddpm","ldm")]
            for path in required:
                if not path.exists() or path.stat().st_size==0: errors.append(f"missing or empty artifact: {path}")
    if any(snapshot!=config for snapshot in snapshots): errors.append("per-run configuration differs from config.json")
    frame=pd.DataFrame(rows)
    expected=len(DATASETS)*len(seeds)*len(MODELS)
    if len(frame)!=expected: errors.append(f"expected {expected} model rows, found {len(frame)}")
    if not frame.empty and frame[["dataset","seed","model"]].duplicated().any():
        errors.append("duplicate dataset/seed/model rows")
    aggregate=RESULTS/"aggregate"/"all_runs.csv"
    if not aggregate.exists(): errors.append(f"missing aggregate table: {aggregate}")
    elif len(pd.read_csv(aggregate))!=expected: errors.append("aggregate table has the wrong row count")
    if errors:
        print("VALIDATION FAILED")
        for error in errors: print(f"- {error}")
        raise SystemExit(1)
    print(f"VALIDATION PASSED: {expected} model runs across {len(DATASETS)} datasets and {len(seeds)} seeds")
    print(frame.groupby(["dataset","model"])[["custom_fid","custom_kid_mean","training_minutes","sampling_ms_per_image"]].mean().round(4))


if __name__=="__main__": main()
