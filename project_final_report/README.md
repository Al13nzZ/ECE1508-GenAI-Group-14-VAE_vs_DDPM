# Report package

`report.tex` is the new three-model, two-dataset report. Its document class,
NeurIPS package, package list, title/author structure, and numeric citation style
retain the original project template. The content has been rewritten around the
five-seed VAE/DDPM/LDM benchmark. A validated compiled copy is included as
`report.pdf`; `Report_Template.tex` preserves the supplied formatting template.

## Rebuild generated material

After `results_final_3models_2datasets_5seeds/` is complete and validated:

```powershell
.\.venv\Scripts\python.exe validate_results.py
.\.venv\Scripts\python.exe generate_report_assets.py
.\.venv\Scripts\python.exe validate_report.py
```

Run those commands from the repository root. The asset script creates
publication-resolution PNG figures in `figures/`, LaTeX result tables in
`tables/`, and CSV/JSON audit tables. The latter include every per-seed metric
and a complete mean/sample-SD table, including metrics not shown in the main
paper for space.

The official NeurIPS style is vendored as `neurips.sty` under the package name
used by the original template. Compile from this directory with a LaTeX
distribution; two passes are recommended so references settle:

```powershell
Set-Location project_final_report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Tectonic can also compile and resolve the cross-references in one command:

```powershell
tectonic -X compile report.tex --reruns 1
```

## Main report figures

- `model_architectures.png`: matched view of the three sampling paths
- `qualitative_fashion_mnist.png`: real/VAE/DDPM/LDM Fashion-MNIST grids
- `qualitative_fashion_product_images.png`: equivalent color product grids
- `distribution_quality.png`: five-seed feature FID and KID
- `diversity.png`: normalized class entropy and class-distribution JSD
- `class_distributions.png`: real and generated class proportions
- `compute_cost.png`: training, sampling, and memory cost
- `quality_compute_tradeoff.png`: FID versus sampling latency
- `training_curves.png`: mean and one-SD training curves
- `seed_variability.png`: every seed's FID rather than only the mean

## Method and evaluator figures

- `classifier_performance.png`
- `classifier_confusion_matrices.png`
- `classifier_per_class_recall.png`
- `dataset_class_balance.png`
- `classifier_hyperparameter_search.png`
- `generator_hyperparameter_search.png`
- `ldm_capacity_sampler_diagnostic.png`
- `ldm_clipping_search.png`
- `ddpm_sampler_search.png`
- `model_size.png`
- `memory_breakdown.png`
- `supporting_quality.png`

`TECHNICAL_REFERENCES.md` duplicates all technical citations in readable IEEE
style with direct URLs. The same references are included in `report.tex`, so the
report remains self-contained. `EXPERIMENT_ENVIRONMENT.md` records the local
hardware and software stack used for the definitive run.
`HYPERPARAMETER_RATIONALE.md` records the selection protocol, exact validation
scores, reasons for structural choices, limitations of the bounded search, and
the most useful follow-up ablations.
