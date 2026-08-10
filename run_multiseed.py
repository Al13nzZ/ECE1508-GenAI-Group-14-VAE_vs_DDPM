from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "vae_ddpm_fashionmnist_multiseed_results"
SEEDS = list(range(10))


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    completed = []
    for index, seed in enumerate(SEEDS, start=1):
        output_dir = RESULTS_ROOT / f"seed_{seed:02d}"
        result_path = output_dir / "results.json"
        if result_path.exists():
            print(f"[{index}/10] Seed {seed}: existing completed result found; skipping.", flush=True)
            completed.append(seed)
            continue

        print(f"[{index}/10] Seed {seed}: starting full report run.", flush=True)
        env = os.environ.copy()
        env.update(
            {
                "EXPERIMENT_PROFILE": "report",
                "EXPERIMENT_SEED": str(seed),
                "EXPERIMENT_DATA_DIR": str(ROOT / "data"),
                "EXPERIMENT_OUTPUT_DIR": str(output_dir),
                "PACKAGE_RESULTS": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        subprocess.run(
            [sys.executable, str(ROOT / "complete_project.py")],
            cwd=ROOT,
            env=env,
            check=True,
        )
        completed.append(seed)
        print(f"[{index}/10] Seed {seed}: complete.", flush=True)

    (RESULTS_ROOT / "completed_seeds.json").write_text(
        json.dumps(completed, indent=2), encoding="utf-8"
    )
    print("All ten seed runs completed.", flush=True)


if __name__ == "__main__":
    main()
