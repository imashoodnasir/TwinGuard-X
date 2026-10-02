from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ============================================================
# EIRT 2026 — STEP 15
# FIVE INDEPENDENT RUNS FOR MEAN ± SD
# ============================================================
#
# IMPORTANT DESIGN:
# - Data, splits, anomaly timestamps/types/magnitudes remain FIXED.
# - Seeds 42–46 vary model initialization/training stochasticity.
# - Existing Step 11, 12, and 13 implementations are reused exactly.
# - Step 11 performs train/validation development.
# - Step 12 performs evaluation on the same locked final-test set.
# - Step 13 performs the fair baseline comparison on exact v3 timestamps.
# - Each run is archived before the next run starts.
#
# This script creates temporary seed-specific copies of Step 11 and Step 13.
# It DOES NOT permanently edit your original scripts.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "models"

STEP11 = SCRIPTS / "11_twinguard_x_v3_development.py"
STEP12 = SCRIPTS / "12_final_locked_evaluation.py"
STEP13 = SCRIPTS / "13_fair_baseline_comparison.py"

SEEDS = [42, 43, 44, 45, 46]

OUT = TABLES / "five_run_experiment"
RUNS_DIR = OUT / "runs"
TEMP_DIR = SCRIPTS

OUT.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = [STEP11, STEP12, STEP13]
for p in REQUIRED:
    if not p.exists():
        raise FileNotFoundError(f"Required script not found: {p}")

# Files produced by Step 11 that are useful to archive.
STEP11_OUTPUTS = [
    "twinguard_x_v3_validation_performance.csv",
    "twinguard_x_v3_reliability_weights.csv",
    "twinguard_x_v3_validation_type_recall.csv",
    "twinguard_x_v3_validation_explanations.csv",
    "twinguard_x_v3_freeze_manifest.json",
]

# Files produced by Step 12.
STEP12_OUTPUTS = [
    "twinguard_x_v3_FINAL_locked_test_performance.csv",
    "twinguard_x_v3_FINAL_type_recall.csv",
    "twinguard_x_v3_FINAL_explanations.csv",
]

# Files produced by Step 13.
STEP13_OUTPUTS = [
    "fair_final_model_comparison_by_building.csv",
    "fair_final_model_comparison_mean.csv",
    "fair_baseline_validation_calibration.csv",
    "fair_baseline_final_predictions.csv",
    "fair_model_anomaly_type_recall.csv",
    "fair_model_mean_anomaly_type_recall.csv",
    "twinguard_v3_relative_to_baselines.csv",
]

METRICS = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate",
]

def make_seeded_copy(source: Path, destination: Path, seed: int):
    """Replace only the first top-level hard-coded SEED assignment."""
    text = source.read_text(encoding="utf-8")

    patched, n = re.subn(
        r"(?m)^SEED\s*=\s*\d+\s*$",
        f"SEED = {seed}",
        text,
        count=1,
    )

    if n != 1:
        raise RuntimeError(
            f"Could not uniquely patch SEED in {source.name}. "
            f"Expected exactly one top-level SEED assignment, found {n}."
        )

    destination.write_text(patched, encoding="utf-8")


def run_script(path: Path):
    print("\n" + "=" * 118)
    print(f"RUNNING: {path.name}")
    print("=" * 118)

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{path.name} failed with exit code {result.returncode}"
        )


def archive_table(filename: str, run_dir: Path):
    src = TABLES / filename
    if not src.exists():
        raise FileNotFoundError(
            f"Expected output was not created: {src}"
        )
    shutil.copy2(src, run_dir / filename)


def add_seed_column(path: Path, seed: int):
    """Add seed to archived CSV for explicit provenance."""
    if path.suffix.lower() != ".csv":
        return

    df = pd.read_csv(path)
    if "seed" in df.columns:
        df["seed"] = seed
    else:
        df.insert(0, "seed", seed)
    df.to_csv(path, index=False)


print("=" * 118)
print("EIRT 2026 — STEP 15")
print("FIVE INDEPENDENT RUNS: TWINGUARD-X v3 + FAIR BASELINES")
print("=" * 118)

print("\nSeeds:", SEEDS)
print("Number of runs:", len(SEEDS))
print("\nFixed across runs:")
print("  - building selection")
print("  - locked train/validation/final-test weeks")
print("  - injected anomaly timestamps, types, and magnitudes")
print("  - architecture and hyperparameters")
print("  - calibration procedure")
print("\nVaried across runs:")
print("  - random seed")
print("  - neural-network initialization")
print("  - shuffled mini-batch training")
print("  - stochastic baseline initialization where applicable")

all_v3_final = []
all_model_final = []
all_validation = []
all_weights = []
all_type_recall = []
run_manifest = []

experiment_start = time.perf_counter()

for run_number, seed in enumerate(SEEDS, start=1):

    print("\n\n" + "#" * 118)
    print(f"INDEPENDENT RUN {run_number}/{len(SEEDS)} — SEED {seed}")
    print("#" * 118)

    run_dir = RUNS_DIR / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seeded11 = TEMP_DIR / f"11_twinguard_x_v3_development_seed_{seed}.py"
    seeded13 = TEMP_DIR / f"13_fair_baseline_comparison_seed_{seed}.py"

    make_seeded_copy(STEP11, seeded11, seed)
    make_seeded_copy(STEP13, seeded13, seed)

    start = time.perf_counter()

    # --------------------------------------------------------
    # 1) Retrain + recalibrate v3 using TRAIN + VALIDATION only
    # --------------------------------------------------------
    run_script(seeded11)

    for filename in STEP11_OUTPUTS:
        archive_table(filename, run_dir)
        add_seed_column(run_dir / filename, seed)

    # Archive seed-specific frozen model files before next seed.
    seed_model_dir = run_dir / "models"
    seed_model_dir.mkdir(parents=True, exist_ok=True)

    for model_file in MODELS.glob("*TwinGuardX_v3*"):
        if model_file.is_file():
            shutil.copy2(model_file, seed_model_dir / model_file.name)

    # --------------------------------------------------------
    # 2) Evaluate this independently trained v3 on SAME locked test
    # --------------------------------------------------------
    run_script(STEP12)

    for filename in STEP12_OUTPUTS:
        archive_table(filename, run_dir)
        add_seed_column(run_dir / filename, seed)

    # --------------------------------------------------------
    # 3) Fair baseline comparison for this seed
    # --------------------------------------------------------
    run_script(seeded13)

    for filename in STEP13_OUTPUTS:
        archive_table(filename, run_dir)
        add_seed_column(run_dir / filename, seed)

    # --------------------------------------------------------
    # Collect run-level results
    # --------------------------------------------------------
    v3_final = pd.read_csv(
        run_dir / "twinguard_x_v3_FINAL_locked_test_performance.csv"
    )
    v3_final.insert(
        0, "seed", seed
    ) if "seed" not in v3_final.columns else None
    all_v3_final.append(v3_final)

    model_final = pd.read_csv(
        run_dir / "fair_final_model_comparison_by_building.csv"
    )
    if "seed" not in model_final.columns:
        model_final.insert(0, "seed", seed)
    all_model_final.append(model_final)

    validation = pd.read_csv(
        run_dir / "twinguard_x_v3_validation_performance.csv"
    )
    if "seed" not in validation.columns:
        validation.insert(0, "seed", seed)
    all_validation.append(validation)

    weights = pd.read_csv(
        run_dir / "twinguard_x_v3_reliability_weights.csv"
    )
    if "seed" not in weights.columns:
        weights.insert(0, "seed", seed)
    all_weights.append(weights)

    type_recall = pd.read_csv(
        run_dir / "fair_model_anomaly_type_recall.csv"
    )
    if "seed" not in type_recall.columns:
        type_recall.insert(0, "seed", seed)
    all_type_recall.append(type_recall)

    elapsed = time.perf_counter() - start

    run_manifest.append({
        "run": run_number,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "status": "complete",
    })

    print(
        f"\nRUN {run_number}/{len(SEEDS)} COMPLETE "
        f"(seed={seed}, {elapsed/60:.2f} min)"
    )


# ============================================================
# COMBINE ALL FIVE RUNS
# ============================================================

v3_all = pd.concat(all_v3_final, ignore_index=True)
models_all = pd.concat(all_model_final, ignore_index=True)
validation_all = pd.concat(all_validation, ignore_index=True)
weights_all = pd.concat(all_weights, ignore_index=True)
type_all = pd.concat(all_type_recall, ignore_index=True)

v3_all.to_csv(
    OUT / "five_run_v3_final_by_building.csv",
    index=False,
)

models_all.to_csv(
    OUT / "five_run_all_models_final_by_building.csv",
    index=False,
)

validation_all.to_csv(
    OUT / "five_run_v3_validation_by_building.csv",
    index=False,
)

weights_all.to_csv(
    OUT / "five_run_v3_reliability_weights.csv",
    index=False,
)

type_all.to_csv(
    OUT / "five_run_anomaly_type_recall.csv",
    index=False,
)


# ============================================================
# PER-SEED MEAN ACROSS BUILDINGS
# This is the unit used for final mean ± SD across independent runs.
# ============================================================

per_seed_model_mean = (
    models_all
    .groupby(["seed", "model"], as_index=False)[METRICS]
    .mean()
)

per_seed_model_mean.to_csv(
    OUT / "five_run_per_seed_model_mean.csv",
    index=False,
)


# ============================================================
# FINAL MEAN ± SAMPLE SD ACROSS FIVE INDEPENDENT RUNS
# ddof=1 = sample standard deviation
# ============================================================

summary_rows = []

for model_name, group in per_seed_model_mean.groupby("model"):

    row = {
        "model": model_name,
        "n_runs": group["seed"].nunique(),
    }

    for metric in METRICS:
        values = group[metric].astype(float).values

        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))

        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_mean_pm_sd"] = (
            f"{mean:.4f} ± {sd:.4f}"
        )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

preferred_order = [
    "Autoencoder",
    "Isolation Forest",
    "LOF",
    "TwinGuard-X v3",
]

summary["_order"] = summary["model"].map(
    {name: i for i, name in enumerate(preferred_order)}
).fillna(999)

summary = (
    summary
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

summary.to_csv(
    OUT / "five_run_FINAL_mean_sd.csv",
    index=False,
)


# ============================================================
# V3 PER-BUILDING MEAN ± SD
# ============================================================

v3_building_rows = []

for building, group in v3_all.groupby("building_id"):

    row = {
        "building_id": building,
        "n_runs": group["seed"].nunique(),
    }

    for metric in METRICS:
        values = group[metric].astype(float).values
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))

        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_mean_pm_sd"] = (
            f"{mean:.4f} ± {sd:.4f}"
        )

    v3_building_rows.append(row)

v3_building_summary = pd.DataFrame(v3_building_rows)

v3_building_summary.to_csv(
    OUT / "five_run_v3_per_building_mean_sd.csv",
    index=False,
)


# ============================================================
# RELIABILITY-WEIGHT STABILITY
# ============================================================

weight_summary = (
    weights_all
    .groupby(
        ["building_id", "component"]
    )["weight"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(
        columns={
            "mean": "weight_mean",
            "std": "weight_sd",
        }
    )
)

weight_summary["weight_mean_pm_sd"] = (
    weight_summary.apply(
        lambda r:
        f"{r['weight_mean']:.4f} ± {r['weight_sd']:.4f}",
        axis=1,
    )
)

weight_summary.to_csv(
    OUT / "five_run_v3_weight_stability.csv",
    index=False,
)


# ============================================================
# ANOMALY-TYPE RECALL MEAN ± SD
# ============================================================

# First average across buildings within each seed/model/type.
type_per_seed = (
    type_all
    .groupby(
        ["seed", "model", "anomaly_type"],
        as_index=False,
    )["recall"]
    .mean()
)

type_summary = (
    type_per_seed
    .groupby(
        ["model", "anomaly_type"]
    )["recall"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(
        columns={
            "mean": "recall_mean",
            "std": "recall_sd",
        }
    )
)

type_summary["recall_mean_pm_sd"] = (
    type_summary.apply(
        lambda r:
        f"{r['recall_mean']:.4f} ± {r['recall_sd']:.4f}",
        axis=1,
    )
)

type_summary.to_csv(
    OUT / "five_run_anomaly_type_recall_mean_sd.csv",
    index=False,
)


# ============================================================
# EXPERIMENT MANIFEST
# ============================================================

total_elapsed = time.perf_counter() - experiment_start

manifest = {
    "experiment": "EIRT 2026 TwinGuard-X v3 five independent runs",
    "seeds": SEEDS,
    "n_runs": len(SEEDS),
    "standard_deviation": "sample SD (ddof=1)",
    "aggregation": (
        "Metrics are first averaged across the three buildings "
        "within each seed; mean and sample SD are then calculated "
        "across the five independent seeds."
    ),
    "fixed_across_runs": [
        "building selection",
        "train/validation/final-test partition",
        "anomaly timestamps",
        "anomaly types",
        "anomaly magnitudes",
        "architecture",
        "hyperparameters",
        "calibration procedure",
    ],
    "varied_across_runs": [
        "random seed",
        "neural-network initialization",
        "mini-batch ordering",
        "stochastic model training",
        "Isolation Forest random state",
        "Autoencoder initialization/training",
    ],
    "runs": run_manifest,
    "total_elapsed_seconds": total_elapsed,
}

with open(
    OUT / "five_run_experiment_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=4)


# ============================================================
# DISPLAY FINAL TABLE
# ============================================================

print("\n\n" + "=" * 118)
print("FIVE-RUN FINAL PERFORMANCE — MEAN ± SAMPLE SD")
print("=" * 118)

display_columns = ["model"]

for metric in [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "false_positive_rate",
]:
    display_columns.append(
        f"{metric}_mean_pm_sd"
    )

print(
    summary[display_columns]
    .to_string(index=False)
)

print("\n" + "=" * 118)
print("STEP 15 COMPLETE")
print("=" * 118)

print(f"\nTotal experiment time: {total_elapsed/60:.2f} min")

print("\nMain outputs:")
print(OUT / "five_run_FINAL_mean_sd.csv")
print(OUT / "five_run_per_seed_model_mean.csv")
print(OUT / "five_run_v3_per_building_mean_sd.csv")
print(OUT / "five_run_v3_weight_stability.csv")
print(OUT / "five_run_anomaly_type_recall_mean_sd.csv")
print(OUT / "five_run_experiment_manifest.json")

print(
    "\nUse five_run_FINAL_mean_sd.csv for the manuscript's "
    "mean ± SD comparison table."
)
