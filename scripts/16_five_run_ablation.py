from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIVE_RUN_ROOT = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
)

RUNS_ROOT = FIVE_RUN_ROOT / "runs"

OUTPUT_DIR = FIVE_RUN_ROOT

SEEDS = [42, 43, 44, 45, 46]

COMPONENTS = [
    "contextual_residual",
    "residual_dynamics",
    "persistence_3h",
    "persistence_6h",
    "persistence_12h",
]

ANOMALY_TYPES = [
    "spike",
    "persistent_shift",
    "gradual_drift",
    "stuck",
    "contextual_after_hours",
]

VARIANT_ORDER = [
    "Full TwinGuard-X v3",
    "w/o Context",
    "w/o Multi-scale Persistence",
    "Residual Only",
    "Equal Fusion",
]

VARIANTS = {
    "Full TwinGuard-X v3": {
        "mode": "full"
    },

    "w/o Context": {
        "mode": "remove",
        "remove": [
            "contextual_residual"
        ],
    },

    "w/o Multi-scale Persistence": {
        "mode": "remove",
        "remove": [
            "persistence_3h",
            "persistence_6h",
            "persistence_12h",
        ],
    },

    "Residual Only": {
        "mode": "residual_only"
    },

    "Equal Fusion": {
        "mode": "equal"
    },
}


print("=" * 120)
print("EIRT 2026 — STEP 16")
print("FIVE-INDEPENDENT-RUN TWINGUARD-X v3 VALIDATION ABLATION")
print("=" * 120)

print(f"\nSeeds: {SEEDS}")
print(f"Number of runs: {len(SEEDS)}")

print("""
IMPORTANT:
- Validation partition ONLY.
- Locked final test is NOT accessed.
- No TwinGuard-X model is retrained.
- Each seed uses its own saved validation explanations.
- Each seed uses its own frozen reliability weights.
- Each seed retains its own frozen building-specific threshold.
- Removed components are followed by renormalization of remaining weights.
- Mean ± sample SD is calculated across the five independent seeds.
""")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction, score):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),

        "auroc": roc_auc_score(
            y_true,
            score,
        ),

        "auprc": average_precision_score(
            y_true,
            score,
        ),

        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),

        "false_positive_rate": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        ),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ============================================================
# WEIGHT HELPERS
# ============================================================

def renormalize_weights(weights, active_components):

    new_weights = {
        component: (
            weights.get(component, 0.0)
            if component in active_components
            else 0.0
        )
        for component in COMPONENTS
    }

    total = sum(new_weights.values())

    if total <= 0:
        raise ValueError(
            "No non-zero weights remain after ablation."
        )

    return {
        component: value / total
        for component, value in new_weights.items()
    }


def calculate_score(df, weights):

    score = np.zeros(
        len(df),
        dtype=float,
    )

    for component in COMPONENTS:

        score += (
            weights[component]
            *
            df[component].astype(float).values
        )

    return score


# ============================================================
# STORAGE
# ============================================================

all_building_results = []
all_seed_results = []
all_type_results = []
all_weight_results = []


# ============================================================
# FIVE-SEED LOOP
# ============================================================

for run_number, seed in enumerate(SEEDS, start=1):

    print("\n" + "#" * 120)
    print(
        f"ABLATION RUN {run_number}/{len(SEEDS)} "
        f"— SEED {seed}"
    )
    print("#" * 120)

    run_dir = RUNS_ROOT / f"seed_{seed}"

    explanations_path = (
        run_dir
        / "twinguard_x_v3_validation_explanations.csv"
    )

    weights_path = (
        run_dir
        / "twinguard_x_v3_reliability_weights.csv"
    )

    performance_path = (
        run_dir
        / "twinguard_x_v3_validation_performance.csv"
    )

    for path in [
        explanations_path,
        weights_path,
        performance_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found for seed {seed}: "
                f"{path}"
            )

    explanations = pd.read_csv(
        explanations_path,
        parse_dates=["timestamp"],
    )

    weights_df = pd.read_csv(
        weights_path
    )

    full_performance = pd.read_csv(
        performance_path
    )

    print(
        f"\nValidation rows: "
        f"{len(explanations):,}"
    )

    print(
        f"Buildings: "
        f"{explanations['building_id'].nunique()}"
    )

    buildings = (
        explanations["building_id"]
        .drop_duplicates()
        .tolist()
    )

    seed_building_results = []
    seed_type_results = []

    # ========================================================
    # BUILDING LOOP
    # ========================================================

    for building in buildings:

        print("\n" + "=" * 100)
        print(f"SEED {seed} | BUILDING: {building}")
        print("=" * 100)

        df = (
            explanations[
                explanations["building_id"] == building
            ]
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        y = (
            df["anomaly_label"]
            .astype(int)
            .values
        )

        threshold_values = (
            df["threshold"]
            .dropna()
            .unique()
        )

        if len(threshold_values) != 1:

            raise RuntimeError(
                f"Seed {seed}, {building}: expected "
                f"exactly one frozen threshold, found "
                f"{threshold_values}"
            )

        frozen_threshold = float(
            threshold_values[0]
        )

        # ----------------------------------------------------
        # LOAD FROZEN RELIABILITY WEIGHTS
        # ----------------------------------------------------

        building_weights = (
            weights_df[
                weights_df["building_id"] == building
            ]
            .copy()
        )

        if len(building_weights) != len(COMPONENTS):

            raise RuntimeError(
                f"Seed {seed}, {building}: expected "
                f"{len(COMPONENTS)} component weights, "
                f"found {len(building_weights)}."
            )

        counts = (
            building_weights["component"]
            .value_counts()
        )

        for component in COMPONENTS:

            if component not in counts.index:

                raise RuntimeError(
                    f"Seed {seed}, {building}: "
                    f"missing weight for {component}."
                )

            if counts[component] != 1:

                raise RuntimeError(
                    f"Seed {seed}, {building}: "
                    f"expected one weight for "
                    f"{component}, found "
                    f"{counts[component]}."
                )

        original_weights = {
            row["component"]: float(row["weight"])
            for _, row in building_weights.iterrows()
        }

        # Numerical cleanup
        for component in COMPONENTS:

            if abs(original_weights[component]) < 1e-12:
                original_weights[component] = 0.0

        weight_sum = sum(
            original_weights.values()
        )

        if not np.isclose(
            weight_sum,
            1.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                f"Seed {seed}, {building}: frozen "
                f"weights sum to {weight_sum:.10f}."
            )

        print(
            f"Validation observations: {len(df):,} | "
            f"Anomalies: {int(y.sum()):,} | "
            f"Threshold: {frozen_threshold:.6f}"
        )

        # ====================================================
        # VARIANT LOOP
        # ====================================================

        for variant_name, config in VARIANTS.items():

            mode = config["mode"]

            # -----------------------------------------------
            # FULL MODEL
            # -----------------------------------------------

            if mode == "full":

                variant_weights = (
                    original_weights.copy()
                )

                score = (
                    df["anomaly_score"]
                    .astype(float)
                    .values
                )

                prediction = (
                    df["prediction"]
                    .astype(int)
                    .values
                )

            # -----------------------------------------------
            # REMOVE COMPONENT(S)
            # -----------------------------------------------

            elif mode == "remove":

                removed = set(
                    config["remove"]
                )

                active = [
                    component
                    for component in COMPONENTS
                    if component not in removed
                ]

                active_weight_sum = sum(
                    original_weights[component]
                    for component in active
                )

                if active_weight_sum <= 0:

                    print(
                        f"WARNING: Seed {seed}, "
                        f"{building}, {variant_name}: "
                        f"all remaining frozen weights "
                        f"are zero. Variant skipped."
                    )

                    continue

                variant_weights = (
                    renormalize_weights(
                        original_weights,
                        active,
                    )
                )

                score = calculate_score(
                    df,
                    variant_weights,
                )

                prediction = (
                    score > frozen_threshold
                ).astype(int)

            # -----------------------------------------------
            # RESIDUAL ONLY
            # -----------------------------------------------

            elif mode == "residual_only":

                variant_weights = {
                    component: 0.0
                    for component in COMPONENTS
                }

                variant_weights[
                    "contextual_residual"
                ] = 1.0

                score = (
                    df["contextual_residual"]
                    .astype(float)
                    .values
                )

                prediction = (
                    score > frozen_threshold
                ).astype(int)

            # -----------------------------------------------
            # EQUAL FUSION
            # -----------------------------------------------

            elif mode == "equal":

                variant_weights = {
                    component: 1.0 / len(COMPONENTS)
                    for component in COMPONENTS
                }

                score = calculate_score(
                    df,
                    variant_weights,
                )

                prediction = (
                    score > frozen_threshold
                ).astype(int)

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            # -----------------------------------------------
            # METRICS
            # -----------------------------------------------

            m = calculate_metrics(
                y,
                prediction,
                score,
            )

            row = {
                "seed": seed,
                "building_id": building,
                "variant": variant_name,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auroc": m["auroc"],
                "auprc": m["auprc"],
                "specificity": m["specificity"],
                "false_positive_rate":
                    m["false_positive_rate"],
                "tp": m["tp"],
                "fp": m["fp"],
                "tn": m["tn"],
                "fn": m["fn"],
                "threshold": frozen_threshold,
            }

            seed_building_results.append(row)
            all_building_results.append(row)

            print(
                f"{variant_name:30s} | "
                f"F1={m['f1']:.4f} | "
                f"AUROC={m['auroc']:.4f} | "
                f"AUPRC={m['auprc']:.4f} | "
                f"FPR={m['false_positive_rate']:.4f}"
            )

            # -----------------------------------------------
            # SAVE VARIANT WEIGHTS
            # -----------------------------------------------

            weight_row = {
                "seed": seed,
                "building_id": building,
                "variant": variant_name,
            }

            weight_row.update(
                variant_weights
            )

            all_weight_results.append(
                weight_row
            )

            # -----------------------------------------------
            # ANOMALY-TYPE RECALL
            # -----------------------------------------------

            for anomaly_type in ANOMALY_TYPES:

                mask = (
                    df["anomaly_type"]
                    == anomaly_type
                )

                n = int(
                    mask.sum()
                )

                if n > 0:

                    recall_type = float(
                        prediction[mask].mean()
                    )

                else:

                    recall_type = np.nan

                type_row = {
                    "seed": seed,
                    "building_id": building,
                    "variant": variant_name,
                    "anomaly_type": anomaly_type,
                    "n": n,
                    "recall": recall_type,
                }

                seed_type_results.append(
                    type_row
                )

                all_type_results.append(
                    type_row
                )

    # ========================================================
    # VERIFY FULL-MODEL CONSISTENCY FOR THIS SEED
    # ========================================================

    seed_building_df = pd.DataFrame(
        seed_building_results
    )

    recomputed_full = (
        seed_building_df[
            seed_building_df["variant"]
            == "Full TwinGuard-X v3"
        ]
        .sort_values("building_id")
        .reset_index(drop=True)
    )

    saved_full = (
        full_performance
        .sort_values("building_id")
        .reset_index(drop=True)
    )

    print("\nFull-model consistency check:")

    for metric in [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "false_positive_rate",
    ]:

        difference = np.max(
            np.abs(
                recomputed_full[metric].values
                -
                saved_full[metric].values
            )
        )

        print(
            f"  {metric:24s}: "
            f"{difference:.10f}"
        )

        # Saved performance tables may contain rounded metrics.
        # Differences below 1e-3 are numerically negligible and
        # do not indicate a mismatch in predictions or scores.
        CONSISTENCY_TOLERANCE = 1e-3

        if difference > CONSISTENCY_TOLERANCE:

            raise RuntimeError(
                f"Seed {seed}: full-model "
                f"consistency check failed for "
                f"{metric}. Difference={difference:.10f}"
            )

    # ========================================================
    # PER-SEED MEAN ACROSS BUILDINGS
    # ========================================================

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "specificity",
        "false_positive_rate",
    ]

    seed_mean = (
        seed_building_df
        .groupby("variant")[
            metric_columns
        ]
        .mean()
        .reset_index()
    )

    seed_mean.insert(
        0,
        "seed",
        seed,
    )

    all_seed_results.append(
        seed_mean
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

building_df = pd.DataFrame(
    all_building_results
)

seed_mean_df = pd.concat(
    all_seed_results,
    ignore_index=True,
)

type_df = pd.DataFrame(
    all_type_results
)

weights_out = pd.DataFrame(
    all_weight_results
)


# ============================================================
# ORDER VARIANTS
# ============================================================

for frame in [
    building_df,
    seed_mean_df,
    type_df,
    weights_out,
]:

    frame["variant"] = pd.Categorical(
        frame["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )


# ============================================================
# FIVE-RUN MEAN ± SAMPLE SD
# ============================================================

metric_columns = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate",
]

summary_rows = []

for variant in VARIANT_ORDER:

    sub = seed_mean_df[
        seed_mean_df["variant"] == variant
    ]

    if len(sub) != len(SEEDS):

        raise RuntimeError(
            f"{variant}: expected {len(SEEDS)} "
            f"seed-level results, found {len(sub)}."
        )

    row = {
        "variant": variant,
        "n_runs": len(sub),
    }

    for metric in metric_columns:

        values = (
            sub[metric]
            .astype(float)
            .values
        )

        mean_value = np.mean(
            values
        )

        sd_value = np.std(
            values,
            ddof=1,
        )

        row[f"{metric}_mean"] = mean_value
        row[f"{metric}_sd"] = sd_value

        row[f"{metric}_mean_pm_sd"] = (
            f"{mean_value:.4f} ± "
            f"{sd_value:.4f}"
        )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# FIVE-RUN DELTA FROM FULL MODEL
# Calculate delta within each seed first, then aggregate.
# ============================================================

delta_rows = []

delta_metrics = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "false_positive_rate",
]

for seed in SEEDS:

    seed_sub = (
        seed_mean_df[
            seed_mean_df["seed"] == seed
        ]
        .copy()
    )

    full_rows = seed_sub[
        seed_sub["variant"]
        == "Full TwinGuard-X v3"
    ]

    if len(full_rows) != 1:

        raise RuntimeError(
            f"Seed {seed}: expected exactly "
            f"one full-model seed mean."
        )

    full_row = full_rows.iloc[0]

    for variant in VARIANT_ORDER[1:]:

        variant_rows = seed_sub[
            seed_sub["variant"] == variant
        ]

        if len(variant_rows) != 1:

            raise RuntimeError(
                f"Seed {seed}, {variant}: "
                f"expected exactly one row."
            )

        variant_row = (
            variant_rows.iloc[0]
        )

        delta_row = {
            "seed": seed,
            "variant": variant,
        }

        for metric in delta_metrics:

            delta_row[
                f"delta_{metric}"
            ] = (
                variant_row[metric]
                -
                full_row[metric]
            )

        delta_rows.append(
            delta_row
        )


delta_df = pd.DataFrame(
    delta_rows
)

delta_summary_rows = []

for variant in VARIANT_ORDER[1:]:

    sub = delta_df[
        delta_df["variant"] == variant
    ]

    row = {
        "variant": variant,
        "n_runs": len(sub),
    }

    for metric in delta_metrics:

        column = f"delta_{metric}"

        values = (
            sub[column]
            .astype(float)
            .values
        )

        mean_value = np.mean(
            values
        )

        sd_value = np.std(
            values,
            ddof=1,
        )

        row[f"{column}_mean"] = (
            mean_value
        )

        row[f"{column}_sd"] = (
            sd_value
        )

        row[f"{column}_mean_pm_sd"] = (
            f"{mean_value:.4f} ± "
            f"{sd_value:.4f}"
        )

    delta_summary_rows.append(
        row
    )


delta_summary_df = pd.DataFrame(
    delta_summary_rows
)


# ============================================================
# ANOMALY-TYPE RECALL:
# Mean across buildings within each seed,
# then mean ± SD across five seeds.
# ============================================================

seed_type_mean = (
    type_df
    .groupby(
        [
            "seed",
            "variant",
            "anomaly_type",
        ],
        observed=False,
    )["recall"]
    .mean()
    .reset_index()
)

type_summary_rows = []

for variant in VARIANT_ORDER:

    for anomaly_type in ANOMALY_TYPES:

        sub = seed_type_mean[
            (
                seed_type_mean["variant"]
                == variant
            )
            &
            (
                seed_type_mean["anomaly_type"]
                == anomaly_type
            )
        ]

        values = (
            sub["recall"]
            .dropna()
            .astype(float)
            .values
        )

        if len(values) == 0:
            continue

        mean_value = np.mean(
            values
        )

        sd_value = (
            np.std(values, ddof=1)
            if len(values) > 1
            else np.nan
        )

        type_summary_rows.append({
            "variant": variant,
            "anomaly_type": anomaly_type,
            "n_runs": len(values),
            "recall_mean": mean_value,
            "recall_sd": sd_value,
            "recall_mean_pm_sd": (
                f"{mean_value:.4f} ± "
                f"{sd_value:.4f}"
            ),
        })


type_summary_df = pd.DataFrame(
    type_summary_rows
)


# ============================================================
# SAVE
# ============================================================

building_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_per_seed_by_building.csv",
    index=False,
)

seed_mean_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_per_seed_mean.csv",
    index=False,
)

summary_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_mean_sd.csv",
    index=False,
)

delta_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_delta_per_seed.csv",
    index=False,
)

delta_summary_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_delta_from_full_mean_sd.csv",
    index=False,
)

type_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_type_recall_by_building.csv",
    index=False,
)

seed_type_mean.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_type_recall_per_seed.csv",
    index=False,
)

type_summary_df.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_type_recall_mean_sd.csv",
    index=False,
)

weights_out.to_csv(
    OUTPUT_DIR
    / "five_run_ablation_weights.csv",
    index=False,
)


# ============================================================
# PRINT MANUSCRIPT-READY SUMMARY
# ============================================================

print("\n" + "=" * 120)
print("FIVE-RUN VALIDATION ABLATION — MEAN ± SAMPLE SD")
print("=" * 120)

display_columns = [
    "variant",
    "precision_mean_pm_sd",
    "recall_mean_pm_sd",
    "f1_mean_pm_sd",
    "auroc_mean_pm_sd",
    "auprc_mean_pm_sd",
    "false_positive_rate_mean_pm_sd",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("FIVE-RUN CHANGE RELATIVE TO FULL TWINGUARD-X")
print("=" * 120)

delta_display_columns = [
    "variant",
    "delta_precision_mean_pm_sd",
    "delta_recall_mean_pm_sd",
    "delta_f1_mean_pm_sd",
    "delta_auroc_mean_pm_sd",
    "delta_auprc_mean_pm_sd",
    "delta_false_positive_rate_mean_pm_sd",
]

print(
    delta_summary_df[
        delta_display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("FIVE-RUN ANOMALY-TYPE RECALL — MEAN ± SAMPLE SD")
print("=" * 120)

type_pivot = (
    type_summary_df
    .pivot(
        index="anomaly_type",
        columns="variant",
        values="recall_mean_pm_sd",
    )
)

existing_order = [
    variant
    for variant in VARIANT_ORDER
    if variant in type_pivot.columns
]

type_pivot = type_pivot[
    existing_order
]

print(
    type_pivot.to_string()
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "experiment":
        "Five-run TwinGuard-X v3 validation ablation",

    "seeds":
        SEEDS,

    "number_of_runs":
        len(SEEDS),

    "data_partition":
        "validation only",

    "final_test_accessed":
        False,

    "model_retrained":
        False,

    "threshold_recalibrated":
        False,

    "aggregation":
        (
            "Metrics are first averaged across buildings "
            "within each independent seed. Mean and sample "
            "standard deviation are then calculated across "
            "the five seed-level results."
        ),

    "standard_deviation":
        "sample SD with ddof=1",

    "ablation_rule":
        (
            "Remove specified frozen evidence components "
            "and renormalize remaining frozen reliability "
            "weights."
        ),

    "threshold_rule":
        (
            "Retain each seed's original frozen "
            "building-specific TwinGuard-X v3 threshold."
        ),

    "variants":
        VARIANT_ORDER,

    "components":
        COMPONENTS,

    "anomaly_types":
        ANOMALY_TYPES,
}

with open(
    OUTPUT_DIR
    / "five_run_ablation_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


print("\n" + "=" * 120)
print("STEP 16 COMPLETE")
print("=" * 120)

print("\nSaved:")

for filename in [
    "five_run_ablation_per_seed_by_building.csv",
    "five_run_ablation_per_seed_mean.csv",
    "five_run_ablation_mean_sd.csv",
    "five_run_ablation_delta_per_seed.csv",
    "five_run_ablation_delta_from_full_mean_sd.csv",
    "five_run_ablation_type_recall_by_building.csv",
    "five_run_ablation_type_recall_per_seed.csv",
    "five_run_ablation_type_recall_mean_sd.csv",
    "five_run_ablation_weights.csv",
    "five_run_ablation_manifest.json",
]:

    print(
        OUTPUT_DIR
        / filename
    )