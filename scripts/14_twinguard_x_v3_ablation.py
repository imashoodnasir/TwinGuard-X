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
TABLES = ROOT / "results" / "tables"

EXPLANATIONS_PATH = (
    TABLES /
    "twinguard_x_v3_validation_explanations.csv"
)

WEIGHTS_PATH = (
    TABLES /
    "twinguard_x_v3_reliability_weights.csv"
)

FULL_PERFORMANCE_PATH = (
    TABLES /
    "twinguard_x_v3_validation_performance.csv"
)

COMPONENTS = [
    "contextual_residual",
    "residual_dynamics",
    "persistence_3h",
    "persistence_6h",
    "persistence_12h",
]

print("=" * 110)
print("EIRT 2026 — STEP 14")
print("TWINGUARD-X v3 VALIDATION ABLATION STUDY")
print("=" * 110)

print("""
IMPORTANT:
- Validation partition ONLY.
- Locked final test is NOT accessed.
- TwinGuard-X v3 is NOT retrained.
- Full-model reliability weights remain frozen.
- Ablations remove evidence components and renormalize remaining weights.
- The original frozen building-specific threshold is retained.
""")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, prediction, score):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "auroc": roc_auc_score(
            y_true,
            score
        ),

        "auprc": average_precision_score(
            y_true,
            score
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
        component:
            value / total

        for component, value
        in new_weights.items()
    }


def calculate_score(df, weights):

    score = np.zeros(
        len(df),
        dtype=float
    )

    for component in COMPONENTS:

        score += (
            weights[component]
            *
            df[component]
            .astype(float)
            .values
        )

    return score


# ============================================================
# LOAD DATA
# ============================================================

explanations = pd.read_csv(
    EXPLANATIONS_PATH,
    parse_dates=["timestamp"]
)

weights_df = pd.read_csv(
    WEIGHTS_PATH
)

full_performance = pd.read_csv(
    FULL_PERFORMANCE_PATH
)

print(
    f"Validation explanation rows: "
    f"{len(explanations):,}"
)

print(
    f"Buildings: "
    f"{explanations['building_id'].nunique()}"
)

print("\nAvailable evidence channels:")

for component in COMPONENTS:
    print(f"  - {component}")


# ============================================================
# DEFINE ABLATIONS
# ============================================================

variants = {

    "Full TwinGuard-X v3": {
        "mode": "full"
    },

    "w/o Context": {
        "mode": "remove",
        "remove": [
            "contextual_residual"
        ]
    },

    "w/o Multi-scale Persistence": {
        "mode": "remove",
        "remove": [
            "persistence_3h",
            "persistence_6h",
            "persistence_12h",
        ]
    },

    "Residual Only": {
        "mode": "residual_only"
    },

    "Equal Fusion": {
        "mode": "equal"
    },
}


# ============================================================
# RESULTS
# ============================================================

results = []
type_results = []
weight_results = []
prediction_results = []

buildings = (
    explanations[
        "building_id"
    ]
    .drop_duplicates()
    .tolist()
)


# ============================================================
# BUILDING LOOP
# ============================================================

for building in buildings:

    print("\n" + "=" * 110)
    print(f"BUILDING: {building}")
    print("=" * 110)

    df = (
        explanations[
            explanations[
                "building_id"
            ]
            ==
            building
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
            f"{building}: expected exactly "
            f"one frozen threshold, found "
            f"{threshold_values}"
        )

    frozen_threshold = float(
        threshold_values[0]
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL FROZEN RELIABILITY WEIGHTS
    # --------------------------------------------------------

    # --------------------------------------------------------
# LOAD ORIGINAL FROZEN RELIABILITY WEIGHTS
# The saved Step-11 file uses long format:
# building_id | component | weight
# --------------------------------------------------------

    building_weights = (
        weights_df[
            weights_df["building_id"] == building
        ]
        .copy()
    )

    if len(building_weights) != len(COMPONENTS):
        raise RuntimeError(
            f"{building}: expected {len(COMPONENTS)} "
            f"component weights, found {len(building_weights)}."
        )

    # Check that every required component exists exactly once
    counts = (
        building_weights["component"]
        .value_counts()
    )

    for component in COMPONENTS:

        if component not in counts.index:
            raise RuntimeError(
                f"{building}: missing reliability weight "
                f"for '{component}'."
            )

        if counts[component] != 1:
            raise RuntimeError(
                f"{building}: expected one weight for "
                f"'{component}', found {counts[component]}."
            )

    original_weights = {
        row["component"]: float(row["weight"])
        for _, row in building_weights.iterrows()
    }

    # Numerical cleanup only.
    # Step 11 can produce tiny floating-point values such as
    # 2.220446e-16 instead of exact zero.
    for component in COMPONENTS:

        if abs(original_weights[component]) < 1e-12:
            original_weights[component] = 0.0

    # Safety check: frozen weights should sum to approximately 1.
    weight_sum = sum(original_weights.values())

    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise RuntimeError(
            f"{building}: frozen weights sum to "
            f"{weight_sum:.10f}, expected approximately 1.0."
        )

    print(
        f"\nValidation observations: "
        f"{len(df):,}"
    )

    print(
        f"Validation anomalies: "
        f"{y.sum():,}"
    )

    print(
        f"Frozen threshold: "
        f"{frozen_threshold:.6f}"
    )

    print("\nFrozen full-model weights:")

    for component in COMPONENTS:

        print(
            f"  {component:24s}: "
            f"{original_weights[component]:.4f}"
        )

    # ========================================================
    # VARIANT LOOP
    # ========================================================

    for variant_name, config in variants.items():

        print(
            "\n" + "-" * 90
        )

        print(
            f"VARIANT: {variant_name}"
        )

        # ----------------------------------------------------
        # FULL MODEL
        # ----------------------------------------------------

        if config["mode"] == "full":

            variant_weights = (
                original_weights.copy()
            )

            # Use exactly the saved score.
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

        # ----------------------------------------------------
        # REMOVE COMPONENT(S)
        # ----------------------------------------------------

        elif config["mode"] == "remove":

            removed = set(
                config["remove"]
            )

            active = [
                component
                for component in COMPONENTS
                if component not in removed
            ]

            active_weight_sum = sum(
                original_weights[
                    component
                ]
                for component in active
            )

            if active_weight_sum <= 0:

                print(
                    "  SKIPPED: all remaining "
                    "frozen weights are zero."
                )

                continue

            variant_weights = (
                renormalize_weights(
                    original_weights,
                    active
                )
            )

            score = calculate_score(
                df,
                variant_weights
            )

            prediction = (
                score
                >
                frozen_threshold
            ).astype(int)

        # ----------------------------------------------------
        # RESIDUAL ONLY
        # ----------------------------------------------------

        elif config["mode"] == "residual_only":

            variant_weights = {
                component: 0.0
                for component in COMPONENTS
            }

            variant_weights[
                "contextual_residual"
            ] = 1.0

            score = (
                df[
                    "contextual_residual"
                ]
                .astype(float)
                .values
            )

            prediction = (
                score
                >
                frozen_threshold
            ).astype(int)

        # ----------------------------------------------------
        # EQUAL FUSION
        # ----------------------------------------------------

        elif config["mode"] == "equal":

            variant_weights = {
                component:
                    1.0 / len(COMPONENTS)

                for component
                in COMPONENTS
            }

            score = calculate_score(
                df,
                variant_weights
            )

            prediction = (
                score
                >
                frozen_threshold
            ).astype(int)

        else:

            raise ValueError(
                f"Unknown mode: "
                f"{config['mode']}"
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        m = calculate_metrics(
            y,
            prediction,
            score
        )

        print(
            f"  Precision = "
            f"{m['precision']:.4f}"
        )

        print(
            f"  Recall    = "
            f"{m['recall']:.4f}"
        )

        print(
            f"  F1        = "
            f"{m['f1']:.4f}"
        )

        print(
            f"  AUROC     = "
            f"{m['auroc']:.4f}"
        )

        print(
            f"  AUPRC     = "
            f"{m['auprc']:.4f}"
        )

        print(
            f"  FPR       = "
            f"{m['false_positive_rate']:.4f}"
        )

        results.append({

            "building_id":
                building,

            "variant":
                variant_name,

            "precision":
                m["precision"],

            "recall":
                m["recall"],

            "f1":
                m["f1"],

            "auroc":
                m["auroc"],

            "auprc":
                m["auprc"],

            "specificity":
                m["specificity"],

            "false_positive_rate":
                m[
                    "false_positive_rate"
                ],

            "tp":
                m["tp"],

            "fp":
                m["fp"],

            "tn":
                m["tn"],

            "fn":
                m["fn"],

            "threshold":
                frozen_threshold,
        })

        # ----------------------------------------------------
        # SAVE VARIANT WEIGHTS
        # ----------------------------------------------------

        weight_row = {
            "building_id":
                building,

            "variant":
                variant_name,
        }

        weight_row.update(
            variant_weights
        )

        weight_results.append(
            weight_row
        )

        # ----------------------------------------------------
        # ANOMALY TYPE RECALL
        # ----------------------------------------------------

        for anomaly_type in [
            "spike",
            "persistent_shift",
            "gradual_drift",
            "stuck",
            "contextual_after_hours",
        ]:

            mask = (
                df["anomaly_type"]
                ==
                anomaly_type
            )

            n = int(
                mask.sum()
            )

            if n > 0:

                recall_type = float(
                    prediction[mask]
                    .mean()
                )

            else:

                recall_type = np.nan

            type_results.append({

                "building_id":
                    building,

                "variant":
                    variant_name,

                "anomaly_type":
                    anomaly_type,

                "n":
                    n,

                "recall":
                    recall_type,
            })

        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        for i in range(
            len(df)
        ):

            prediction_results.append({

                "building_id":
                    building,

                "timestamp":
                    df.loc[
                        i,
                        "timestamp"
                    ],

                "variant":
                    variant_name,

                "anomaly_label":
                    int(y[i]),

                "anomaly_type":
                    df.loc[
                        i,
                        "anomaly_type"
                    ],

                "score":
                    float(
                        score[i]
                    ),

                "threshold":
                    frozen_threshold,

                "prediction":
                    int(
                        prediction[i]
                    ),
            })


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    results
)

type_df = pd.DataFrame(
    type_results
)

weights_out = pd.DataFrame(
    weight_results
)

predictions_df = pd.DataFrame(
    prediction_results
)


# ============================================================
# SAVE BUILDING RESULTS
# ============================================================

results_df.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_by_building.csv",
    index=False
)

type_df.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_type_recall.csv",
    index=False
)

weights_out.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_weights.csv",
    index=False
)

predictions_df.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_predictions.csv",
    index=False
)


# ============================================================
# MEAN PERFORMANCE
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

mean_results = (
    results_df
    .groupby("variant")[
        metric_columns
    ]
    .mean()
    .reset_index()
)

variant_order = [
    "Full TwinGuard-X v3",
    "w/o Context",
    "w/o Multi-scale Persistence",
    "Residual Only",
    "Equal Fusion",
]

mean_results[
    "variant"
] = pd.Categorical(
    mean_results["variant"],
    categories=variant_order,
    ordered=True
)

mean_results = (
    mean_results
    .sort_values("variant")
    .reset_index(drop=True)
)

mean_results[
    "variant"
] = mean_results[
    "variant"
].astype(str)

mean_results.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_mean.csv",
    index=False
)


# ============================================================
# MEAN TYPE RECALL
# ============================================================

mean_type = (
    type_df
    .groupby(
        [
            "variant",
            "anomaly_type"
        ]
    )["recall"]
    .mean()
    .reset_index()
)

mean_type.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_mean_type_recall.csv",
    index=False
)


# ============================================================
# DIFFERENCE FROM FULL MODEL
# ============================================================

full_row = (
    mean_results[
        mean_results[
            "variant"
        ]
        ==
        "Full TwinGuard-X v3"
    ]
    .iloc[0]
)

difference_rows = []

for _, row in mean_results.iterrows():

    if (
        row["variant"]
        ==
        "Full TwinGuard-X v3"
    ):
        continue

    difference_rows.append({

        "variant":
            row["variant"],

        "delta_precision":
            row["precision"]
            -
            full_row["precision"],

        "delta_recall":
            row["recall"]
            -
            full_row["recall"],

        "delta_f1":
            row["f1"]
            -
            full_row["f1"],

        "delta_auroc":
            row["auroc"]
            -
            full_row["auroc"],

        "delta_auprc":
            row["auprc"]
            -
            full_row["auprc"],

        "delta_fpr":
            row[
                "false_positive_rate"
            ]
            -
            full_row[
                "false_positive_rate"
            ],
    })

difference_df = pd.DataFrame(
    difference_rows
)

difference_df.to_csv(
    TABLES /
    "twinguard_x_v3_ablation_delta_from_full.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 110)
print("MEAN VALIDATION ABLATION PERFORMANCE")
print("=" * 110)

display = mean_results.copy()

for column in metric_columns:

    display[column] = (
        display[column]
        .round(4)
    )

print(
    display.to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("CHANGE RELATIVE TO FULL TWINGUARD-X v3")
print("=" * 110)

delta_display = (
    difference_df.copy()
)

for column in [
    "delta_precision",
    "delta_recall",
    "delta_f1",
    "delta_auroc",
    "delta_auprc",
    "delta_fpr",
]:

    delta_display[column] = (
        delta_display[column]
        .round(4)
    )

print(
    delta_display.to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 110)

pivot = mean_type.pivot(
    index="anomaly_type",
    columns="variant",
    values="recall"
)

existing_order = [
    x
    for x in variant_order
    if x in pivot.columns
]

pivot = pivot[
    existing_order
]

print(
    pivot
    .round(4)
    .to_string()
)


# ============================================================
# CONSISTENCY CHECK FOR FULL MODEL
# ============================================================

print("\n" + "=" * 110)
print("FULL-MODEL CONSISTENCY CHECK")
print("=" * 110)

recomputed_full = (
    results_df[
        results_df[
            "variant"
        ]
        ==
        "Full TwinGuard-X v3"
    ]
    .sort_values("building_id")
    .reset_index(drop=True)
)

saved_full = (
    full_performance
    .sort_values("building_id")
    .reset_index(drop=True)
)

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
            recomputed_full[
                metric
            ].values
            -
            saved_full[
                metric
            ].values
        )
    )

    print(
        f"{metric:25s}: "
        f"max absolute difference = "
        f"{difference:.10f}"
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = {

    "experiment":
        "TwinGuard-X v3 validation ablation",

    "data_partition":
        "validation only",

    "final_test_accessed":
        False,

    "model_retrained":
        False,

    "threshold_recalibrated":
        False,

    "ablation_rule":
        (
            "Remove specified frozen evidence "
            "components and renormalize remaining "
            "frozen reliability weights."
        ),

    "threshold_rule":
        (
            "Retain original frozen building-specific "
            "TwinGuard-X v3 threshold."
        ),

    "variants":
        variant_order,

    "components":
        COMPONENTS,
}

with open(
    TABLES /
    "twinguard_x_v3_ablation_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print("\n" + "=" * 110)
print("STEP 14 COMPLETE")
print("=" * 110)

print("\nSaved:")

for filename in [
    "twinguard_x_v3_ablation_by_building.csv",
    "twinguard_x_v3_ablation_mean.csv",
    "twinguard_x_v3_ablation_type_recall.csv",
    "twinguard_x_v3_ablation_mean_type_recall.csv",
    "twinguard_x_v3_ablation_weights.csv",
    "twinguard_x_v3_ablation_predictions.csv",
    "twinguard_x_v3_ablation_delta_from_full.csv",
    "twinguard_x_v3_ablation_manifest.json",
]:

    print(
        TABLES /
        filename
    )