from pathlib import Path
import json
import warnings

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

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "models"

SELECTION_PATH = TABLES / "selected_three_public_buildings.csv"
PROTOCOL_PATH = TABLES / "locked_protocol.json"
STEP8_PATH = TABLES / "twinguard_x_explanations.csv"
FREEZE_PATH = TABLES / "twinguard_x_v3_freeze_manifest.json"

HISTORY = 24

CONTEXT_COLUMNS = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

DISAGREEMENT_NAMES = [
    "contextual_residual",
    "residual_dynamics",
    "persistence_3h",
    "persistence_6h",
    "persistence_12h",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("EIRT 2026 — TWINGUARD-X v3")
print("ONE-SHOT LOCKED FINAL-TEST EVALUATION")
print("=" * 110)

print(f"\nDevice: {DEVICE}")


# ============================================================
# LOAD PROTOCOL
# ============================================================

with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
    protocol = json.load(f)

TRAIN_WEEKS = protocol["training_weeks"]
VALIDATION_WEEKS = protocol["validation_weeks"]
FINAL_TEST_WEEKS = protocol["final_test_weeks"]

print("\nLocked final-test weeks:")
print(FINAL_TEST_WEEKS)


# ============================================================
# FREEZE CHECK
# ============================================================

with open(FREEZE_PATH, "r", encoding="utf-8") as f:
    freeze_manifest = json.load(f)

if freeze_manifest.get("final_test_evaluated", False):
    raise RuntimeError(
        "The manifest says that the final test has already been evaluated. "
        "Do not repeatedly tune against the locked final test."
    )

freeze_manifest["status"] = "FROZEN_BEFORE_FINAL_TEST"

with open(FREEZE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        freeze_manifest,
        f,
        indent=4
    )

print("\nModel status: FROZEN")
print("No training or calibration will occur.")


# ============================================================
# MODEL — MUST MATCH STEP 11 EXACTLY
# ============================================================

class OperationalDigitalTwin(nn.Module):

    def __init__(self, context_dim):

        super().__init__()

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv1d(
            1, 8,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            8, 16,
            kernel_size=3,
            dilation=2,
            padding=2,
        )

        self.conv3 = nn.Conv1d(
            16, 8,
            kernel_size=3,
            dilation=4,
            padding=4,
        )

        self.activation = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fusion = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def operational_encode(self, history):

        x = self.activation(
            self.conv1(history)
        )

        x = self.activation(
            self.conv2(x)
        )

        x = self.activation(
            self.conv3(x)
        )

        x = self.pool(x)

        return x.squeeze(-1)

    def forward(self, context, history):

        context_state = self.context_encoder(
            context
        )

        operational_state = self.operational_encode(
            history
        )

        state = torch.cat(
            [
                context_state,
                operational_state,
            ],
            dim=1,
        )

        state = self.fusion(state)

        expected = self.output(state)

        return expected


# ============================================================
# PERSISTENCE
# ============================================================

def persistence_features(history):

    history = np.asarray(
        history,
        dtype=float
    )

    reference_std = (
        np.std(history)
        + 1e-6
    )

    output = []

    for k in [3, 6, 12]:

        local_std = np.std(
            history[-k:]
        )

        ratio = (
            local_std
            /
            reference_std
        )

        stagnation = (
            1.0
            /
            (
                ratio
                +
                0.10
            )
        )

        output.append(
            stagnation
        )

    return output


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y,
    prediction,
    score,
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                prediction,
                zero_division=0,
            ),

        "auroc":
            roc_auc_score(
                y,
                score,
            ),

        "auprc":
            average_precision_score(
                y,
                score,
            ),

        "specificity":
            tn / (tn + fp),

        "false_positive_rate":
            fp / (fp + tn),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ============================================================
# LOAD ORIGINAL LOCKED TEST ANOMALIES
# ============================================================

step8 = pd.read_csv(
    STEP8_PATH,
    parse_dates=["timestamp"]
)

selected = pd.read_csv(
    SELECTION_PATH
)

buildings = selected[
    "building_id"
].tolist()

performance_rows = []
type_rows = []
explanation_rows = []


# ============================================================
# BUILDING LOOP
# ============================================================

for building in buildings:

    print("\n" + "=" * 110)
    print(f"FINAL TEST: {building}")
    print("=" * 110)

    # --------------------------------------------------------
    # Load protocol data
    # --------------------------------------------------------

    df = pd.read_csv(
        PROCESSED /
        f"{building}_locked_protocol.csv",
        parse_dates=["timestamp"],
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Restore exact Step-8 locked test anomalies
    # --------------------------------------------------------

    df["electricity_test"] = (
        df["electricity_original"]
        .astype(float)
    )

    df["test_anomaly_label"] = 0
    df["test_anomaly_type"] = "normal"

    old = (
        step8[
            step8["building_id"]
            ==
            building
        ]
        .copy()
    )

    old_lookup = old.set_index(
        "timestamp"
    )

    matched = 0

    for i in df.index[
        df["experimental_split"]
        ==
        "final_test"
    ]:

        timestamp = df.loc[
            i,
            "timestamp"
        ]

        if timestamp in old_lookup.index:

            row = old_lookup.loc[
                timestamp
            ]

            if isinstance(
                row,
                pd.DataFrame
            ):
                row = row.iloc[0]

            df.loc[
                i,
                "electricity_test"
            ] = row[
                "electricity_observed"
            ]

            df.loc[
                i,
                "test_anomaly_label"
            ] = int(
                row[
                    "anomaly_label"
                ]
            )

            df.loc[
                i,
                "test_anomaly_type"
            ] = row[
                "anomaly_type"
            ]

            matched += 1

    print(
        f"\nMatched locked-test observations: "
        f"{matched:,}"
    )

    print(
        f"Injected locked-test anomalies: "
        f"{df.loc[df['experimental_split']=='final_test', 'test_anomaly_label'].sum():,}"
    )

    # ========================================================
    # LOAD FROZEN PARAMETERS
    # ========================================================

    calibration = np.load(
        MODELS /
        f"{building}_TwinGuardX_v3_calibration.npz"
    )

    weights = calibration[
        "weights"
    ]

    threshold = float(
        calibration[
            "threshold"
        ]
    )

    electricity_mean = float(
        calibration[
            "electricity_mean"
        ]
    )

    electricity_std = float(
        calibration[
            "electricity_std"
        ]
    )

    context_mean = calibration[
        "context_scaler_mean"
    ]

    context_scale = calibration[
        "context_scaler_scale"
    ]

    descriptor_median = calibration[
        "descriptor_median"
    ]

    descriptor_scale = calibration[
        "descriptor_scale"
    ]

    temp_quartiles = calibration[
        "residual_temp_quartiles"
    ]

    global_residual_median = float(
        calibration[
            "residual_global_median"
        ]
    )

    global_residual_mad = float(
        calibration[
            "residual_global_mad"
        ]
    )

    context_bins = pd.read_csv(
        MODELS /
        f"{building}_TwinGuardX_v3_context_bins.csv"
    )

    context_bin_lookup = {}

    for _, row in (
        context_bins.iterrows()
    ):

        context_bin_lookup[
            (
                int(
                    row[
                        "hour_group"
                    ]
                ),
                int(
                    row[
                        "temperature_group"
                    ]
                ),
            )
        ] = (
            float(
                row[
                    "residual_median"
                ]
            ),
            float(
                row[
                    "residual_mad"
                ]
            ),
        )

    print("\nFrozen reliability weights:")

    for name, weight in zip(
        DISAGREEMENT_NAMES,
        weights,
    ):

        print(
            f"  {name:24s}: "
            f"{weight:.4f}"
        )

    print(
        f"Frozen threshold: "
        f"{threshold:.6f}"
    )

    # ========================================================
    # LOAD FROZEN DIGITAL TWIN
    # ========================================================

    model = OperationalDigitalTwin(
        len(CONTEXT_COLUMNS)
    ).to(DEVICE)

    state = torch.load(
        MODELS /
        f"{building}_TwinGuardX_v3_twin.pt",
        map_location=DEVICE,
    )

    model.load_state_dict(
        state
    )

    model.eval()

    # ========================================================
    # BUILD FINAL TEST SAMPLES
    # ========================================================

    samples = []

    electricity = (
        df["electricity_test"]
        .astype(float)
        .values
    )

    for i in range(
        HISTORY,
        len(df)
    ):

        if (
            df.loc[
                i,
                "experimental_split"
            ]
            !=
            "final_test"
        ):
            continue

        history_splits = (
            df.loc[
                i-HISTORY:i-1,
                "experimental_split"
            ]
            .values
        )

        # Same leakage rule as Step 11.
        if not np.all(
            history_splits
            ==
            "final_test"
        ):
            continue

        history = electricity[
            i-HISTORY:i
        ]

        context = (
            df.loc[
                i,
                CONTEXT_COLUMNS
            ]
            .astype(float)
            .values
        )

        if not np.all(
            np.isfinite(history)
        ):
            continue

        if not np.all(
            np.isfinite(context)
        ):
            continue

        samples.append({
            "index": i,
            "history": history.copy(),
            "context": context.copy(),
            "target": float(
                electricity[i]
            ),
        })

    print(
        f"\nValid locked-test sequences: "
        f"{len(samples):,}"
    )

    # ========================================================
    # DIGITAL-TWIN INFERENCE
    # ========================================================

    context_raw = np.vstack([
        x["context"]
        for x in samples
    ])

    context_scaled = (
        context_raw
        -
        context_mean
    ) / context_scale

    histories = np.stack([
        (
            x["history"]
            -
            electricity_mean
        )
        /
        electricity_std
        for x in samples
    ])

    histories = histories[
        :,
        None,
        :
    ]

    with torch.no_grad():

        predicted_scaled = model(
            torch.tensor(
                context_scaled,
                dtype=torch.float32,
            ).to(DEVICE),

            torch.tensor(
                histories,
                dtype=torch.float32,
            ).to(DEVICE),
        )

    predicted = (
        predicted_scaled
        .cpu()
        .numpy()
        .ravel()
        *
        electricity_std
        +
        electricity_mean
    )

    indices = np.asarray([
        x["index"]
        for x in samples
    ])

    observed = np.asarray([
        x["target"]
        for x in samples
    ])

    original = (
        df.loc[
            indices,
            "electricity_original",
        ]
        .astype(float)
        .values
    )

    labels = (
        df.loc[
            indices,
            "test_anomaly_label",
        ]
        .astype(int)
        .values
    )

    types = (
        df.loc[
            indices,
            "test_anomaly_type",
        ]
        .astype(str)
        .values
    )

    # ========================================================
    # CONTEXTUAL RESIDUAL
    # ========================================================

    raw_residual = np.abs(
        observed
        -
        predicted
    )

    contextual_residual = []

    for idx, residual in zip(
        indices,
        raw_residual,
    ):

        hour = int(
            df.loc[
                idx,
                "hour"
            ]
        )

        temp = float(
            df.loc[
                idx,
                "airTemperature"
            ]
        )

        hour_group = (
            hour // 6
        )

        temp_group = int(
            np.digitize(
                temp,
                temp_quartiles
            )
        )

        med, mad = (
            context_bin_lookup.get(
                (
                    hour_group,
                    temp_group,
                ),
                (
                    global_residual_median,
                    global_residual_mad,
                ),
            )
        )

        value = (
            residual
            -
            med
        ) / (
            mad
            +
            1e-6
        )

        contextual_residual.append(
            max(
                0.0,
                value
            )
        )

    contextual_residual = np.asarray(
        contextual_residual
    )

    # ========================================================
    # DYNAMICS
    # ========================================================

    dynamics = np.abs(
        np.diff(
            contextual_residual,
            prepend=
            contextual_residual[0],
        )
    )

    # ========================================================
    # PERSISTENCE
    # ========================================================

    persistence = np.asarray([
        persistence_features(
            x["history"]
        )
        for x in samples
    ])

    descriptor_raw = np.column_stack([
        contextual_residual,
        dynamics,
        persistence,
    ])

    descriptor = (
        descriptor_raw
        -
        descriptor_median
    ) / descriptor_scale

    descriptor = np.maximum(
        descriptor,
        0.0,
    )

    descriptor = np.clip(
        descriptor,
        0.0,
        10.0,
    )

    # ========================================================
    # FROZEN RELIABILITY FUSION
    # ========================================================

    score = (
        descriptor
        @
        weights
    )

    prediction = (
        score
        >
        threshold
    ).astype(int)

    # ========================================================
    # FINAL METRICS
    # ========================================================

    metrics = calculate_metrics(
        labels,
        prediction,
        score,
    )

    twin_mae = float(
        np.mean(
            np.abs(
                original
                -
                predicted
            )
        )
    )

    twin_rmse = float(
        np.sqrt(
            np.mean(
                (
                    original
                    -
                    predicted
                ) ** 2
            )
        )
    )

    print(
        "\nLOCKED FINAL-TEST PERFORMANCE:"
    )

    print(
        f"  Precision = "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall    = "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1        = "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  AUROC     = "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"  AUPRC     = "
        f"{metrics['auprc']:.4f}"
    )

    print(
        f"  FPR       = "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print(
        f"  Twin MAE  = "
        f"{twin_mae:.4f}"
    )

    print(
        f"  Twin RMSE = "
        f"{twin_rmse:.4f}"
    )

    performance_rows.append({
        "building_id":
            building,

        "precision":
            metrics[
                "precision"
            ],

        "recall":
            metrics[
                "recall"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "auroc":
            metrics[
                "auroc"
            ],

        "auprc":
            metrics[
                "auprc"
            ],

        "specificity":
            metrics[
                "specificity"
            ],

        "false_positive_rate":
            metrics[
                "false_positive_rate"
            ],

        "tp":
            metrics["tp"],

        "fp":
            metrics["fp"],

        "tn":
            metrics["tn"],

        "fn":
            metrics["fn"],

        "twin_mae":
            twin_mae,

        "twin_rmse":
            twin_rmse,
    })

    # ========================================================
    # ANOMALY TYPE RECALL
    # ========================================================

    print(
        "\nRecall by anomaly type:"
    )

    for anomaly_type in [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours",
    ]:

        mask = (
            types
            ==
            anomaly_type
        )

        n = int(
            mask.sum()
        )

        recall = (
            float(
                prediction[
                    mask
                ].mean()
            )
            if n > 0
            else np.nan
        )

        print(
            f"  {anomaly_type:25s} "
            f"{recall:.4f} "
            f"(n={n})"
        )

        type_rows.append({
            "building_id":
                building,

            "anomaly_type":
                anomaly_type,

            "n":
                n,

            "recall":
                recall,
        })

    # ========================================================
    # EXPLANATION TABLE
    # ========================================================

    explanation = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            df.loc[
                indices,
                "timestamp",
            ].values,

        "anomaly_label":
            labels,

        "anomaly_type":
            types,

        "electricity_original":
            original,

        "electricity_observed":
            observed,

        "digital_twin_expected":
            predicted,

        "contextual_residual":
            descriptor[:, 0],

        "residual_dynamics":
            descriptor[:, 1],

        "persistence_3h":
            descriptor[:, 2],

        "persistence_6h":
            descriptor[:, 3],

        "persistence_12h":
            descriptor[:, 4],

        "anomaly_score":
            score,

        "threshold":
            threshold,

        "prediction":
            prediction,
    })

    explanation_rows.append(
        explanation
    )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

performance = pd.DataFrame(
    performance_rows
)

performance.to_csv(
    TABLES /
    "twinguard_x_v3_FINAL_locked_test_performance.csv",
    index=False,
)

types_df = pd.DataFrame(
    type_rows
)

types_df.to_csv(
    TABLES /
    "twinguard_x_v3_FINAL_type_recall.csv",
    index=False,
)

explanations = pd.concat(
    explanation_rows,
    ignore_index=True,
)

explanations.to_csv(
    TABLES /
    "twinguard_x_v3_FINAL_explanations.csv",
    index=False,
)


# ============================================================
# DISPLAY FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 110)
print("TWINGUARD-X v3 — LOCKED FINAL TEST")
print("=" * 110)

display = performance.copy()

numeric = (
    display
    .select_dtypes(
        include=np.number
    )
    .columns
)

display[numeric] = (
    display[numeric]
    .round(4)
)

print(
    display.to_string(
        index=False
    )
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

mean_performance = (
    performance[
        metric_columns
    ]
    .mean()
)

print("\n" + "=" * 110)
print("MEAN LOCKED FINAL-TEST PERFORMANCE")
print("=" * 110)

for metric, value in (
    mean_performance.items()
):

    print(
        f"{metric:25s}: "
        f"{value:.4f}"
    )


# ============================================================
# MEAN TYPE RECALL
# ============================================================

mean_type = (
    types_df
    .groupby(
        "anomaly_type"
    )["recall"]
    .mean()
)

print("\n" + "=" * 110)
print("MEAN LOCKED FINAL-TEST RECALL BY ANOMALY TYPE")
print("=" * 110)

print(
    mean_type
    .round(4)
    .to_string()
)


# ============================================================
# MARK FINAL TEST AS EVALUATED
# ============================================================

freeze_manifest[
    "final_test_evaluated"
] = True

freeze_manifest[
    "status"
] = "FINAL_TEST_EVALUATED_DO_NOT_RETUNE"

freeze_manifest[
    "final_test_results_file"
] = (
    "twinguard_x_v3_FINAL_locked_test_performance.csv"
)

freeze_manifest[
    "warning"
] = (
    "The locked final-test results have now been observed. "
    "Do not modify model architecture, reliability weights, "
    "thresholds, preprocessing, or hyperparameters based on "
    "these results and then report evaluation on the same "
    "test set as unseen performance."
)

with open(
    FREEZE_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=4,
    )


print("\n" + "=" * 110)
print("STEP 12 COMPLETE — FINAL TEST IS NOW CONSUMED")
print("=" * 110)

print("\nSaved:")
print(
    TABLES /
    "twinguard_x_v3_FINAL_locked_test_performance.csv"
)
print(
    TABLES /
    "twinguard_x_v3_FINAL_type_recall.csv"
)
print(
    TABLES /
    "twinguard_x_v3_FINAL_explanations.csv"
)

print(
    "\nIMPORTANT: Do not tune TwinGuard-X v3 "
    "against these final-test results."
)