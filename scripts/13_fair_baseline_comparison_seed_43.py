from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

SELECTION_PATH = TABLES / "selected_three_public_buildings.csv"

V3_FINAL_PATH = (
    TABLES /
    "twinguard_x_v3_FINAL_explanations.csv"
)

V3_PERFORMANCE_PATH = (
    TABLES /
    "twinguard_x_v3_FINAL_locked_test_performance.csv"
)

SEED = 43
EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("EIRT 2026 — STEP 13")
print("FAIR BASELINE COMPARISON ON EXACT TWINGUARD-X v3 FINAL-TEST TIMESTAMPS")
print("=" * 110)

print(f"\nDevice: {DEVICE}")
print(f"Seed: {SEED}")

print(
    "\nTwinGuard-X v3 remains frozen. "
    "This script does not modify or retrain v3."
)


# ============================================================
# FEATURE SET
# ============================================================

FEATURE_COLUMNS = [
    "electricity_original",
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)


# ============================================================
# METRICS
# ============================================================

def metrics(y, pred, score):

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    return {
        "precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "auroc":
            roc_auc_score(
                y,
                score
            ),

        "auprc":
            average_precision_score(
                y,
                score
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
# THRESHOLD CALIBRATION
# ============================================================

def choose_threshold(
    validation_scores,
    validation_labels
):

    """
    Select threshold using VALIDATION ONLY.

    Primary objective:
        maximum F1

    Tie-breakers:
        lower FPR
        higher recall
        higher precision
    """

    percentiles = np.arange(
        70.0,
        99.51,
        0.25
    )

    candidate_thresholds = np.unique(
        np.percentile(
            validation_scores,
            percentiles
        )
    )

    best = None

    for threshold in candidate_thresholds:

        pred = (
            validation_scores
            >
            threshold
        ).astype(int)

        f1 = f1_score(
            validation_labels,
            pred,
            zero_division=0
        )

        precision = precision_score(
            validation_labels,
            pred,
            zero_division=0
        )

        recall = recall_score(
            validation_labels,
            pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            validation_labels,
            pred,
            labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn)

        key = (
            f1,
            -fpr,
            recall,
            precision
        )

        if (
            best is None
            or
            key > best["key"]
        ):

            best = {
                "key": key,
                "threshold":
                    float(threshold),

                "f1":
                    float(f1),

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "fpr":
                    float(fpr),
            }

    return best


# ============================================================
# AUTOENCODER TRAINING
# ============================================================

def train_autoencoder(X_train):

    torch.manual_seed(SEED)

    model = Autoencoder(
        X_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        losses = []

        for (batch,) in loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = criterion(
                reconstruction,
                batch
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
        ):

            print(
                f"    AE epoch "
                f"{epoch + 1:02d}/"
                f"{EPOCHS} "
                f"| loss="
                f"{np.mean(losses):.6f}"
            )

    training_time = (
        time.perf_counter()
        -
        start
    )

    return model, training_time


def ae_scores(model, X):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    start = time.perf_counter()

    with torch.no_grad():

        reconstructed = (
            model(tensor)
            .cpu()
            .numpy()
        )

    inference_time = (
        time.perf_counter()
        -
        start
    )

    score = np.mean(
        (
            X
            -
            reconstructed
        ) ** 2,
        axis=1
    )

    return score, inference_time


# ============================================================
# LOAD DATA
# ============================================================

selected = pd.read_csv(
    SELECTION_PATH
)

buildings = selected[
    "building_id"
].tolist()

v3_final = pd.read_csv(
    V3_FINAL_PATH,
    parse_dates=["timestamp"]
)

v3_performance = pd.read_csv(
    V3_PERFORMANCE_PATH
)

results = []
validation_rows = []
prediction_rows = []


# ============================================================
# BUILDING LOOP
# ============================================================

for building_number, building in enumerate(
    buildings,
    start=1
):

    print("\n" + "=" * 110)
    print(f"BUILDING: {building}")
    print("=" * 110)

    df = pd.read_csv(
        PROCESSED /
        f"{building}_locked_protocol.csv",
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ========================================================
    # EXACT FINAL TIMESTAMPS FROM V3
    # ========================================================

    final_v3 = (
        v3_final[
            v3_final["building_id"]
            ==
            building
        ]
        .copy()
    )

    final_timestamps = set(
        final_v3["timestamp"]
    )

    print(
        f"\nExact v3 final timestamps: "
        f"{len(final_timestamps):,}"
    )

    # ========================================================
    # BUILD TRAIN DATA
    # ========================================================

    train = (
        df[
            df["experimental_split"]
            ==
            "train"
        ]
        .copy()
    )

    X_train_raw = (
        train[
            FEATURE_COLUMNS
        ]
        .astype(float)
        .values
    )

    # ========================================================
    # BUILD VALIDATION DATA
    # ========================================================

    validation = (
        df[
            df["experimental_split"]
            ==
            "validation"
        ]
        .copy()
    )

    # Use observed/injected electricity for validation.
    validation[
        "electricity_model_input"
    ] = validation[
        "electricity_validation"
    ]

    validation_features = [
        "electricity_model_input",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windSpeed",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    X_val_raw = (
        validation[
            validation_features
        ]
        .astype(float)
        .values
    )

    y_val = (
        validation[
            "validation_anomaly_label"
        ]
        .astype(int)
        .values
    )

    # ========================================================
    # BUILD EXACT FINAL DATA
    # ========================================================

    final = (
        df[
            df["timestamp"].isin(
                final_timestamps
            )
        ]
        .copy()
    )

    final = final.merge(
        final_v3[
            [
                "timestamp",
                "electricity_observed",
                "anomaly_label",
                "anomaly_type",
            ]
        ],
        on="timestamp",
        how="inner",
        validate="one_to_one"
    )

    final = (
        final
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Matched baseline final rows: "
        f"{len(final):,}"
    )

    if len(final) != len(final_v3):

        raise RuntimeError(
            "Final timestamp mismatch."
        )

    final[
        "electricity_model_input"
    ] = final[
        "electricity_observed"
    ]

    X_test_raw = (
        final[
            validation_features
        ]
        .astype(float)
        .values
    )

    y_test = (
        final[
            "anomaly_label"
        ]
        .astype(int)
        .values
    )

    anomaly_types = (
        final[
            "anomaly_type"
        ]
        .astype(str)
        .values
    )

    # ========================================================
    # SCALING — TRAIN ONLY
    # ========================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_raw
    )

    X_val = scaler.transform(
        X_val_raw
    )

    X_test = scaler.transform(
        X_test_raw
    )

    print(
        f"Training rows:   "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )

    print(
        f"Final rows:      "
        f"{len(X_test):,}"
    )

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print(
        "\nTraining Isolation Forest..."
    )

    start = time.perf_counter()

    iso = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )

    iso.fit(
        X_train
    )

    iso_train_time = (
        time.perf_counter()
        -
        start
    )

    # Higher score = more anomalous
    val_iso_score = (
        -iso.score_samples(
            X_val
        )
    )

    iso_calibration = choose_threshold(
        val_iso_score,
        y_val
    )

    iso_threshold = (
        iso_calibration[
            "threshold"
        ]
    )

    start = time.perf_counter()

    test_iso_score = (
        -iso.score_samples(
            X_test
        )
    )

    iso_inference_time = (
        time.perf_counter()
        -
        start
    )

    test_iso_pred = (
        test_iso_score
        >
        iso_threshold
    ).astype(int)

    iso_metrics = metrics(
        y_test,
        test_iso_pred,
        test_iso_score
    )

    print(
        f"  Validation F1="
        f"{iso_calibration['f1']:.4f}"
        f" | threshold="
        f"{iso_threshold:.6f}"
    )

    print(
        f"  FINAL F1="
        f"{iso_metrics['f1']:.4f}"
        f" | AUROC="
        f"{iso_metrics['auroc']:.4f}"
        f" | AUPRC="
        f"{iso_metrics['auprc']:.4f}"
        f" | FPR="
        f"{iso_metrics['false_positive_rate']:.4f}"
    )

    # ========================================================
    # LOF
    # ========================================================

    print(
        "\nTraining Local Outlier Factor..."
    )

    start = time.perf_counter()

    lof = LocalOutlierFactor(
        n_neighbors=35,
        novelty=True,
        contamination="auto",
        n_jobs=-1,
    )

    lof.fit(
        X_train
    )

    lof_train_time = (
        time.perf_counter()
        -
        start
    )

    val_lof_score = (
        -lof.score_samples(
            X_val
        )
    )

    lof_calibration = choose_threshold(
        val_lof_score,
        y_val
    )

    lof_threshold = (
        lof_calibration[
            "threshold"
        ]
    )

    start = time.perf_counter()

    test_lof_score = (
        -lof.score_samples(
            X_test
        )
    )

    lof_inference_time = (
        time.perf_counter()
        -
        start
    )

    test_lof_pred = (
        test_lof_score
        >
        lof_threshold
    ).astype(int)

    lof_metrics = metrics(
        y_test,
        test_lof_pred,
        test_lof_score
    )

    print(
        f"  Validation F1="
        f"{lof_calibration['f1']:.4f}"
        f" | threshold="
        f"{lof_threshold:.6f}"
    )

    print(
        f"  FINAL F1="
        f"{lof_metrics['f1']:.4f}"
        f" | AUROC="
        f"{lof_metrics['auroc']:.4f}"
        f" | AUPRC="
        f"{lof_metrics['auprc']:.4f}"
        f" | FPR="
        f"{lof_metrics['false_positive_rate']:.4f}"
    )

    # ========================================================
    # AUTOENCODER
    # ========================================================

    print(
        "\nTraining Autoencoder..."
    )

    ae, ae_train_time = (
        train_autoencoder(
            X_train
        )
    )

    val_ae_score, _ = (
        ae_scores(
            ae,
            X_val
        )
    )

    ae_calibration = choose_threshold(
        val_ae_score,
        y_val
    )

    ae_threshold = (
        ae_calibration[
            "threshold"
        ]
    )

    test_ae_score, ae_inference_time = (
        ae_scores(
            ae,
            X_test
        )
    )

    test_ae_pred = (
        test_ae_score
        >
        ae_threshold
    ).astype(int)

    ae_metrics = metrics(
        y_test,
        test_ae_pred,
        test_ae_score
    )

    print(
        f"  Validation F1="
        f"{ae_calibration['f1']:.4f}"
        f" | threshold="
        f"{ae_threshold:.6f}"
    )

    print(
        f"  FINAL F1="
        f"{ae_metrics['f1']:.4f}"
        f" | AUROC="
        f"{ae_metrics['auroc']:.4f}"
        f" | AUPRC="
        f"{ae_metrics['auprc']:.4f}"
        f" | FPR="
        f"{ae_metrics['false_positive_rate']:.4f}"
    )

    # ========================================================
    # STORE BASELINE RESULTS
    # ========================================================

    model_data = [
        (
            "Isolation Forest",
            iso_metrics,
            iso_train_time,
            iso_inference_time,
            iso_threshold,
            iso_calibration,
            test_iso_score,
            test_iso_pred,
        ),
        (
            "LOF",
            lof_metrics,
            lof_train_time,
            lof_inference_time,
            lof_threshold,
            lof_calibration,
            test_lof_score,
            test_lof_pred,
        ),
        (
            "Autoencoder",
            ae_metrics,
            ae_train_time,
            ae_inference_time,
            ae_threshold,
            ae_calibration,
            test_ae_score,
            test_ae_pred,
        ),
    ]

    for (
        model_name,
        model_metrics,
        train_time,
        inference_time,
        threshold,
        calibration,
        test_score,
        test_pred,
    ) in model_data:

        results.append({
            "building_id":
                building,

            "model":
                model_name,

            "precision":
                model_metrics["precision"],

            "recall":
                model_metrics["recall"],

            "f1":
                model_metrics["f1"],

            "auroc":
                model_metrics["auroc"],

            "auprc":
                model_metrics["auprc"],

            "specificity":
                model_metrics["specificity"],

            "false_positive_rate":
                model_metrics[
                    "false_positive_rate"
                ],

            "tp":
                model_metrics["tp"],

            "fp":
                model_metrics["fp"],

            "tn":
                model_metrics["tn"],

            "fn":
                model_metrics["fn"],

            "training_time_s":
                train_time,

            "inference_time_s":
                inference_time,

            "validation_selected_threshold":
                threshold,

            "validation_f1":
                calibration["f1"],
        })

        validation_rows.append({
            "building_id":
                building,

            "model":
                model_name,

            "validation_f1":
                calibration["f1"],

            "validation_precision":
                calibration["precision"],

            "validation_recall":
                calibration["recall"],

            "validation_fpr":
                calibration["fpr"],

            "selected_threshold":
                threshold,
        })

        for i in range(
            len(final)
        ):

            prediction_rows.append({
                "building_id":
                    building,

                "timestamp":
                    final.loc[
                        i,
                        "timestamp"
                    ],

                "model":
                    model_name,

                "anomaly_label":
                    int(
                        y_test[i]
                    ),

                "anomaly_type":
                    anomaly_types[i],

                "score":
                    float(
                        test_score[i]
                    ),

                "prediction":
                    int(
                        test_pred[i]
                    ),
            })

    # ========================================================
    # ADD FROZEN V3 RESULTS
    # ========================================================

    v3_row = (
        v3_performance[
            v3_performance[
                "building_id"
            ]
            ==
            building
        ]
        .iloc[0]
    )

    results.append({
        "building_id":
            building,

        "model":
            "TwinGuard-X v3",

        "precision":
            float(
                v3_row[
                    "precision"
                ]
            ),

        "recall":
            float(
                v3_row[
                    "recall"
                ]
            ),

        "f1":
            float(
                v3_row[
                    "f1"
                ]
            ),

        "auroc":
            float(
                v3_row[
                    "auroc"
                ]
            ),

        "auprc":
            float(
                v3_row[
                    "auprc"
                ]
            ),

        "specificity":
            float(
                v3_row[
                    "specificity"
                ]
            ),

        "false_positive_rate":
            float(
                v3_row[
                    "false_positive_rate"
                ]
            ),

        "tp":
            int(
                v3_row["tp"]
            ),

        "fp":
            int(
                v3_row["fp"]
            ),

        "tn":
            int(
                v3_row["tn"]
            ),

        "fn":
            int(
                v3_row["fn"]
            ),

        "training_time_s":
            np.nan,

        "inference_time_s":
            np.nan,

        "validation_selected_threshold":
            np.nan,

        "validation_f1":
            np.nan,
    })


# ============================================================
# SAVE PER-BUILDING RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    TABLES /
    "fair_final_model_comparison_by_building.csv",
    index=False
)

validation_df = pd.DataFrame(
    validation_rows
)

validation_df.to_csv(
    TABLES /
    "fair_baseline_validation_calibration.csv",
    index=False
)

predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(
    TABLES /
    "fair_baseline_final_predictions.csv",
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

mean_df = (
    results_df
    .groupby("model")[
        metric_columns
    ]
    .mean()
    .reset_index()
)

mean_df.to_csv(
    TABLES /
    "fair_final_model_comparison_mean.csv",
    index=False
)


# ============================================================
# DISPLAY PER-BUILDING RESULTS
# ============================================================

print("\n" + "=" * 110)
print("FAIR FINAL-TEST PERFORMANCE BY BUILDING")
print("=" * 110)

display = results_df[
    [
        "building_id",
        "model",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "false_positive_rate",
    ]
].copy()

for column in [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "false_positive_rate",
]:

    display[column] = (
        display[column]
        .round(4)
    )

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# DISPLAY MEAN RESULTS
# ============================================================

print("\n" + "=" * 110)
print("MEAN FAIR FINAL-TEST PERFORMANCE")
print("=" * 110)

mean_display = mean_df.copy()

for column in metric_columns:

    mean_display[column] = (
        mean_display[column]
        .round(4)
    )

print(
    mean_display.to_string(
        index=False
    )
)


# ============================================================
# ANOMALY-TYPE RECALL FOR ALL METHODS
# ============================================================

type_rows = []

baseline_predictions = predictions_df.copy()

v3_type_source = (
    v3_final[
        [
            "building_id",
            "timestamp",
            "anomaly_label",
            "anomaly_type",
            "prediction",
        ]
    ]
    .copy()
)

v3_type_source[
    "model"
] = "TwinGuard-X v3"

combined = pd.concat(
    [
        baseline_predictions[
            [
                "building_id",
                "timestamp",
                "model",
                "anomaly_label",
                "anomaly_type",
                "prediction",
            ]
        ],
        v3_type_source[
            [
                "building_id",
                "timestamp",
                "model",
                "anomaly_label",
                "anomaly_type",
                "prediction",
            ]
        ],
    ],
    ignore_index=True
)

anomaly_names = [
    "spike",
    "persistent_shift",
    "gradual_drift",
    "stuck",
    "contextual_after_hours",
]

for (
    building,
    model_name
), group in combined.groupby(
    [
        "building_id",
        "model"
    ]
):

    for anomaly_type in anomaly_names:

        subset = group[
            group[
                "anomaly_type"
            ]
            ==
            anomaly_type
        ]

        n = len(subset)

        recall = (
            float(
                subset[
                    "prediction"
                ]
                .mean()
            )
            if n > 0
            else np.nan
        )

        type_rows.append({
            "building_id":
                building,

            "model":
                model_name,

            "anomaly_type":
                anomaly_type,

            "n":
                n,

            "recall":
                recall,
        })


type_df = pd.DataFrame(
    type_rows
)

type_df.to_csv(
    TABLES /
    "fair_model_anomaly_type_recall.csv",
    index=False
)

mean_type = (
    type_df
    .groupby(
        [
            "model",
            "anomaly_type"
        ]
    )["recall"]
    .mean()
    .reset_index()
)

mean_type.to_csv(
    TABLES /
    "fair_model_mean_anomaly_type_recall.csv",
    index=False
)


print("\n" + "=" * 110)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 110)

pivot = mean_type.pivot(
    index="anomaly_type",
    columns="model",
    values="recall"
)

print(
    pivot
    .round(4)
    .to_string()
)


# ============================================================
# SIMPLE RELATIVE DIFFERENCES
# ============================================================

v3_mean = (
    mean_df[
        mean_df["model"]
        ==
        "TwinGuard-X v3"
    ]
    .iloc[0]
)

comparison_rows = []

for _, row in mean_df.iterrows():

    if (
        row["model"]
        ==
        "TwinGuard-X v3"
    ):
        continue

    comparison_rows.append({
        "baseline":
            row["model"],

        "f1_difference":
            v3_mean["f1"]
            -
            row["f1"],

        "auroc_difference":
            v3_mean["auroc"]
            -
            row["auroc"],

        "auprc_difference":
            v3_mean["auprc"]
            -
            row["auprc"],

        "fpr_difference":
            v3_mean[
                "false_positive_rate"
            ]
            -
            row[
                "false_positive_rate"
            ],
    })

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    TABLES /
    "twinguard_v3_relative_to_baselines.csv",
    index=False
)


print("\n" + "=" * 110)
print("TWINGUARD-X v3 MINUS BASELINE")
print("=" * 110)

print(
    comparison_df
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("STEP 13 COMPLETE")
print("=" * 110)

print("\nSaved:")
print(
    TABLES /
    "fair_final_model_comparison_by_building.csv"
)
print(
    TABLES /
    "fair_final_model_comparison_mean.csv"
)
print(
    TABLES /
    "fair_baseline_validation_calibration.csv"
)
print(
    TABLES /
    "fair_baseline_final_predictions.csv"
)
print(
    TABLES /
    "fair_model_anomaly_type_recall.csv"
)
print(
    TABLES /
    "fair_model_mean_anomaly_type_recall.csv"
)
print(
    TABLES /
    "twinguard_v3_relative_to_baselines.csv"
)