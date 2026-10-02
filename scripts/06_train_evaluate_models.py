from pathlib import Path
import time
import random
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
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "models"

TABLES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SELECTION_PATH = (
    TABLES /
    "selected_three_public_buildings.csv"
)

SEED = 42
THRESHOLD_PERCENTILE = 95

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 95)
print("EIRT 2026 — MODEL TRAINING AND EVALUATION")
print("=" * 95)

print(f"\nDevice: {DEVICE}")
print(f"Random seed: {SEED}")
print(
    f"Training-score threshold percentile: "
    f"{THRESHOLD_PERCENTILE}"
)

# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
    "electricity",
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "electricity_lag_1h",
    "electricity_lag_24h",
    "electricity_rolling_24h",
    "electricity_rolling_std_24h"
]

TEST_RENAME = {
    "electricity_injected":
        "electricity",

    "electricity_lag_1h_injected":
        "electricity_lag_1h",

    "electricity_lag_24h_injected":
        "electricity_lag_24h",

    "electricity_rolling_24h_injected":
        "electricity_rolling_24h",

    "electricity_rolling_std_24h_injected":
        "electricity_rolling_std_24h"
}

# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        hidden1 = max(
            8,
            input_dim
        )

        hidden2 = max(
            4,
            input_dim // 2
        )

        latent = 3

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden1
            ),
            nn.ReLU(),

            nn.Linear(
                hidden1,
                hidden2
            ),
            nn.ReLU(),

            nn.Linear(
                hidden2,
                latent
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent,
                hidden2
            ),
            nn.ReLU(),

            nn.Linear(
                hidden2,
                hidden1
            ),
            nn.ReLU(),

            nn.Linear(
                hidden1,
                input_dim
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        return self.decoder(z)


# ============================================================
# AUTOENCODER TRAINING
# ============================================================

def train_autoencoder(
    X_train,
    input_dim,
    building
):

    model = Autoencoder(
        input_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    epochs = 40

    model.train()

    start_time = time.perf_counter()

    for epoch in range(epochs):

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
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"  AE epoch "
                f"{epoch + 1:02d}/{epochs} "
                f"| loss="
                f"{np.mean(losses):.6f}"
            )

    training_time = (
        time.perf_counter()
        -
        start_time
    )

    model_path = (
        MODELS /
        f"{building}_autoencoder.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    return (
        model,
        training_time
    )


# ============================================================
# AUTOENCODER SCORES
# ============================================================

def ae_scores(
    model,
    X
):

    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        reconstruction = model(
            tensor
        )

        errors = torch.mean(
            (
                tensor
                -
                reconstruction
            ) ** 2,
            dim=1
        )

    return (
        errors
        .cpu()
        .numpy()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    scores
):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        auroc = roc_auc_score(
            y_true,
            scores
        )

    except Exception:

        auroc = np.nan

    try:

        auprc = average_precision_score(
            y_true,
            scores
        )

    except Exception:

        auprc = np.nan

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "specificity": specificity,
        "false_positive_rate":
            false_positive_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# ============================================================
# LOAD BUILDINGS
# ============================================================

selected = pd.read_csv(
    SELECTION_PATH
)

buildings = (
    selected["building_id"]
    .tolist()
)

all_results = []
all_predictions = []
type_results = []

# ============================================================
# PROCESS EACH BUILDING
# ============================================================

for building in buildings:

    print("\n" + "=" * 95)
    print(f"BUILDING: {building}")
    print("=" * 95)

    path = (
        PROCESSED /
        f"{building}_with_anomalies.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    # --------------------------------------------------------
    # Training dataframe
    # --------------------------------------------------------

    train = df[
        df["split"] == "train"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    print(
        f"\nTrain observations: "
        f"{len(train):,}"
    )

    print(
        f"Test observations:  "
        f"{len(test):,}"
    )

    print(
        f"Test anomalies:      "
        f"{int(test['anomaly_label'].sum()):,}"
    )

    # --------------------------------------------------------
    # Training features
    # --------------------------------------------------------

    X_train_df = (
        train[
            BASE_FEATURES
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Test features
    # --------------------------------------------------------

    # Start with non-electricity context features
    X_test_df = pd.DataFrame(
        index=test.index
    )

    X_test_df[
        "electricity"
    ] = (
        test[
            "electricity_injected"
        ]
    )

    for feature in [
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windSpeed",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos"
    ]:

        X_test_df[
            feature
        ] = test[
            feature
        ]

    X_test_df[
        "electricity_lag_1h"
    ] = test[
        "electricity_lag_1h_injected"
    ]

    X_test_df[
        "electricity_lag_24h"
    ] = test[
        "electricity_lag_24h_injected"
    ]

    X_test_df[
        "electricity_rolling_24h"
    ] = test[
        "electricity_rolling_24h_injected"
    ]

    X_test_df[
        "electricity_rolling_std_24h"
    ] = test[
        "electricity_rolling_std_24h_injected"
    ]

    # Ensure exact same order
    X_test_df = X_test_df[
        BASE_FEATURES
    ]

    # --------------------------------------------------------
    # Handle any residual missing values
    # --------------------------------------------------------

    train_medians = (
        X_train_df.median()
    )

    X_train_df = (
        X_train_df
        .fillna(
            train_medians
        )
    )

    X_test_df = (
        X_test_df
        .fillna(
            train_medians
        )
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_df
    )

    X_test = scaler.transform(
        X_test_df
    )

    y_test = (
        test["anomaly_label"]
        .astype(int)
        .values
    )

    # ========================================================
    # MODEL 1 — ISOLATION FOREST
    # ========================================================

    print(
        "\nTraining Isolation Forest..."
    )

    start = time.perf_counter()

    iforest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )

    iforest.fit(
        X_train
    )

    if_train_time = (
        time.perf_counter()
        -
        start
    )

    # sklearn decision_function:
    # higher = more normal
    # therefore negate for anomaly score
    if_train_scores = (
        -iforest.decision_function(
            X_train
        )
    )

    start = time.perf_counter()

    if_test_scores = (
        -iforest.decision_function(
            X_test
        )
    )

    if_inference_time = (
        time.perf_counter()
        -
        start
    )

    if_threshold = np.percentile(
        if_train_scores,
        THRESHOLD_PERCENTILE
    )

    if_pred = (
        if_test_scores
        >
        if_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        if_pred,
        if_test_scores
    )

    all_results.append({
        "building_id": building,
        "model": "Isolation Forest",
        "threshold": if_threshold,
        "training_time_s":
            if_train_time,
        "inference_time_s":
            if_inference_time,
        **metrics
    })

    print(
        f"  F1={metrics['f1']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"AUROC={metrics['auroc']:.4f}"
    )

    # ========================================================
    # MODEL 2 — LOF NOVELTY DETECTION
    # ========================================================

    print(
        "\nTraining Local Outlier Factor..."
    )

    start = time.perf_counter()

    lof = LocalOutlierFactor(
        n_neighbors=35,
        novelty=True,
        contamination="auto",
        n_jobs=-1
    )

    lof.fit(
        X_train
    )

    lof_train_time = (
        time.perf_counter()
        -
        start
    )

    lof_train_scores = (
        -lof.decision_function(
            X_train
        )
    )

    start = time.perf_counter()

    lof_test_scores = (
        -lof.decision_function(
            X_test
        )
    )

    lof_inference_time = (
        time.perf_counter()
        -
        start
    )

    lof_threshold = np.percentile(
        lof_train_scores,
        THRESHOLD_PERCENTILE
    )

    lof_pred = (
        lof_test_scores
        >
        lof_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        lof_pred,
        lof_test_scores
    )

    all_results.append({
        "building_id": building,
        "model": "LOF",
        "threshold":
            lof_threshold,
        "training_time_s":
            lof_train_time,
        "inference_time_s":
            lof_inference_time,
        **metrics
    })

    print(
        f"  F1={metrics['f1']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"AUROC={metrics['auroc']:.4f}"
    )

    # ========================================================
    # MODEL 3 — AUTOENCODER
    # ========================================================

    print(
        "\nTraining Autoencoder..."
    )

    autoencoder, ae_train_time = (
        train_autoencoder(
            X_train,
            X_train.shape[1],
            building
        )
    )

    ae_train_scores = ae_scores(
        autoencoder,
        X_train
    )

    start = time.perf_counter()

    ae_test_scores = ae_scores(
        autoencoder,
        X_test
    )

    ae_inference_time = (
        time.perf_counter()
        -
        start
    )

    ae_threshold = np.percentile(
        ae_train_scores,
        THRESHOLD_PERCENTILE
    )

    ae_pred = (
        ae_test_scores
        >
        ae_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        ae_pred,
        ae_test_scores
    )

    all_results.append({
        "building_id": building,
        "model": "Autoencoder",
        "threshold":
            ae_threshold,
        "training_time_s":
            ae_train_time,
        "inference_time_s":
            ae_inference_time,
        **metrics
    })

    print(
        f"  F1={metrics['f1']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"AUROC={metrics['auroc']:.4f}"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            test["timestamp"].values,

        "anomaly_label":
            y_test,

        "anomaly_type":
            test["anomaly_type"].values,

        "anomaly_event_id":
            test[
                "anomaly_event_id"
            ].values,

        "electricity_original":
            test[
                "electricity_original"
            ].values,

        "electricity_injected":
            test[
                "electricity_injected"
            ].values,

        "if_score":
            if_test_scores,

        "if_prediction":
            if_pred,

        "lof_score":
            lof_test_scores,

        "lof_prediction":
            lof_pred,

        "ae_score":
            ae_test_scores,

        "ae_prediction":
            ae_pred
    })

    all_predictions.append(
        prediction_df
    )

    # ========================================================
    # ANOMALY-TYPE RECALL
    # ========================================================

    model_predictions = {
        "Isolation Forest":
            if_pred,
        "LOF":
            lof_pred,
        "Autoencoder":
            ae_pred
    }

    anomaly_types = [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours"
    ]

    test_types = (
        test["anomaly_type"]
        .values
    )

    for model_name, pred in (
        model_predictions.items()
    ):

        for anomaly_type in (
            anomaly_types
        ):

            mask = (
                test_types
                ==
                anomaly_type
            )

            n = int(
                mask.sum()
            )

            if n == 0:
                recall = np.nan
            else:
                recall = float(
                    pred[mask].mean()
                )

            type_results.append({
                "building_id":
                    building,
                "model":
                    model_name,
                "anomaly_type":
                    anomaly_type,
                "n_anomalous_hours":
                    n,
                "recall":
                    recall
            })


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    all_results
)

results_path = (
    TABLES /
    "model_performance_by_building.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 95)
print("MODEL PERFORMANCE")
print("=" * 95)

show_cols = [
    "building_id",
    "model",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate",
    "training_time_s",
    "inference_time_s"
]

display = results[
    show_cols
].copy()

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

mean_results = (
    results
    .groupby(
        "model",
        as_index=False
    )[
        [
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "specificity",
            "false_positive_rate",
            "training_time_s",
            "inference_time_s"
        ]
    ]
    .mean()
)

mean_path = (
    TABLES /
    "model_mean_performance.csv"
)

mean_results.to_csv(
    mean_path,
    index=False
)

print("\n" + "=" * 95)
print("MEAN PERFORMANCE ACROSS THREE BUILDINGS")
print("=" * 95)

mean_display = (
    mean_results.copy()
)

num = (
    mean_display
    .select_dtypes(
        include=np.number
    )
    .columns
)

mean_display[num] = (
    mean_display[num]
    .round(4)
)

print(
    mean_display.to_string(
        index=False
    )
)

# ============================================================
# ANOMALY TYPE RESULTS
# ============================================================

type_df = pd.DataFrame(
    type_results
)

type_path = (
    TABLES /
    "anomaly_type_recall.csv"
)

type_df.to_csv(
    type_path,
    index=False
)

type_mean = (
    type_df
    .groupby(
        [
            "model",
            "anomaly_type"
        ],
        as_index=False
    )["recall"]
    .mean()
)

type_mean_path = (
    TABLES /
    "anomaly_type_mean_recall.csv"
)

type_mean.to_csv(
    type_mean_path,
    index=False
)

print("\n" + "=" * 95)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 95)

pivot = type_mean.pivot(
    index="anomaly_type",
    columns="model",
    values="recall"
)

print(
    pivot.round(4)
    .to_string()
)

# ============================================================
# SAVE ALL TEST PREDICTIONS
# ============================================================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_path = (
    TABLES /
    "all_test_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)

print("\nSaved:")
print(results_path)
print(mean_path)
print(type_path)
print(type_mean_path)
print(predictions_path)

print("\n" + "=" * 95)
print("STEP 6 COMPLETE")
print("=" * 95)