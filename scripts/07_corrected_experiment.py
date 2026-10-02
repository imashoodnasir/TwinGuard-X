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
    TABLES / "selected_three_public_buildings.csv"
)

SEED = 42
TRAIN_WEEK_FRACTION = 0.70
THRESHOLD_PERCENTILE = 95

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

FEATURES = [
    "electricity",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
]

print("=" * 100)
print("EIRT 2026 — CORRECTED BLOCKED EXPERIMENT")
print("=" * 100)

print(f"\nDevice: {DEVICE}")
print(f"Seed: {SEED}")
print(
    f"Training week fraction: "
    f"{TRAIN_WEEK_FRACTION}"
)
print(
    f"Threshold percentile: "
    f"{THRESHOLD_PERCENTILE}"
)

# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):

        return self.decoder(
            self.encoder(x)
        )


def train_autoencoder(
    X_train,
    building
):

    model = Autoencoder(
        X_train.shape[1]
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

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=64,
        shuffle=True
    )

    start = time.perf_counter()

    model.train()

    for epoch in range(40):

        losses = []

        for (batch,) in loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            reconstructed = model(
                batch
            )

            loss = criterion(
                reconstructed,
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
                f"{epoch + 1:02d}/40 "
                f"| loss="
                f"{np.mean(losses):.6f}"
            )

    training_time = (
        time.perf_counter() - start
    )

    torch.save(
        model.state_dict(),
        MODELS /
        f"{building}_blocked_autoencoder.pt"
    )

    return model, training_time


def reconstruction_scores(
    model,
    X
):

    model.eval()

    x = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        reconstructed = model(x)

        error = torch.mean(
            (x - reconstructed) ** 2,
            dim=1
        )

    return (
        error.cpu().numpy()
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y,
    prediction,
    score
):

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                prediction,
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

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


# ============================================================
# NON-OVERLAPPING INTERVAL FINDER
# ============================================================

def find_interval(
    candidates,
    duration,
    occupied,
    rng
):

    candidates = np.asarray(
        sorted(candidates)
    )

    candidate_set = set(
        candidates.tolist()
    )

    possible_starts = (
        candidates.copy()
    )

    rng.shuffle(
        possible_starts
    )

    for start in possible_starts:

        interval = list(
            range(
                int(start),
                int(start) + duration
            )
        )

        if not all(
            i in candidate_set
            for i in interval
        ):
            continue

        if any(
            i in occupied
            for i in interval
        ):
            continue

        return np.array(
            interval,
            dtype=int
        )

    return None


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
all_type_results = []
all_split_summaries = []
all_events = []

# ============================================================
# BUILDING LOOP
# ============================================================

for building_number, building in enumerate(
    buildings,
    start=1
):

    print("\n" + "=" * 100)
    print(f"BUILDING: {building}")
    print("=" * 100)

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the CLEAN pre-injection dataset from Step 4
    # --------------------------------------------------------

    path = (
        PROCESSED /
        f"{building}_experiment.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Ignore old chronological split
    if "split" in df.columns:
        df = df.drop(
            columns=["split"]
        )

    # --------------------------------------------------------
    # Create week blocks
    # --------------------------------------------------------

    # Period starts on 2016-07-02 after the 24-h feature
    # rows were removed in Step 4.
    # Assign each observation to a 7-day block.

    first_time = (
        df["timestamp"].min()
    )

    hours_from_start = (
        (
            df["timestamp"]
            -
            first_time
        )
        .dt.total_seconds()
        / 3600
    )

    df["week_block"] = (
        hours_from_start // (24 * 7)
    ).astype(int)

    weeks = np.array(
        sorted(
            df["week_block"]
            .unique()
        )
    )

    # --------------------------------------------------------
    # Stratified temporal spread:
    # randomly select train weeks, but ensure
    # train/test blocks are spread across full period.
    # --------------------------------------------------------

    rng_split = np.random.default_rng(
        SEED
    )

    train_weeks = []
    test_weeks = []

    # Divide sequential weeks into groups of 10.
    # In each group, ~7 become train and ~3 test.
    for start_idx in range(
        0,
        len(weeks),
        10
    ):

        group = weeks[
            start_idx:start_idx + 10
        ]

        if len(group) == 1:
            train_weeks.extend(
                group.tolist()
            )
            continue

        n_train = int(
            round(
                len(group)
                *
                TRAIN_WEEK_FRACTION
            )
        )

        n_train = min(
            max(n_train, 1),
            len(group) - 1
        )

        shuffled = (
            group.copy()
        )

        rng_split.shuffle(
            shuffled
        )

        train_weeks.extend(
            shuffled[:n_train]
            .tolist()
        )

        test_weeks.extend(
            shuffled[n_train:]
            .tolist()
        )

    train_weeks = sorted(
        train_weeks
    )

    test_weeks = sorted(
        test_weeks
    )

    df["split_blocked"] = np.where(
        df["week_block"].isin(
            train_weeks
        ),
        "train",
        "test"
    )

    print(
        f"\nTotal weekly blocks: "
        f"{len(weeks)}"
    )

    print(
        f"Train blocks: "
        f"{len(train_weeks)}"
    )

    print(
        f"Test blocks:  "
        f"{len(test_weeks)}"
    )

    print(
        f"Train observations: "
        f"{sum(df['split_blocked'] == 'train'):,}"
    )

    print(
        f"Test observations:  "
        f"{sum(df['split_blocked'] == 'test'):,}"
    )

    print(
        "\nTrain weeks:",
        train_weeks
    )

    print(
        "Test weeks: ",
        test_weeks
    )

    # --------------------------------------------------------
    # Preserve original electricity
    # --------------------------------------------------------

    df["electricity_original"] = (
        df["electricity"].astype(float)
    )

    df["electricity_injected"] = (
        df["electricity_original"].copy()
    )

    df["anomaly_label"] = 0
    df["anomaly_type"] = "normal"
    df["anomaly_event_id"] = ""

    train_mask = (
        df["split_blocked"] == "train"
    )

    test_mask = (
        df["split_blocked"] == "test"
    )

    train_electricity = (
        df.loc[
            train_mask,
            "electricity_original"
        ]
    )

    train_std = (
        train_electricity.std()
    )

    train_median = (
        train_electricity.median()
    )

    print(
        f"\nTrain electricity median: "
        f"{train_median:.3f}"
    )

    print(
        f"Train electricity std:    "
        f"{train_std:.3f}"
    )

    # --------------------------------------------------------
    # Test indices
    # --------------------------------------------------------

    test_indices = np.where(
        test_mask.values
    )[0]

    occupied = set()

    rng = np.random.default_rng(
        SEED + building_number
    )

    event_counter = 0

    # --------------------------------------------------------
    # Register helper
    # --------------------------------------------------------

    def register(
        interval,
        anomaly_type,
        values,
        code
    ):

        nonlocal_event_id = None

        df.loc[
            interval,
            "electricity_injected"
        ] = values

        df.loc[
            interval,
            "anomaly_label"
        ] = 1

        df.loc[
            interval,
            "anomaly_type"
        ] = anomaly_type

        return

    # ========================================================
    # A1 — SPIKES
    # ========================================================

    available = np.array(
        [
            i for i in test_indices
            if i not in occupied
        ]
    )

    spike_indices = rng.choice(
        available,
        size=10,
        replace=False
    )

    for idx in spike_indices:

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        magnitude = (
            rng.uniform(
                3.0,
                5.0
            )
            *
            train_std
        )

        value = (
            df.loc[
                idx,
                "electricity_original"
            ]
            +
            direction * magnitude
        )

        value = max(
            0.01,
            value
        )

        event_id = (
            f"{building}_SP_"
            f"{event_counter:02d}"
        )

        register(
            [idx],
            "spike",
            [value],
            "SP"
        )

        df.loc[
            idx,
            "anomaly_event_id"
        ] = event_id

        occupied.add(
            int(idx)
        )

    # ========================================================
    # A2 — PERSISTENT SHIFTS
    # ========================================================

    for _ in range(3):

        interval = find_interval(
            test_indices,
            12,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place shift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        shift = (
            direction
            *
            rng.uniform(
                1.5,
                2.5
            )
            *
            train_std
        )

        values = (
            df.loc[
                interval,
                "electricity_original"
            ].values
            +
            shift
        )

        values = np.maximum(
            values,
            0.01
        )

        event_id = (
            f"{building}_SH_"
            f"{event_counter:02d}"
        )

        register(
            interval,
            "persistent_shift",
            values,
            "SH"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A3 — GRADUAL DRIFT
    # ========================================================

    for _ in range(2):

        interval = find_interval(
            test_indices,
            24,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place drift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        final_shift = (
            direction
            *
            rng.uniform(
                2.0,
                3.0
            )
            *
            train_std
        )

        drift = np.linspace(
            0,
            final_shift,
            24
        )

        values = (
            df.loc[
                interval,
                "electricity_original"
            ].values
            +
            drift
        )

        values = np.maximum(
            values,
            0.01
        )

        event_id = (
            f"{building}_DR_"
            f"{event_counter:02d}"
        )

        register(
            interval,
            "gradual_drift",
            values,
            "DR"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A4 — STUCK MEASUREMENT
    # ========================================================

    for _ in range(2):

        interval = find_interval(
            test_indices,
            12,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place stuck event."
            )

        event_counter += 1

        stuck_value = (
            df.loc[
                interval[0],
                "electricity_original"
            ]
        )

        values = np.repeat(
            stuck_value,
            12
        )

        event_id = (
            f"{building}_ST_"
            f"{event_counter:02d}"
        )

        register(
            interval,
            "stuck",
            values,
            "ST"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A5 — CONTEXTUAL AFTER-HOURS
    # ========================================================

    after_hours = np.where(
        (
            test_mask.values
        )
        &
        (
            df["hour"]
            .isin(
                [0, 1, 2, 3, 4, 5]
            )
            .values
        )
    )[0]

    for _ in range(2):

        interval = find_interval(
            after_hours,
            6,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place contextual event."
            )

        event_counter += 1

        boost = (
            rng.uniform(
                1.5,
                2.5
            )
            *
            train_std
        )

        values = (
            df.loc[
                interval,
                "electricity_original"
            ].values
            +
            boost
        )

        event_id = (
            f"{building}_CX_"
            f"{event_counter:02d}"
        )

        register(
            interval,
            "contextual_after_hours",
            values,
            "CX"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # --------------------------------------------------------
    # Verify clean training
    # --------------------------------------------------------

    assert (
        df.loc[
            train_mask,
            "anomaly_label"
        ].sum()
        == 0
    )

    # ========================================================
    # BUILD MODEL FEATURES
    # ========================================================

    train = df[
        train_mask
    ].copy()

    test = df[
        test_mask
    ].copy()

    X_train_df = pd.DataFrame({
        "electricity":
            train[
                "electricity_original"
            ],

        "hour_sin":
            train["hour_sin"],

        "hour_cos":
            train["hour_cos"],

        "dow_sin":
            train["dow_sin"],

        "dow_cos":
            train["dow_cos"]
    })

    X_test_df = pd.DataFrame({
        "electricity":
            test[
                "electricity_injected"
            ],

        "hour_sin":
            test["hour_sin"],

        "hour_cos":
            test["hour_cos"],

        "dow_sin":
            test["dow_sin"],

        "dow_cos":
            test["dow_cos"]
    })

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

    print(
        f"\nInjected test anomalies: "
        f"{y_test.sum():,} / "
        f"{len(y_test):,} "
        f"({100*y_test.mean():.2f}%)"
    )

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    print(
        "\nTraining Isolation Forest..."
    )

    start = time.perf_counter()

    model_if = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )

    model_if.fit(
        X_train
    )

    if_train_time = (
        time.perf_counter()
        -
        start
    )

    if_train_score = (
        -model_if.decision_function(
            X_train
        )
    )

    start = time.perf_counter()

    if_test_score = (
        -model_if.decision_function(
            X_test
        )
    )

    if_inference = (
        time.perf_counter()
        -
        start
    )

    if_threshold = np.percentile(
        if_train_score,
        THRESHOLD_PERCENTILE
    )

    if_pred = (
        if_test_score
        >
        if_threshold
    ).astype(int)

    if_metrics = metrics(
        y_test,
        if_pred,
        if_test_score
    )

    # ========================================================
    # LOF
    # ========================================================

    print(
        "Training LOF..."
    )

    start = time.perf_counter()

    model_lof = LocalOutlierFactor(
        n_neighbors=35,
        novelty=True,
        contamination="auto",
        n_jobs=-1
    )

    model_lof.fit(
        X_train
    )

    lof_train_time = (
        time.perf_counter()
        -
        start
    )

    lof_train_score = (
        -model_lof.decision_function(
            X_train
        )
    )

    start = time.perf_counter()

    lof_test_score = (
        -model_lof.decision_function(
            X_test
        )
    )

    lof_inference = (
        time.perf_counter()
        -
        start
    )

    lof_threshold = np.percentile(
        lof_train_score,
        THRESHOLD_PERCENTILE
    )

    lof_pred = (
        lof_test_score
        >
        lof_threshold
    ).astype(int)

    lof_metrics = metrics(
        y_test,
        lof_pred,
        lof_test_score
    )

    # ========================================================
    # AUTOENCODER
    # ========================================================

    print(
        "Training Autoencoder..."
    )

    ae, ae_train_time = (
        train_autoencoder(
            X_train,
            building
        )
    )

    ae_train_score = (
        reconstruction_scores(
            ae,
            X_train
        )
    )

    start = time.perf_counter()

    ae_test_score = (
        reconstruction_scores(
            ae,
            X_test
        )
    )

    ae_inference = (
        time.perf_counter()
        -
        start
    )

    ae_threshold = np.percentile(
        ae_train_score,
        THRESHOLD_PERCENTILE
    )

    ae_pred = (
        ae_test_score
        >
        ae_threshold
    ).astype(int)

    ae_metrics = metrics(
        y_test,
        ae_pred,
        ae_test_score
    )

    # ========================================================
    # RESULTS
    # ========================================================

    model_info = [
        (
            "Isolation Forest",
            if_threshold,
            if_train_time,
            if_inference,
            if_metrics,
            if_test_score,
            if_pred
        ),
        (
            "LOF",
            lof_threshold,
            lof_train_time,
            lof_inference,
            lof_metrics,
            lof_test_score,
            lof_pred
        ),
        (
            "Autoencoder",
            ae_threshold,
            ae_train_time,
            ae_inference,
            ae_metrics,
            ae_test_score,
            ae_pred
        )
    ]

    for (
        name,
        threshold,
        train_time,
        inference_time,
        m,
        score,
        prediction
    ) in model_info:

        all_results.append({
            "building_id":
                building,

            "model":
                name,

            "threshold":
                threshold,

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

            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],

            "training_time_s":
                train_time,

            "inference_time_s":
                inference_time
        })

        print(
            f"\n{name}:"
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
            f"  FPR       = "
            f"{m['false_positive_rate']:.4f}"
        )

    # ========================================================
    # ANOMALY-TYPE RECALL
    # ========================================================

    test_types = (
        test["anomaly_type"]
        .values
    )

    anomaly_types = [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours"
    ]

    predictions_dict = {
        "Isolation Forest":
            if_pred,
        "LOF":
            lof_pred,
        "Autoencoder":
            ae_pred
    }

    for model_name, prediction in (
        predictions_dict.items()
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

            recall = (
                float(
                    prediction[
                        mask
                    ].mean()
                )
                if n > 0
                else np.nan
            )

            all_type_results.append({
                "building_id":
                    building,

                "model":
                    model_name,

                "anomaly_type":
                    anomaly_type,

                "n":
                    n,

                "recall":
                    recall
            })

    # ========================================================
    # EVENT MANIFEST
    # ========================================================

    events = (
        test[
            test["anomaly_label"] == 1
        ]
        .groupby(
            [
                "anomaly_event_id",
                "anomaly_type"
            ]
        )
        .agg(
            start=(
                "timestamp",
                "min"
            ),
            end=(
                "timestamp",
                "max"
            ),
            duration_hours=(
                "timestamp",
                "size"
            )
        )
        .reset_index()
    )

    events.insert(
        0,
        "building_id",
        building
    )

    all_events.append(
        events
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            test["timestamp"].values,

        "week_block":
            test[
                "week_block"
            ].values,

        "anomaly_label":
            y_test,

        "anomaly_type":
            test[
                "anomaly_type"
            ].values,

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
            if_test_score,

        "if_prediction":
            if_pred,

        "lof_score":
            lof_test_score,

        "lof_prediction":
            lof_pred,

        "ae_score":
            ae_test_score,

        "ae_prediction":
            ae_pred
    })

    all_predictions.append(
        prediction_df
    )

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    all_split_summaries.append({
        "building_id":
            building,

        "train_blocks":
            len(train_weeks),

        "test_blocks":
            len(test_weeks),

        "train_observations":
            len(train),

        "test_observations":
            len(test),

        "test_anomalies":
            int(y_test.sum()),

        "test_anomaly_rate":
            y_test.mean(),

        "train_start":
            train["timestamp"].min(),

        "train_end":
            train["timestamp"].max(),

        "test_start":
            test["timestamp"].min(),

        "test_end":
            test["timestamp"].max()
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    TABLES /
    "corrected_model_performance.csv",
    index=False
)

print("\n" + "=" * 100)
print("CORRECTED MODEL PERFORMANCE")
print("=" * 100)

cols = [
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
    cols
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

mean_results.to_csv(
    TABLES /
    "corrected_mean_performance.csv",
    index=False
)

print("\n" + "=" * 100)
print("MEAN PERFORMANCE ACROSS THREE BUILDINGS")
print("=" * 100)

mean_display = (
    mean_results.copy()
)

numeric = (
    mean_display
    .select_dtypes(
        include=np.number
    )
    .columns
)

mean_display[numeric] = (
    mean_display[numeric]
    .round(4)
)

print(
    mean_display.to_string(
        index=False
    )
)

# ============================================================
# TYPE RESULTS
# ============================================================

type_results = pd.DataFrame(
    all_type_results
)

type_results.to_csv(
    TABLES /
    "corrected_anomaly_type_recall.csv",
    index=False
)

type_mean = (
    type_results
    .groupby(
        [
            "model",
            "anomaly_type"
        ],
        as_index=False
    )["recall"]
    .mean()
)

type_mean.to_csv(
    TABLES /
    "corrected_anomaly_type_mean_recall.csv",
    index=False
)

print("\n" + "=" * 100)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 100)

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
# SAVE PREDICTIONS / EVENTS / SPLITS
# ============================================================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions.to_csv(
    TABLES /
    "corrected_all_test_predictions.csv",
    index=False
)

events = pd.concat(
    all_events,
    ignore_index=True
)

events.to_csv(
    TABLES /
    "corrected_anomaly_event_manifest.csv",
    index=False
)

split_summary = pd.DataFrame(
    all_split_summaries
)

split_summary.to_csv(
    TABLES /
    "corrected_split_summary.csv",
    index=False
)

print("\n" + "=" * 100)
print("STEP 7 COMPLETE")
print("=" * 100)

print(
    "\nOriginal Step-6 results were NOT overwritten."
)

print(
    "Corrected results have been saved "
    "with the 'corrected_' prefix."
)