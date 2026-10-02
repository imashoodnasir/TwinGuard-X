from pathlib import Path
import time
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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

SELECTION_PATH = TABLES / "selected_three_public_buildings.csv"

SEED = 42

# Exact Step-7 blocked split
TEST_WEEKS = [1, 8, 9, 10, 11, 13, 21, 25]

# Threshold derived ONLY from normal training scores
THRESHOLD_PERCENTILE = 95

EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.001

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("EIRT 2026 — TWINGUARD-X")
print("Explainable Operational Digital Twin for Public-Building Anomaly Detection")
print("=" * 100)

print(f"\nDevice: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Epochs: {EPOCHS}")
print(f"Test weeks: {TEST_WEEKS}")


# ============================================================
# TWINGUARD-X ARCHITECTURE
# ============================================================

class TwinGuardX(nn.Module):

    def __init__(
        self,
        context_dim,
        operational_dim
    ):

        super().__init__()

        # ----------------------------------------------------
        # Context encoder
        # Weather + cyclic temporal context
        # ----------------------------------------------------

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Operational-state encoder
        # Historical electricity behaviour
        # ----------------------------------------------------

        self.operational_encoder = nn.Sequential(
            nn.Linear(operational_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Digital-twin prediction head
        # Predict expected electricity
        # ----------------------------------------------------

        self.twin_head = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        # ----------------------------------------------------
        # Context-dependent anomaly gate
        #
        # Three outputs:
        # 0 = twin residual importance
        # 1 = residual-dynamics importance
        # 2 = persistence importance
        # ----------------------------------------------------

        self.anomaly_gate = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        context,
        operational
    ):

        z_context = self.context_encoder(
            context
        )

        z_operational = self.operational_encoder(
            operational
        )

        joint = torch.cat(
            [
                z_context,
                z_operational
            ],
            dim=1
        )

        fused = self.fusion(
            joint
        )

        prediction = self.twin_head(
            fused
        )

        gates = self.anomaly_gate(
            joint
        )

        return (
            prediction,
            gates,
            z_context,
            z_operational
        )


# ============================================================
# INTERVAL HELPER
# Same principle as Step 7
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

    starts = candidates.copy()

    rng.shuffle(starts)

    for start in starts:

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

        return np.asarray(
            interval,
            dtype=int
        )

    return None


# ============================================================
# REPRODUCE STEP-7 ANOMALIES
# ============================================================

def inject_anomalies(
    df,
    building_number,
    train_mask,
    test_mask
):

    df = df.copy()

    df["electricity_original"] = (
        df["electricity"].astype(float)
    )

    df["electricity_injected"] = (
        df["electricity_original"].copy()
    )

    df["anomaly_label"] = 0
    df["anomaly_type"] = "normal"
    df["anomaly_event_id"] = ""

    train_values = df.loc[
        train_mask,
        "electricity_original"
    ]

    train_std = train_values.std()

    test_indices = np.where(
        test_mask.values
    )[0]

    rng = np.random.default_rng(
        SEED + building_number
    )

    occupied = set()
    event_counter = 0

    # --------------------------------------------------------
    # 1. SPIKES
    # --------------------------------------------------------

    available = np.asarray([
        i for i in test_indices
        if i not in occupied
    ])

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
            rng.uniform(3.0, 5.0)
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

        df.loc[
            idx,
            "electricity_injected"
        ] = value

        df.loc[
            idx,
            "anomaly_label"
        ] = 1

        df.loc[
            idx,
            "anomaly_type"
        ] = "spike"

        df.loc[
            idx,
            "anomaly_event_id"
        ] = (
            f"SP_{event_counter:02d}"
        )

        occupied.add(
            int(idx)
        )

    # --------------------------------------------------------
    # 2. PERSISTENT SHIFT
    # --------------------------------------------------------

    for _ in range(3):

        interval = find_interval(
            test_indices,
            12,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Cannot place persistent shift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        shift = (
            direction
            *
            rng.uniform(1.5, 2.5)
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
        ] = "persistent_shift"

        df.loc[
            interval,
            "anomaly_event_id"
        ] = (
            f"SH_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # --------------------------------------------------------
    # 3. GRADUAL DRIFT
    # --------------------------------------------------------

    for _ in range(2):

        interval = find_interval(
            test_indices,
            24,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Cannot place gradual drift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        final_shift = (
            direction
            *
            rng.uniform(2.0, 3.0)
            *
            train_std
        )

        drift = np.linspace(
            0,
            final_shift,
            len(interval)
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
        ] = "gradual_drift"

        df.loc[
            interval,
            "anomaly_event_id"
        ] = (
            f"DR_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # --------------------------------------------------------
    # 4. STUCK SENSOR
    # --------------------------------------------------------

    for _ in range(2):

        interval = find_interval(
            test_indices,
            12,
            occupied,
            rng
        )

        if interval is None:
            raise RuntimeError(
                "Cannot place stuck event."
            )

        event_counter += 1

        stuck_value = df.loc[
            interval[0],
            "electricity_original"
        ]

        df.loc[
            interval,
            "electricity_injected"
        ] = stuck_value

        df.loc[
            interval,
            "anomaly_label"
        ] = 1

        df.loc[
            interval,
            "anomaly_type"
        ] = "stuck"

        df.loc[
            interval,
            "anomaly_event_id"
        ] = (
            f"ST_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # --------------------------------------------------------
    # 5. CONTEXTUAL AFTER-HOURS
    # --------------------------------------------------------

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
                "Cannot place contextual event."
            )

        event_counter += 1

        boost = (
            rng.uniform(1.5, 2.5)
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
        ] = "contextual_after_hours"

        df.loc[
            interval,
            "anomaly_event_id"
        ] = (
            f"CX_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    return df


# ============================================================
# CREATE OPERATIONAL HISTORY
# ============================================================

def create_operational_features(
    df,
    electricity_column
):

    e = df[
        electricity_column
    ].astype(float)

    features = pd.DataFrame(
        index=df.index
    )

    features["lag_1"] = e.shift(1)
    features["lag_2"] = e.shift(2)
    features["lag_3"] = e.shift(3)
    features["lag_24"] = e.shift(24)

    # IMPORTANT:
    # rolling statistics exclude current observation
    previous = e.shift(1)

    features["mean_6"] = (
        previous
        .rolling(
            6,
            min_periods=1
        )
        .mean()
    )

    features["std_6"] = (
        previous
        .rolling(
            6,
            min_periods=2
        )
        .std()
    )

    features["mean_24"] = (
        previous
        .rolling(
            24,
            min_periods=1
        )
        .mean()
    )

    features["std_24"] = (
        previous
        .rolling(
            24,
            min_periods=2
        )
        .std()
    )

    return features


# ============================================================
# CONTEXT FEATURES
# ============================================================

CONTEXT_COLUMNS = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
]

OPERATIONAL_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "mean_6",
    "std_6",
    "mean_24",
    "std_24"
]


# ============================================================
# TRAIN DIGITAL TWIN
# ============================================================

def train_twinguard(
    X_context,
    X_operational,
    y,
    building
):

    model = TwinGuardX(
        context_dim=X_context.shape[1],
        operational_dim=X_operational.shape[1]
    ).to(DEVICE)

    context_tensor = torch.tensor(
        X_context,
        dtype=torch.float32
    )

    operational_tensor = torch.tensor(
        X_operational,
        dtype=torch.float32
    )

    target_tensor = torch.tensor(
        y.reshape(-1, 1),
        dtype=torch.float32
    )

    dataset = TensorDataset(
        context_tensor,
        operational_tensor,
        target_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    mse = nn.MSELoss()

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        epoch_losses = []

        for (
            context_batch,
            operational_batch,
            target_batch
        ) in loader:

            context_batch = (
                context_batch.to(DEVICE)
            )

            operational_batch = (
                operational_batch.to(DEVICE)
            )

            target_batch = (
                target_batch.to(DEVICE)
            )

            optimizer.zero_grad()

            prediction, gates, _, _ = model(
                context_batch,
                operational_batch
            )

            # Digital-twin reconstruction/prediction loss
            loss_prediction = mse(
                prediction,
                target_batch
            )

            # Encourage non-collapsed gate usage.
            # The gate remains context dependent.
            gate_mean = gates.mean(
                dim=0
            )

            uniform = torch.full_like(
                gate_mean,
                1.0 / 3.0
            )

            gate_regularization = torch.mean(
                (
                    gate_mean
                    -
                    uniform
                ) ** 2
            )

            loss = (
                loss_prediction
                +
                0.01
                *
                gate_regularization
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            epoch_losses.append(
                loss.item()
            )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"  Epoch "
                f"{epoch + 1:03d}/{EPOCHS} "
                f"| loss="
                f"{np.mean(epoch_losses):.6f}"
            )

    training_time = (
        time.perf_counter()
        -
        start_time
    )

    torch.save(
        model.state_dict(),
        MODELS /
        f"{building}_TwinGuardX.pt"
    )

    return (
        model,
        training_time
    )


# ============================================================
# DIGITAL-TWIN PREDICTION
# ============================================================

def predict_twin(
    model,
    context,
    operational
):

    model.eval()

    c = torch.tensor(
        context,
        dtype=torch.float32
    ).to(DEVICE)

    o = torch.tensor(
        operational,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        prediction, gates, _, _ = model(
            c,
            o
        )

    return (
        prediction
        .cpu()
        .numpy()
        .ravel(),

        gates
        .cpu()
        .numpy()
    )


# ============================================================
# ROBUST NORMALIZATION
# ============================================================

def robust_parameters(x):

    x = np.asarray(
        x,
        dtype=float
    )

    median = np.median(x)

    mad = np.median(
        np.abs(
            x - median
        )
    )

    scale = (
        1.4826 * mad
    )

    if scale < 1e-8:

        scale = np.std(x)

    if scale < 1e-8:

        scale = 1.0

    return median, scale


def positive_robust_score(
    x,
    median,
    scale
):

    z = (
        np.asarray(x)
        -
        median
    ) / scale

    # Only unusually high deviation is anomalous
    return np.maximum(
        z,
        0.0
    )


# ============================================================
# PERFORMANCE
# ============================================================

def calculate_metrics(
    y,
    pred,
    score
):

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
all_type_results = []
all_gate_results = []


# ============================================================
# BUILDING LOOP
# ============================================================

for building_number, building in enumerate(
    buildings,
    start=1
):

    print("\n" + "=" * 100)
    print(f"TWINGUARD-X: {building}")
    print("=" * 100)

    path = (
        PROCESSED /
        f"{building}_experiment.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Reconstruct exact Step-7 week blocks
    # --------------------------------------------------------

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
        hours_from_start
        // (24 * 7)
    ).astype(int)

    test_mask = (
        df["week_block"]
        .isin(TEST_WEEKS)
    )

    train_mask = ~test_mask

    print(
        f"\nTrain observations: "
        f"{train_mask.sum():,}"
    )

    print(
        f"Test observations:  "
        f"{test_mask.sum():,}"
    )

    # --------------------------------------------------------
    # Reproduce exactly same anomaly protocol
    # --------------------------------------------------------

    df = inject_anomalies(
        df,
        building_number,
        train_mask,
        test_mask
    )

    print(
        f"Injected anomalies: "
        f"{df.loc[test_mask, 'anomaly_label'].sum():,}"
    )

    # --------------------------------------------------------
    # Context data
    # --------------------------------------------------------

    context = (
        df[CONTEXT_COLUMNS]
        .copy()
    )

    # Cloud coverage intentionally excluded.

    # --------------------------------------------------------
    # Clean operational history for training
    # --------------------------------------------------------

    operational_clean = (
        create_operational_features(
            df,
            "electricity_original"
        )
    )

    # --------------------------------------------------------
    # Injected operational history for test
    # --------------------------------------------------------

    operational_injected = (
        create_operational_features(
            df,
            "electricity_injected"
        )
    )

    # --------------------------------------------------------
    # Valid rows
    # Need 24-hour history
    # --------------------------------------------------------

    valid_clean = (
        operational_clean
        .notna()
        .all(axis=1)
    )

    valid_injected = (
        operational_injected
        .notna()
        .all(axis=1)
    )

    valid_context = (
        context
        .notna()
        .all(axis=1)
    )

    valid = (
        valid_clean
        &
        valid_injected
        &
        valid_context
    )

    train_valid = (
        train_mask
        &
        valid
    )

    test_valid = (
        test_mask
        &
        valid
    )

    # --------------------------------------------------------
    # Training matrices
    # --------------------------------------------------------

    context_train_df = (
        context.loc[
            train_valid
        ].copy()
    )

    operational_train_df = (
        operational_clean.loc[
            train_valid
        ].copy()
    )

    target_train = (
        df.loc[
            train_valid,
            "electricity_original"
        ]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # Test matrices
    # --------------------------------------------------------

    context_test_df = (
        context.loc[
            test_valid
        ].copy()
    )

    operational_test_df = (
        operational_injected.loc[
            test_valid
        ].copy()
    )

    target_test_observed = (
        df.loc[
            test_valid,
            "electricity_injected"
        ]
        .astype(float)
        .values
    )

    target_test_original = (
        df.loc[
            test_valid,
            "electricity_original"
        ]
        .astype(float)
        .values
    )

    y_test = (
        df.loc[
            test_valid,
            "anomaly_label"
        ]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Scaling
    # Fit ONLY training data
    # --------------------------------------------------------

    context_scaler = StandardScaler()

    operational_scaler = StandardScaler()

    target_scaler = StandardScaler()

    Xc_train = (
        context_scaler
        .fit_transform(
            context_train_df
        )
    )

    Xo_train = (
        operational_scaler
        .fit_transform(
            operational_train_df
        )
    )

    y_train_scaled = (
        target_scaler
        .fit_transform(
            target_train.reshape(-1, 1)
        )
        .ravel()
    )

    Xc_test = (
        context_scaler
        .transform(
            context_test_df
        )
    )

    Xo_test = (
        operational_scaler
        .transform(
            operational_test_df
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining TwinGuard-X...")

    model, training_time = (
        train_twinguard(
            Xc_train,
            Xo_train,
            y_train_scaled,
            building
        )
    )

    # --------------------------------------------------------
    # TRAIN predictions
    # --------------------------------------------------------

    train_pred_scaled, train_gates = (
        predict_twin(
            model,
            Xc_train,
            Xo_train
        )
    )

    train_pred = (
        target_scaler
        .inverse_transform(
            train_pred_scaled
            .reshape(-1, 1)
        )
        .ravel()
    )

    # --------------------------------------------------------
    # TEST predictions
    # --------------------------------------------------------

    start = time.perf_counter()

    test_pred_scaled, test_gates = (
        predict_twin(
            model,
            Xc_test,
            Xo_test
        )
    )

    inference_time = (
        time.perf_counter()
        -
        start
    )

    test_pred = (
        target_scaler
        .inverse_transform(
            test_pred_scaled
            .reshape(-1, 1)
        )
        .ravel()
    )

    # ========================================================
    # BRANCH 1 — TWIN RESIDUAL
    # ========================================================

    train_residual = np.abs(
        target_train
        -
        train_pred
    )

    test_residual = np.abs(
        target_test_observed
        -
        test_pred
    )

    # ========================================================
    # BRANCH 2 — RESIDUAL DYNAMICS
    # ========================================================

    train_residual_delta = np.abs(
        np.diff(
            train_residual,
            prepend=train_residual[0]
        )
    )

    test_residual_delta = np.abs(
        np.diff(
            test_residual,
            prepend=test_residual[0]
        )
    )

    # ========================================================
    # BRANCH 3 — PERSISTENCE
    #
    # Stuck measurement:
    # current reading changes abnormally little relative
    # to recent operating variability.
    # ========================================================

    train_obs = target_train

    test_obs = target_test_observed

    train_change = np.abs(
        np.diff(
            train_obs,
            prepend=train_obs[0]
        )
    )

    test_change = np.abs(
        np.diff(
            test_obs,
            prepend=test_obs[0]
        )
    )

    # Invert change:
    # smaller change -> larger persistence evidence.
    persistence_reference = np.percentile(
        train_change,
        75
    )

    train_persistence = np.maximum(
        persistence_reference
        -
        train_change,
        0
    )

    test_persistence = np.maximum(
        persistence_reference
        -
        test_change,
        0
    )

    # ========================================================
    # ROBUST NORMALIZATION FROM TRAIN ONLY
    # ========================================================

    r_med, r_scale = (
        robust_parameters(
            train_residual
        )
    )

    d_med, d_scale = (
        robust_parameters(
            train_residual_delta
        )
    )

    p_med, p_scale = (
        robust_parameters(
            train_persistence
        )
    )

    train_r = positive_robust_score(
        train_residual,
        r_med,
        r_scale
    )

    test_r = positive_robust_score(
        test_residual,
        r_med,
        r_scale
    )

    train_d = positive_robust_score(
        train_residual_delta,
        d_med,
        d_scale
    )

    test_d = positive_robust_score(
        test_residual_delta,
        d_med,
        d_scale
    )

    train_p = positive_robust_score(
        train_persistence,
        p_med,
        p_scale
    )

    test_p = positive_robust_score(
        test_persistence,
        p_med,
        p_scale
    )

    # ========================================================
    # CONTEXT-ADAPTIVE GATED FUSION
    # ========================================================

    train_score = (
        train_gates[:, 0] * train_r
        +
        train_gates[:, 1] * train_d
        +
        train_gates[:, 2] * train_p
    )

    test_score = (
        test_gates[:, 0] * test_r
        +
        test_gates[:, 1] * test_d
        +
        test_gates[:, 2] * test_p
    )

    threshold = np.percentile(
        train_score,
        THRESHOLD_PERCENTILE
    )

    prediction = (
        test_score
        >
        threshold
    ).astype(int)

    m = calculate_metrics(
        y_test,
        prediction,
        test_score
    )

    # ========================================================
    # DIGITAL TWIN PREDICTION QUALITY
    # Evaluate on ORIGINAL uncorrupted test electricity
    # ========================================================

    mae = np.mean(
        np.abs(
            target_test_original
            -
            test_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                target_test_original
                -
                test_pred
            ) ** 2
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    all_results.append({
        "building_id":
            building,

        "model":
            "TwinGuard-X",

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
            m["false_positive_rate"],

        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"],

        "threshold":
            threshold,

        "twin_mae":
            mae,

        "twin_rmse":
            rmse,

        "training_time_s":
            training_time,

        "inference_time_s":
            inference_time
    })

    print("\nTwinGuard-X performance:")

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

    print(
        f"  Twin MAE  = "
        f"{mae:.3f}"
    )

    print(
        f"  Twin RMSE = "
        f"{rmse:.3f}"
    )

    # ========================================================
    # GATE INTERPRETABILITY
    # ========================================================

    print("\nMean anomaly-gate weights:")

    print(
        f"  Residual branch:    "
        f"{test_gates[:,0].mean():.4f}"
    )

    print(
        f"  Dynamics branch:    "
        f"{test_gates[:,1].mean():.4f}"
    )

    print(
        f"  Persistence branch: "
        f"{test_gates[:,2].mean():.4f}"
    )

    all_gate_results.append({
        "building_id":
            building,

        "residual_gate":
            test_gates[:, 0].mean(),

        "dynamics_gate":
            test_gates[:, 1].mean(),

        "persistence_gate":
            test_gates[:, 2].mean()
    })

    # ========================================================
    # ANOMALY-TYPE RECALL
    # ========================================================

    test_types = (
        df.loc[
            test_valid,
            "anomaly_type"
        ]
        .values
    )

    anomaly_types = [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours"
    ]

    print("\nRecall by anomaly type:")

    for anomaly_type in anomaly_types:

        mask = (
            test_types
            ==
            anomaly_type
        )

        n = int(
            mask.sum()
        )

        recall = (
            prediction[mask].mean()
            if n > 0
            else np.nan
        )

        print(
            f"  {anomaly_type:25s} "
            f"{recall:.4f} "
            f"(n={n})"
        )

        all_type_results.append({
            "building_id":
                building,

            "model":
                "TwinGuard-X",

            "anomaly_type":
                anomaly_type,

            "n":
                n,

            "recall":
                recall
        })

    # ========================================================
    # SAVE PER-HOUR EXPLANATIONS
    # ========================================================

    test_indices_valid = (
        df.index[
            test_valid
        ]
    )

    explanations = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            df.loc[
                test_indices_valid,
                "timestamp"
            ].values,

        "anomaly_label":
            y_test,

        "anomaly_type":
            test_types,

        "electricity_original":
            target_test_original,

        "electricity_observed":
            target_test_observed,

        "digital_twin_expected":
            test_pred,

        "twin_residual":
            test_residual,

        "residual_dynamics":
            test_residual_delta,

        "persistence_signal":
            test_persistence,

        "residual_gate":
            test_gates[:, 0],

        "dynamics_gate":
            test_gates[:, 1],

        "persistence_gate":
            test_gates[:, 2],

        "anomaly_score":
            test_score,

        "threshold":
            threshold,

        "prediction":
            prediction
    })

    all_predictions.append(
        explanations
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    TABLES /
    "twinguard_x_performance.csv",
    index=False
)

print("\n" + "=" * 100)
print("TWINGUARD-X PERFORMANCE")
print("=" * 100)

display_cols = [
    "building_id",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate",
    "twin_mae",
    "twin_rmse"
]

display = (
    results[
        display_cols
    ]
    .copy()
)

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
# MEAN RESULTS
# ============================================================

mean_metrics = (
    results[
        [
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "specificity",
            "false_positive_rate"
        ]
    ]
    .mean()
)

print("\n" + "=" * 100)
print("MEAN TWINGUARD-X PERFORMANCE")
print("=" * 100)

for metric, value in (
    mean_metrics.items()
):

    print(
        f"{metric:25s}: "
        f"{value:.4f}"
    )


# ============================================================
# ANOMALY-TYPE RESULTS
# ============================================================

type_results = pd.DataFrame(
    all_type_results
)

type_results.to_csv(
    TABLES /
    "twinguard_x_anomaly_type_recall.csv",
    index=False
)

type_mean = (
    type_results
    .groupby(
        "anomaly_type"
    )["recall"]
    .mean()
)

print("\n" + "=" * 100)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 100)

print(
    type_mean
    .round(4)
    .to_string()
)


# ============================================================
# GATE RESULTS
# ============================================================

gate_results = pd.DataFrame(
    all_gate_results
)

gate_results.to_csv(
    TABLES /
    "twinguard_x_gate_weights.csv",
    index=False
)


# ============================================================
# EXPLANATION OUTPUT
# ============================================================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions.to_csv(
    TABLES /
    "twinguard_x_explanations.csv",
    index=False
)


# ============================================================
# COMPARE WITH STEP 7 BASELINES
# ============================================================

baseline_path = (
    TABLES /
    "corrected_mean_performance.csv"
)

if baseline_path.exists():

    baseline = pd.read_csv(
        baseline_path
    )

    comparison_rows = []

    for _, row in baseline.iterrows():

        comparison_rows.append({
            "model":
                row["model"],

            "precision":
                row["precision"],

            "recall":
                row["recall"],

            "f1":
                row["f1"],

            "auroc":
                row["auroc"],

            "auprc":
                row["auprc"],

            "false_positive_rate":
                row[
                    "false_positive_rate"
                ]
        })

    comparison_rows.append({
        "model":
            "TwinGuard-X",

        "precision":
            mean_metrics["precision"],

        "recall":
            mean_metrics["recall"],

        "f1":
            mean_metrics["f1"],

        "auroc":
            mean_metrics["auroc"],

        "auprc":
            mean_metrics["auprc"],

        "false_positive_rate":
            mean_metrics[
                "false_positive_rate"
            ]
    })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        TABLES /
        "final_model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("BASELINE VS TWINGUARD-X")
    print("=" * 100)

    comp = comparison.copy()

    numeric = (
        comp
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    comp[numeric] = (
        comp[numeric]
        .round(4)
    )

    print(
        comp.to_string(
            index=False
        )
    )


print("\n" + "=" * 100)
print("STEP 8 COMPLETE")
print("=" * 100)

print(
    "\nOutputs:"
)

print(
    "  twinguard_x_performance.csv"
)

print(
    "  twinguard_x_anomaly_type_recall.csv"
)

print(
    "  twinguard_x_gate_weights.csv"
)

print(
    "  twinguard_x_explanations.csv"
)

print(
    "  final_model_comparison.csv"
)