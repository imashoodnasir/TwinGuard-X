from pathlib import Path
import itertools
import json
import random
import time
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
    confusion_matrix,
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
PROTOCOL_PATH = TABLES / "locked_protocol.json"

SEED = 45
HISTORY = 24
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Coarse-to-fine simplex search
COARSE_STEP = 0.10
FINE_STEP = 0.025

# Threshold candidates are percentiles of TRAINING normal scores.
THRESHOLD_PERCENTILES = np.arange(
    90.0,
    99.51,
    0.5
)

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

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("EIRT 2026 — TWINGUARD-X v3 DEVELOPMENT")
print("Validation-Calibrated Interpretable Reliability Fusion")
print("=" * 110)

print(f"\nDevice: {DEVICE}")
print(f"Seed: {SEED}")
print(f"History: {HISTORY} hours")
print(f"Epochs: {EPOCHS}")

print("\nIMPORTANT:")
print("This script uses TRAIN + VALIDATION only.")
print("The locked FINAL TEST is NOT evaluated.")


# ============================================================
# LOAD LOCKED PROTOCOL
# ============================================================

with open(
    PROTOCOL_PATH,
    "r",
    encoding="utf-8"
) as f:
    protocol = json.load(f)

if protocol.get("status") != "LOCKED":
    raise RuntimeError(
        "Experimental protocol is not marked LOCKED."
    )

TRAIN_WEEKS = protocol["training_weeks"]
VALIDATION_WEEKS = protocol["validation_weeks"]
FINAL_TEST_WEEKS = protocol["final_test_weeks"]

print("\nTraining weeks:")
print(TRAIN_WEEKS)

print("\nValidation weeks:")
print(VALIDATION_WEEKS)

print("\nLocked final-test weeks:")
print(FINAL_TEST_WEEKS)


# ============================================================
# DIGITAL TWIN
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
            1,
            8,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            8,
            16,
            kernel_size=3,
            dilation=2,
            padding=2,
        )

        self.conv3 = nn.Conv1d(
            16,
            8,
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

        return (
            expected,
            context_state,
            operational_state,
        )


# ============================================================
# BUILD SEQUENTIAL SAMPLES
# ============================================================

def build_samples(
    df,
    electricity_column,
    allowed_split,
):

    samples = []

    electricity = (
        df[electricity_column]
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
            allowed_split
        ):
            continue

        # Prevent history from crossing into another split.
        history_splits = (
            df.loc[
                i-HISTORY:i-1,
                "experimental_split"
            ]
            .values
        )

        if not np.all(
            history_splits
            ==
            allowed_split
        ):
            continue

        history = electricity[
            i-HISTORY:i
        ]

        if not np.all(
            np.isfinite(history)
        ):
            continue

        context = (
            df.loc[
                i,
                CONTEXT_COLUMNS
            ]
            .astype(float)
            .values
        )

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

    return samples


# ============================================================
# ROBUST SCALING
# ============================================================

def robust_fit(X):

    X = np.asarray(
        X,
        dtype=float
    )

    median = np.median(
        X,
        axis=0
    )

    mad = np.median(
        np.abs(
            X - median
        ),
        axis=0
    )

    scale = 1.4826 * mad

    std = np.std(
        X,
        axis=0
    )

    scale = np.where(
        scale < 1e-8,
        std,
        scale
    )

    scale = np.where(
        scale < 1e-8,
        1.0,
        scale
    )

    return median, scale


def robust_transform(
    X,
    median,
    scale,
):

    return (
        np.asarray(
            X,
            dtype=float
        )
        -
        median
    ) / scale


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
        +
        1e-6
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
# CONTEXT-CONDITIONED RESIDUAL NORMALIZATION
# ============================================================

def build_context_bins(
    df,
    sample_indices,
    residuals,
):

    """
    Estimate normal residual scale by:
        hour group × temperature quartile.

    This remains fully interpretable.
    """

    temp = (
        df.loc[
            sample_indices,
            "airTemperature"
        ]
        .astype(float)
        .values
    )

    hours = (
        df.loc[
            sample_indices,
            "hour"
        ]
        .astype(int)
        .values
    )

    quartiles = np.quantile(
        temp,
        [0.25, 0.50, 0.75]
    )

    global_median = np.median(
        residuals
    )

    global_mad = (
        1.4826
        *
        np.median(
            np.abs(
                residuals
                -
                global_median
            )
        )
    )

    if global_mad < 1e-8:
        global_mad = (
            np.std(residuals)
            +
            1e-6
        )

    bins = {}

    for hour_group in range(4):

        h_start = (
            hour_group
            *
            6
        )

        h_end = (
            h_start
            +
            6
        )

        hour_mask = (
            (hours >= h_start)
            &
            (hours < h_end)
        )

        temp_group = np.digitize(
            temp,
            quartiles
        )

        for tg in range(4):

            mask = (
                hour_mask
                &
                (
                    temp_group
                    ==
                    tg
                )
            )

            values = residuals[
                mask
            ]

            if len(values) >= 20:

                med = np.median(
                    values
                )

                mad = (
                    1.4826
                    *
                    np.median(
                        np.abs(
                            values
                            -
                            med
                        )
                    )
                )

                if mad < 1e-8:
                    mad = global_mad

                bins[
                    (
                        hour_group,
                        tg,
                    )
                ] = (
                    med,
                    mad,
                )

    return {
        "quartiles":
            quartiles,

        "bins":
            bins,

        "global_median":
            global_median,

        "global_mad":
            global_mad,
    }


def contextualize_residuals(
    df,
    sample_indices,
    residuals,
    context_model,
):

    quartiles = (
        context_model[
            "quartiles"
        ]
    )

    bins = (
        context_model[
            "bins"
        ]
    )

    global_med = (
        context_model[
            "global_median"
        ]
    )

    global_mad = (
        context_model[
            "global_mad"
        ]
    )

    output = []

    for idx, residual in zip(
        sample_indices,
        residuals,
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
            hour
            //
            6
        )

        temp_group = int(
            np.digitize(
                temp,
                quartiles
            )
        )

        med, mad = bins.get(
            (
                hour_group,
                temp_group,
            ),
            (
                global_med,
                global_mad,
            ),
        )

        score = (
            residual
            -
            med
        ) / (
            mad
            +
            1e-6
        )

        output.append(
            max(
                0.0,
                score
            )
        )

    return np.asarray(
        output
    )


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
# SIMPLEX GENERATION
# ============================================================

def simplex_weights(step):

    units = int(
        round(
            1.0 / step
        )
    )

    weights = []

    for a in range(
        units + 1
    ):

        for b in range(
            units - a + 1
        ):

            for c in range(
                units - a - b + 1
            ):

                for d in range(
                    units - a - b - c + 1
                ):

                    e = (
                        units
                        -
                        a
                        -
                        b
                        -
                        c
                        -
                        d
                    )

                    weights.append(
                        np.asarray(
                            [
                                a,
                                b,
                                c,
                                d,
                                e,
                            ],
                            dtype=float,
                        )
                        /
                        units
                    )

    return weights


def fine_simplex_weights(
    best,
    radius=0.10,
    step=0.025,
):

    candidates = []

    values = np.arange(
        -radius,
        radius + 1e-9,
        step,
    )

    for delta in itertools.product(
        values,
        repeat=4,
    ):

        candidate = (
            best.copy()
        )

        candidate[:4] += (
            np.asarray(delta)
        )

        candidate[4] = (
            1.0
            -
            candidate[:4].sum()
        )

        if np.any(
            candidate < -1e-9
        ):
            continue

        if np.any(
            candidate > 1.0 + 1e-9
        ):
            continue

        candidate = np.clip(
            candidate,
            0,
            1,
        )

        if not np.isclose(
            candidate.sum(),
            1.0,
            atol=1e-6,
        ):
            continue

        candidates.append(
            candidate
        )

    return candidates


# ============================================================
# CALIBRATION SEARCH
# ============================================================

def search_calibration(
    train_descriptors,
    val_descriptors,
    val_labels,
):

    best = None

    coarse = simplex_weights(
        COARSE_STEP
    )

    print(
        f"  Coarse reliability candidates: "
        f"{len(coarse):,}"
    )

    def evaluate_candidate(
        weights
    ):

        train_score = (
            train_descriptors
            @
            weights
        )

        val_score = (
            val_descriptors
            @
            weights
        )

        candidate_best = None

        for percentile in (
            THRESHOLD_PERCENTILES
        ):

            threshold = np.percentile(
                train_score,
                percentile,
            )

            prediction = (
                val_score
                >
                threshold
            ).astype(int)

            f1 = f1_score(
                val_labels,
                prediction,
                zero_division=0,
            )

            precision = precision_score(
                val_labels,
                prediction,
                zero_division=0,
            )

            recall = recall_score(
                val_labels,
                prediction,
                zero_division=0,
            )

            tn, fp, fn, tp = (
                confusion_matrix(
                    val_labels,
                    prediction,
                    labels=[0, 1],
                )
                .ravel()
            )

            fpr = (
                fp
                /
                (fp + tn)
            )

            # Primary objective: F1.
            # Tie-breakers:
            # 1. lower FPR
            # 2. higher recall
            # 3. higher precision
            key = (
                f1,
                -fpr,
                recall,
                precision,
            )

            if (
                candidate_best is None
                or
                key
                >
                candidate_best["key"]
            ):

                candidate_best = {
                    "key":
                        key,

                    "weights":
                        weights.copy(),

                    "percentile":
                        float(
                            percentile
                        ),

                    "threshold":
                        float(
                            threshold
                        ),

                    "val_score":
                        val_score.copy(),

                    "prediction":
                        prediction.copy(),

                    "f1":
                        float(f1),

                    "fpr":
                        float(fpr),

                    "recall":
                        float(recall),

                    "precision":
                        float(precision),
                }

        return candidate_best

    for weights in coarse:

        candidate = evaluate_candidate(
            weights
        )

        if (
            best is None
            or
            candidate["key"]
            >
            best["key"]
        ):
            best = candidate

    fine = fine_simplex_weights(
        best["weights"],
        radius=0.10,
        step=FINE_STEP,
    )

    print(
        f"  Fine reliability candidates: "
        f"{len(fine):,}"
    )

    for weights in fine:

        candidate = evaluate_candidate(
            weights
        )

        if (
            candidate["key"]
            >
            best["key"]
        ):
            best = candidate

    return best


# ============================================================
# BUILDING LIST
# ============================================================

selected = pd.read_csv(
    SELECTION_PATH
)

buildings = (
    selected[
        "building_id"
    ]
    .tolist()
)

performance_rows = []
weight_rows = []
type_rows = []
prediction_rows = []
twin_rows = []


# ============================================================
# BUILDING LOOP
# ============================================================

for building_number, building in enumerate(
    buildings,
    start=1,
):

    print("\n" + "=" * 110)
    print(f"TWINGUARD-X v3 DEVELOPMENT: {building}")
    print("=" * 110)

    path = (
        PROCESSED
        /
        f"{building}_locked_protocol.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Explicit leakage guard
    # --------------------------------------------------------

    test_rows = (
        df[
            "experimental_split"
        ]
        ==
        "final_test"
    )

    print(
        f"\nLocked final-test rows detected: "
        f"{test_rows.sum():,}"
    )

    print(
        "Locked final-test rows will NOT "
        "be evaluated."
    )

    # ========================================================
    # TRAINING SAMPLES
    # ========================================================

    train_samples = build_samples(
        df,
        "electricity_original",
        "train",
    )

    val_clean_samples = build_samples(
        df,
        "electricity_original",
        "validation",
    )

    val_observed_samples = build_samples(
        df,
        "electricity_validation",
        "validation",
    )

    if (
        len(val_clean_samples)
        !=
        len(val_observed_samples)
    ):
        raise RuntimeError(
            "Validation sample mismatch."
        )

    print(
        f"\nValid training sequences: "
        f"{len(train_samples):,}"
    )

    print(
        f"Valid validation sequences: "
        f"{len(val_observed_samples):,}"
    )

    # ========================================================
    # FIT SCALERS USING TRAIN ONLY
    # ========================================================

    train_context_raw = np.vstack([
        x["context"]
        for x in train_samples
    ])

    context_scaler = StandardScaler()

    train_context = (
        context_scaler
        .fit_transform(
            train_context_raw
        )
    )

    train_values = (
        df.loc[
            df[
                "experimental_split"
            ]
            ==
            "train",
            "electricity_original",
        ]
        .astype(float)
        .values
    )

    electricity_mean = float(
        np.mean(
            train_values
        )
    )

    electricity_std = float(
        np.std(
            train_values
        )
    )

    if electricity_std < 1e-8:
        electricity_std = 1.0

    train_history = np.stack([
        (
            sample["history"]
            -
            electricity_mean
        )
        /
        electricity_std
        for sample in train_samples
    ])

    train_history = (
        train_history[
            :,
            None,
            :
        ]
    )

    train_target = np.asarray([
        (
            sample["target"]
            -
            electricity_mean
        )
        /
        electricity_std
        for sample in train_samples
    ])

    # ========================================================
    # TRAIN DIGITAL TWIN
    # ========================================================

    torch.manual_seed(
        SEED
        +
        building_number
    )

    model = OperationalDigitalTwin(
        len(
            CONTEXT_COLUMNS
        )
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(
            train_context,
            dtype=torch.float32,
        ),
        torch.tensor(
            train_history,
            dtype=torch.float32,
        ),
        torch.tensor(
            train_target[:, None],
            dtype=torch.float32,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(
        "\nTraining operational digital twin..."
    )

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        model.train()

        epoch_losses = []

        for (
            context_batch,
            history_batch,
            target_batch,
        ) in train_loader:

            context_batch = (
                context_batch.to(
                    DEVICE
                )
            )

            history_batch = (
                history_batch.to(
                    DEVICE
                )
            )

            target_batch = (
                target_batch.to(
                    DEVICE
                )
            )

            optimizer.zero_grad()

            prediction, _, _ = model(
                context_batch,
                history_batch,
            )

            loss = criterion(
                prediction,
                target_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            epoch_losses.append(
                loss.item()
            )

        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
        ):

            print(
                f"  Epoch "
                f"{epoch + 1:03d}/"
                f"{EPOCHS} "
                f"| loss="
                f"{np.mean(epoch_losses):.6f}"
            )

    training_time = (
        time.perf_counter()
        -
        start
    )

    # ========================================================
    # FUNCTION FOR DIGITAL-TWIN INFERENCE
    # ========================================================

    def twin_predict(samples):

        context_raw = np.vstack([
            x["context"]
            for x in samples
        ])

        context_scaled = (
            context_scaler
            .transform(
                context_raw
            )
        )

        history = np.stack([
            (
                x["history"]
                -
                electricity_mean
            )
            /
            electricity_std
            for x in samples
        ])

        history = history[
            :,
            None,
            :
        ]

        model.eval()

        with torch.no_grad():

            prediction, zc, zo = model(
                torch.tensor(
                    context_scaled,
                    dtype=torch.float32,
                ).to(DEVICE),

                torch.tensor(
                    history,
                    dtype=torch.float32,
                ).to(DEVICE),
            )

        prediction = (
            prediction
            .cpu()
            .numpy()
            .ravel()
            *
            electricity_std
            +
            electricity_mean
        )

        return (
            prediction,
            zc.cpu().numpy(),
            zo.cpu().numpy(),
        )

    # ========================================================
    # TRAIN NORMAL DISAGREEMENTS
    # ========================================================

    (
        train_prediction,
        train_zc,
        train_zo,
    ) = twin_predict(
        train_samples
    )

    train_actual = np.asarray([
        x["target"]
        for x in train_samples
    ])

    train_indices = np.asarray([
        x["index"]
        for x in train_samples
    ])

    train_residual_raw = np.abs(
        train_actual
        -
        train_prediction
    )

    # --------------------------------------------------------
    # Fit context-conditioned residual model
    # from TRAINING NORMAL data only.
    # --------------------------------------------------------

    context_residual_model = (
        build_context_bins(
            df,
            train_indices,
            train_residual_raw,
        )
    )

    train_contextual_residual = (
        contextualize_residuals(
            df,
            train_indices,
            train_residual_raw,
            context_residual_model,
        )
    )

    # Residual dynamics
    train_dynamics = np.abs(
        np.diff(
            train_contextual_residual,
            prepend=
            train_contextual_residual[0],
        )
    )

    train_persistence = np.asarray([
        persistence_features(
            x["history"]
        )
        for x in train_samples
    ])

    train_descriptor_raw = np.column_stack([
        train_contextual_residual,
        train_dynamics,
        train_persistence,
    ])

    # --------------------------------------------------------
    # Fit robust scaling on NORMAL TRAINING descriptors only
    # --------------------------------------------------------

    descriptor_median, descriptor_scale = (
        robust_fit(
            train_descriptor_raw
        )
    )

    train_descriptor = (
        robust_transform(
            train_descriptor_raw,
            descriptor_median,
            descriptor_scale,
        )
    )

    # One-sided anomaly evidence.
    train_descriptor = np.maximum(
        train_descriptor,
        0.0,
    )

    train_descriptor = np.clip(
        train_descriptor,
        0.0,
        10.0,
    )

    # ========================================================
    # VALIDATION OBSERVED DISAGREEMENTS
    # ========================================================

    (
        val_prediction,
        val_zc,
        val_zo,
    ) = twin_predict(
        val_observed_samples
    )

    val_indices = np.asarray([
        x["index"]
        for x in val_observed_samples
    ])

    val_observed = np.asarray([
        x["target"]
        for x in val_observed_samples
    ])

    val_original = (
        df.loc[
            val_indices,
            "electricity_original",
        ]
        .astype(float)
        .values
    )

    val_labels = (
        df.loc[
            val_indices,
            "validation_anomaly_label",
        ]
        .astype(int)
        .values
    )

    val_types = (
        df.loc[
            val_indices,
            "validation_anomaly_type",
        ]
        .astype(str)
        .values
    )

    val_residual_raw = np.abs(
        val_observed
        -
        val_prediction
    )

    val_contextual_residual = (
        contextualize_residuals(
            df,
            val_indices,
            val_residual_raw,
            context_residual_model,
        )
    )

    val_dynamics = np.abs(
        np.diff(
            val_contextual_residual,
            prepend=
            val_contextual_residual[0],
        )
    )

    val_persistence = np.asarray([
        persistence_features(
            x["history"]
        )
        for x in val_observed_samples
    ])

    val_descriptor_raw = np.column_stack([
        val_contextual_residual,
        val_dynamics,
        val_persistence,
    ])

    val_descriptor = (
        robust_transform(
            val_descriptor_raw,
            descriptor_median,
            descriptor_scale,
        )
    )

    val_descriptor = np.maximum(
        val_descriptor,
        0.0,
    )

    val_descriptor = np.clip(
        val_descriptor,
        0.0,
        10.0,
    )

    # ========================================================
    # VALIDATION CALIBRATION
    # ========================================================

    print(
        "\nSearching interpretable reliability fusion..."
    )

    calibration = search_calibration(
        train_descriptor,
        val_descriptor,
        val_labels,
    )

    weights = calibration[
        "weights"
    ]

    threshold = calibration[
        "threshold"
    ]

    percentile = calibration[
        "percentile"
    ]

    val_score = calibration[
        "val_score"
    ]

    val_prediction_label = calibration[
        "prediction"
    ]

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    m = calculate_metrics(
        val_labels,
        val_prediction_label,
        val_score,
    )

    # Digital-twin error is measured against CLEAN target
    twin_mae = float(
        np.mean(
            np.abs(
                val_original
                -
                val_prediction
            )
        )
    )

    twin_rmse = float(
        np.sqrt(
            np.mean(
                (
                    val_original
                    -
                    val_prediction
                ) ** 2
            )
        )
    )

    print(
        "\nSelected reliability weights:"
    )

    for name, weight in zip(
        DISAGREEMENT_NAMES,
        weights,
    ):

        print(
            f"  {name:24s}: "
            f"{weight:.4f}"
        )

    print(
        f"\nSelected training-score percentile: "
        f"{percentile:.1f}"
    )

    print(
        f"Selected threshold: "
        f"{threshold:.6f}"
    )

    print(
        "\nVALIDATION performance:"
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

    print(
        f"  Twin MAE  = "
        f"{twin_mae:.4f}"
    )

    print(
        f"  Twin RMSE = "
        f"{twin_rmse:.4f}"
    )

    # ========================================================
    # RECALL BY ANOMALY TYPE
    # ========================================================

    anomaly_types = [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours",
    ]

    print(
        "\nVALIDATION recall by anomaly type:"
    )

    for anomaly_type in anomaly_types:

        mask = (
            val_types
            ==
            anomaly_type
        )

        n = int(
            mask.sum()
        )

        if n > 0:

            recall = float(
                val_prediction_label[
                    mask
                ].mean()
            )

        else:

            recall = np.nan

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
    # SAVE PERFORMANCE
    # ========================================================

    performance_rows.append({
        "building_id":
            building,

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

        "threshold_percentile":
            percentile,

        "threshold":
            threshold,

        "twin_mae":
            twin_mae,

        "twin_rmse":
            twin_rmse,

        "training_time_s":
            training_time,
    })

    for name, weight in zip(
        DISAGREEMENT_NAMES,
        weights,
    ):

        weight_rows.append({
            "building_id":
                building,

            "component":
                name,

            "weight":
                float(weight),
        })

    # ========================================================
    # VALIDATION EXPLANATION OUTPUT
    # ========================================================

    explanation = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            df.loc[
                val_indices,
                "timestamp",
            ].values,

        "anomaly_label":
            val_labels,

        "anomaly_type":
            val_types,

        "electricity_original":
            val_original,

        "electricity_observed":
            val_observed,

        "digital_twin_expected":
            val_prediction,

        "contextual_residual":
            val_descriptor[:, 0],

        "residual_dynamics":
            val_descriptor[:, 1],

        "persistence_3h":
            val_descriptor[:, 2],

        "persistence_6h":
            val_descriptor[:, 3],

        "persistence_12h":
            val_descriptor[:, 4],

        "anomaly_score":
            val_score,

        "threshold":
            threshold,

        "prediction":
            val_prediction_label,
    })

    prediction_rows.append(
        explanation
    )

    # ========================================================
    # SAVE FROZEN DEVELOPMENT PARAMETERS
    # ========================================================

    torch.save(
        model.state_dict(),
        MODELS /
        f"{building}_TwinGuardX_v3_twin.pt",
    )

    calibration_path = (
        MODELS /
        f"{building}_TwinGuardX_v3_calibration.npz"
    )

    np.savez(
        calibration_path,

        weights=
            weights,

        threshold=
            threshold,

        threshold_percentile=
            percentile,

        electricity_mean=
            electricity_mean,

        electricity_std=
            electricity_std,

        context_scaler_mean=
            context_scaler.mean_,

        context_scaler_scale=
            context_scaler.scale_,

        descriptor_median=
            descriptor_median,

        descriptor_scale=
            descriptor_scale,

        residual_temp_quartiles=
            context_residual_model[
                "quartiles"
            ],

        residual_global_median=
            context_residual_model[
                "global_median"
            ],

        residual_global_mad=
            context_residual_model[
                "global_mad"
            ],
    )

    # Save context-bin model separately as CSV
    context_rows = []

    for (
        hour_group,
        temp_group,
    ), (
        median,
        mad,
    ) in (
        context_residual_model[
            "bins"
        ]
        .items()
    ):

        context_rows.append({
            "hour_group":
                hour_group,

            "temperature_group":
                temp_group,

            "residual_median":
                median,

            "residual_mad":
                mad,
        })

    pd.DataFrame(
        context_rows
    ).to_csv(
        MODELS /
        f"{building}_TwinGuardX_v3_context_bins.csv",
        index=False,
    )


# ============================================================
# SAVE TABLES
# ============================================================

performance = pd.DataFrame(
    performance_rows
)

performance.to_csv(
    TABLES /
    "twinguard_x_v3_validation_performance.csv",
    index=False,
)

weights_df = pd.DataFrame(
    weight_rows
)

weights_df.to_csv(
    TABLES /
    "twinguard_x_v3_reliability_weights.csv",
    index=False,
)

types_df = pd.DataFrame(
    type_rows
)

types_df.to_csv(
    TABLES /
    "twinguard_x_v3_validation_type_recall.csv",
    index=False,
)

explanations = pd.concat(
    prediction_rows,
    ignore_index=True,
)

explanations.to_csv(
    TABLES /
    "twinguard_x_v3_validation_explanations.csv",
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 110)
print("TWINGUARD-X v3 — VALIDATION PERFORMANCE")
print("=" * 110)

display_columns = [
    "building_id",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate",
    "threshold_percentile",
    "twin_mae",
    "twin_rmse",
]

display = performance[
    display_columns
].copy()

numeric_columns = (
    display
    .select_dtypes(
        include=np.number
    )
    .columns
)

display[numeric_columns] = (
    display[
        numeric_columns
    ]
    .round(4)
)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# MEAN VALIDATION PERFORMANCE
# ============================================================

metrics_to_average = [
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
        metrics_to_average
    ]
    .mean()
)

print("\n" + "=" * 110)
print("MEAN VALIDATION PERFORMANCE")
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

mean_type_recall = (
    types_df
    .groupby(
        "anomaly_type"
    )["recall"]
    .mean()
)

print("\n" + "=" * 110)
print("MEAN VALIDATION RECALL BY ANOMALY TYPE")
print("=" * 110)

print(
    mean_type_recall
    .round(4)
    .to_string()
)


# ============================================================
# RELIABILITY WEIGHTS
# ============================================================

weight_pivot = (
    weights_df
    .pivot(
        index="building_id",
        columns="component",
        values="weight",
    )
)

print("\n" + "=" * 110)
print("LEARNED RELIABILITY WEIGHTS")
print("=" * 110)

print(
    weight_pivot
    .round(4)
    .to_string()
)


# ============================================================
# FREEZE MANIFEST
# ============================================================

freeze_manifest = {
    "model":
        "TwinGuard-X v3",

    "status":
        "DEVELOPMENT_COMPLETE_PENDING_REVIEW",

    "seed":
        SEED,

    "history_hours":
        HISTORY,

    "epochs":
        EPOCHS,

    "training_weeks":
        TRAIN_WEEKS,

    "validation_weeks":
        VALIDATION_WEEKS,

    "locked_final_test_weeks":
        FINAL_TEST_WEEKS,

    "final_test_evaluated":
        False,

    "note":
        (
            "All model training, reliability-weight "
            "selection and threshold calibration were "
            "performed without evaluating the locked "
            "final-test partition."
        ),
}

with open(
    TABLES /
    "twinguard_x_v3_freeze_manifest.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        freeze_manifest,
        f,
        indent=4,
    )


print("\n" + "=" * 110)
print("STEP 11 COMPLETE")
print("=" * 110)

print("\nSaved:")
print(
    TABLES /
    "twinguard_x_v3_validation_performance.csv"
)
print(
    TABLES /
    "twinguard_x_v3_reliability_weights.csv"
)
print(
    TABLES /
    "twinguard_x_v3_validation_type_recall.csv"
)
print(
    TABLES /
    "twinguard_x_v3_validation_explanations.csv"
)
print(
    TABLES /
    "twinguard_x_v3_freeze_manifest.json"
)