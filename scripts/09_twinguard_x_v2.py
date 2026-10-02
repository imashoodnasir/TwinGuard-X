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
STEP8_PATH = TABLES / "twinguard_x_explanations.csv"

SEED = 42

TEST_WEEKS = [
    1, 8, 9, 10,
    11, 13, 21, 25
]

HISTORY = 24

TWIN_EPOCHS = 80
FUSION_EPOCHS = 50

BATCH_SIZE = 64
LEARNING_RATE = 0.001

THRESHOLD_PERCENTILE = 95

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 105)
print("EIRT 2026 — TWINGUARD-X v2")
print("Context-Conditioned Multi-Scale Twin Disagreement (CMTD)")
print("=" * 105)

print(f"\nDevice: {DEVICE}")
print(f"Seed: {SEED}")
print(f"History: {HISTORY} hours")
print(f"Twin epochs: {TWIN_EPOCHS}")
print(f"Fusion epochs: {FUSION_EPOCHS}")


# ============================================================
# CONTEXT
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


# ============================================================
# MODEL
# ============================================================

class TwinGuardV2(nn.Module):

    def __init__(
        self,
        context_dim
    ):

        super().__init__()

        # ----------------------------------------------------
        # Context encoder
        # ----------------------------------------------------

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # ----------------------------------------------------
        # Lightweight temporal convolutional encoder
        # Input: [batch, 1, 24]
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            dilation=2,
            padding=2
        )

        self.conv3 = nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            dilation=4,
            padding=4
        )

        self.activation = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # Digital-twin fusion
        # 8 context + 8 operational
        # ----------------------------------------------------

        self.twin_fusion = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.twin_head = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def encode_operation(
        self,
        history
    ):

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

    def forward(
        self,
        context,
        history
    ):

        zc = self.context_encoder(
            context
        )

        zo = self.encode_operation(
            history
        )

        joint = torch.cat(
            [zc, zo],
            dim=1
        )

        z = self.twin_fusion(
            joint
        )

        expected = self.twin_head(
            z
        )

        return expected, zc, zo


# ============================================================
# SELF-SUPERVISED ANOMALY FUSION
# ============================================================

class CMTDFusion(nn.Module):

    def __init__(
        self,
        descriptor_dim=5,
        context_dim=8,
        operational_dim=8
    ):

        super().__init__()

        total = (
            descriptor_dim
            +
            context_dim
            +
            operational_dim
        )

        # Attention/gating representation
        self.feature_gate = nn.Sequential(
            nn.Linear(total, 16),
            nn.ReLU(),
            nn.Linear(16, descriptor_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(total, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(
        self,
        descriptor,
        context_latent,
        operational_latent
    ):

        joint = torch.cat(
            [
                descriptor,
                context_latent,
                operational_latent
            ],
            dim=1
        )

        gate = self.feature_gate(
            joint
        )

        gated_descriptor = (
            descriptor
            *
            gate
        )

        classifier_input = torch.cat(
            [
                gated_descriptor,
                context_latent,
                operational_latent
            ],
            dim=1
        )

        logit = self.classifier(
            classifier_input
        )

        return logit, gate


# ============================================================
# BUILD 24-HOUR SEQUENCES
# ============================================================

def build_samples(
    df,
    electricity_column
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
            "target": electricity[i]
        })

    return samples


# ============================================================
# MULTI-SCALE PERSISTENCE
# ============================================================

def persistence_descriptor(
    history
):

    history = np.asarray(
        history,
        dtype=float
    )

    reference_std = (
        np.std(history)
        +
        1e-6
    )

    values = []

    for k in [3, 6, 12]:

        local_std = np.std(
            history[-k:]
        )

        ratio = (
            local_std
            /
            reference_std
        )

        # High score = suspicious stagnation
        stagnation = 1.0 / (
            ratio + 0.10
        )

        values.append(
            stagnation
        )

    return values


# ============================================================
# ROBUST SCALING
# ============================================================

def robust_fit(
    X
):

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

    scale = (
        1.4826 * mad
    )

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
    scale
):

    return (
        np.asarray(X)
        -
        median
    ) / scale


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
# LOAD STEP-8 TEST VALUES
# ============================================================

if not STEP8_PATH.exists():

    raise FileNotFoundError(
        "Run Step 8 first. "
        "twinguard_x_explanations.csv not found."
    )

step8 = pd.read_csv(
    STEP8_PATH,
    parse_dates=["timestamp"]
)

selected = pd.read_csv(
    SELECTION_PATH
)

buildings = (
    selected["building_id"]
    .tolist()
)

all_results = []
all_types = []
all_predictions = []
all_gates = []


# ============================================================
# BUILDING LOOP
# ============================================================

for building_number, building in enumerate(
    buildings,
    start=1
):

    print("\n" + "=" * 105)
    print(f"TWINGUARD-X v2: {building}")
    print("=" * 105)

    # --------------------------------------------------------
    # Original clean data
    # --------------------------------------------------------

    df = pd.read_csv(
        PROCESSED /
        f"{building}_experiment.csv",
        parse_dates=["timestamp"]
    )

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Week blocks
    # --------------------------------------------------------

    first_time = (
        df["timestamp"].min()
    )

    hours = (
        (
            df["timestamp"]
            -
            first_time
        )
        .dt.total_seconds()
        / 3600
    )

    df["week_block"] = (
        hours
        // (24 * 7)
    ).astype(int)

    test_mask = (
        df["week_block"]
        .isin(TEST_WEEKS)
    )

    train_mask = ~test_mask

    # --------------------------------------------------------
    # Insert EXACT Step-8 corrupted test electricity
    # --------------------------------------------------------

    df["electricity_original"] = (
        df["electricity"]
        .astype(float)
    )

    df["electricity_observed"] = (
        df["electricity_original"]
        .copy()
    )

    df["anomaly_label"] = 0
    df["anomaly_type"] = "normal"

    old = (
        step8[
            step8["building_id"]
            ==
            building
        ]
        .copy()
    )

    old_lookup = (
        old.set_index("timestamp")
    )

    matched = 0

    for i in df.index[test_mask]:

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
                "electricity_observed"
            ] = row[
                "electricity_observed"
            ]

            df.loc[
                i,
                "anomaly_label"
            ] = int(
                row["anomaly_label"]
            )

            df.loc[
                i,
                "anomaly_type"
            ] = row[
                "anomaly_type"
            ]

            matched += 1

    print(
        f"\nMatched Step-8 test observations: "
        f"{matched:,}"
    )

    print(
        f"Training observations: "
        f"{train_mask.sum():,}"
    )

    print(
        f"Test observations: "
        f"{test_mask.sum():,}"
    )

    print(
        f"Test anomalies: "
        f"{df.loc[test_mask, 'anomaly_label'].sum():,}"
    )

    # ========================================================
    # BUILD CLEAN TRAINING SAMPLES
    # ========================================================

    clean_samples = build_samples(
        df,
        "electricity_original"
    )

    train_samples = [
        x for x in clean_samples
        if train_mask.iloc[
            x["index"]
        ]
    ]

    print(
        f"Valid clean training samples: "
        f"{len(train_samples):,}"
    )

    # --------------------------------------------------------
    # Context scaling
    # --------------------------------------------------------

    context_train_raw = np.vstack([
        x["context"]
        for x in train_samples
    ])

    context_scaler = StandardScaler()

    context_train = (
        context_scaler
        .fit_transform(
            context_train_raw
        )
    )

    # --------------------------------------------------------
    # Electricity scaling
    # --------------------------------------------------------

    electricity_train = (
        df.loc[
            train_mask,
            "electricity_original"
        ]
        .astype(float)
        .values
    )

    e_mean = np.mean(
        electricity_train
    )

    e_std = np.std(
        electricity_train
    )

    if e_std < 1e-8:
        e_std = 1.0

    histories_train = np.stack([
        (
            x["history"]
            -
            e_mean
        )
        /
        e_std

        for x in train_samples
    ])

    histories_train = (
        histories_train[:, None, :]
    )

    targets_train = np.asarray([
        (
            x["target"]
            -
            e_mean
        )
        /
        e_std

        for x in train_samples
    ])

    # ========================================================
    # TRAIN DIGITAL TWIN
    # ========================================================

    model = TwinGuardV2(
        len(CONTEXT_COLUMNS)
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(
            context_train,
            dtype=torch.float32
        ),
        torch.tensor(
            histories_train,
            dtype=torch.float32
        ),
        torch.tensor(
            targets_train[:, None],
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("\nTraining operational digital twin...")

    start_training = time.perf_counter()

    for epoch in range(
        TWIN_EPOCHS
    ):

        model.train()

        losses = []

        for c, h, y in loader:

            c = c.to(DEVICE)
            h = h.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            prediction, _, _ = model(
                c,
                h
            )

            loss = criterion(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"  Twin epoch "
                f"{epoch + 1:03d}/"
                f"{TWIN_EPOCHS} "
                f"| loss="
                f"{np.mean(losses):.6f}"
            )

    twin_training_time = (
        time.perf_counter()
        -
        start_training
    )

    # ========================================================
    # EXTRACT CLEAN TRAIN REPRESENTATIONS
    # ========================================================

    model.eval()

    with torch.no_grad():

        c_tensor = torch.tensor(
            context_train,
            dtype=torch.float32
        ).to(DEVICE)

        h_tensor = torch.tensor(
            histories_train,
            dtype=torch.float32
        ).to(DEVICE)

        (
            train_pred_scaled,
            train_zc,
            train_zo
        ) = model(
            c_tensor,
            h_tensor
        )

    train_pred = (
        train_pred_scaled
        .cpu()
        .numpy()
        .ravel()
        *
        e_std
        +
        e_mean
    )

    train_zc = (
        train_zc
        .cpu()
        .numpy()
    )

    train_zo = (
        train_zo
        .cpu()
        .numpy()
    )

    train_actual = np.asarray([
        x["target"]
        for x in train_samples
    ])

    # ========================================================
    # CLEAN TRAIN DESCRIPTORS
    # ========================================================

    train_residual = np.abs(
        train_actual
        -
        train_pred
    )

    train_delta = np.abs(
        np.diff(
            train_residual,
            prepend=train_residual[0]
        )
    )

    train_descriptor = []

    for j, sample in enumerate(
        train_samples
    ):

        persistence = (
            persistence_descriptor(
                sample["history"]
            )
        )

        train_descriptor.append([
            train_residual[j],
            train_delta[j],
            persistence[0],
            persistence[1],
            persistence[2]
        ])

    train_descriptor = np.asarray(
        train_descriptor
    )

    # --------------------------------------------------------
    # Robust descriptor scaling
    # --------------------------------------------------------

    desc_med, desc_scale = (
        robust_fit(
            train_descriptor
        )
    )

    train_descriptor_scaled = (
        robust_transform(
            train_descriptor,
            desc_med,
            desc_scale
        )
    )

    # Clip extreme training values
    train_descriptor_scaled = np.clip(
        train_descriptor_scaled,
        -8,
        8
    )

    # ========================================================
    # SELF-SUPERVISED PROXY CORRUPTIONS
    # ========================================================

    print(
        "\nGenerating self-supervised "
        "operational perturbations..."
    )

    rng = np.random.default_rng(
        SEED + building_number
    )

    proxy_descriptors = []
    proxy_zc = []
    proxy_zo = []

    # --------------------------------------------------------
    # We corrupt DESCRIPTORS rather than reproduce the
    # five final test faults.
    #
    # Generic perturbations:
    #  - disagreement amplification
    #  - abrupt residual change
    #  - low-variance persistence
    #  - mixed inconsistency
    # --------------------------------------------------------

    for i in range(
        len(train_descriptor_scaled)
    ):

        base = (
            train_descriptor_scaled[i]
            .copy()
        )

        corruption = int(
            rng.integers(
                0,
                4
            )
        )

        corrupted = base.copy()

        if corruption == 0:

            # Twin disagreement
            corrupted[0] += (
                rng.uniform(
                    2.0,
                    5.0
                )
            )

        elif corruption == 1:

            # Dynamic disagreement
            corrupted[1] += (
                rng.uniform(
                    2.0,
                    5.0
                )
            )

        elif corruption == 2:

            # Multi-scale stagnation
            corrupted[2:] += (
                rng.uniform(
                    2.0,
                    5.0,
                    size=3
                )
            )

        else:

            # Mixed operational inconsistency
            chosen = rng.choice(
                5,
                size=3,
                replace=False
            )

            corrupted[chosen] += (
                rng.uniform(
                    1.5,
                    4.0,
                    size=3
                )
            )

        proxy_descriptors.append(
            corrupted
        )

        proxy_zc.append(
            train_zc[i]
        )

        proxy_zo.append(
            train_zo[i]
        )

    proxy_descriptors = np.asarray(
        proxy_descriptors
    )

    proxy_zc = np.asarray(
        proxy_zc
    )

    proxy_zo = np.asarray(
        proxy_zo
    )

    # ========================================================
    # FUSION TRAINING DATA
    # ========================================================

    fusion_descriptor = np.vstack([
        train_descriptor_scaled,
        proxy_descriptors
    ])

    fusion_zc = np.vstack([
        train_zc,
        proxy_zc
    ])

    fusion_zo = np.vstack([
        train_zo,
        proxy_zo
    ])

    fusion_labels = np.concatenate([
        np.zeros(
            len(train_descriptor_scaled)
        ),
        np.ones(
            len(proxy_descriptors)
        )
    ])

    permutation = rng.permutation(
        len(fusion_labels)
    )

    fusion_descriptor = (
        fusion_descriptor[
            permutation
        ]
    )

    fusion_zc = (
        fusion_zc[
            permutation
        ]
    )

    fusion_zo = (
        fusion_zo[
            permutation
        ]
    )

    fusion_labels = (
        fusion_labels[
            permutation
        ]
    )

    # ========================================================
    # TRAIN CMTD FUSION
    # ========================================================

    fusion = CMTDFusion().to(
        DEVICE
    )

    fusion_optimizer = (
        torch.optim.Adam(
            fusion.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5
        )
    )

    bce = nn.BCEWithLogitsLoss()

    fusion_dataset = TensorDataset(
        torch.tensor(
            fusion_descriptor,
            dtype=torch.float32
        ),
        torch.tensor(
            fusion_zc,
            dtype=torch.float32
        ),
        torch.tensor(
            fusion_zo,
            dtype=torch.float32
        ),
        torch.tensor(
            fusion_labels[:, None],
            dtype=torch.float32
        )
    )

    fusion_loader = DataLoader(
        fusion_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(
        "\nTraining CMTD self-supervised fusion..."
    )

    fusion_start = time.perf_counter()

    for epoch in range(
        FUSION_EPOCHS
    ):

        fusion.train()

        losses = []

        for (
            descriptor_batch,
            zc_batch,
            zo_batch,
            label_batch
        ) in fusion_loader:

            descriptor_batch = (
                descriptor_batch.to(
                    DEVICE
                )
            )

            zc_batch = zc_batch.to(
                DEVICE
            )

            zo_batch = zo_batch.to(
                DEVICE
            )

            label_batch = (
                label_batch.to(
                    DEVICE
                )
            )

            fusion_optimizer.zero_grad()

            logit, gate = fusion(
                descriptor_batch,
                zc_batch,
                zo_batch
            )

            loss = bce(
                logit,
                label_batch
            )

            loss.backward()

            fusion_optimizer.step()

            losses.append(
                loss.item()
            )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"  Fusion epoch "
                f"{epoch + 1:03d}/"
                f"{FUSION_EPOCHS} "
                f"| loss="
                f"{np.mean(losses):.6f}"
            )

    fusion_training_time = (
        time.perf_counter()
        -
        fusion_start
    )

    # ========================================================
    # CLEAN TRAIN ANOMALY SCORES
    # Threshold comes ONLY from clean training data
    # ========================================================

    fusion.eval()

    with torch.no_grad():

        train_logits, train_gates = (
            fusion(
                torch.tensor(
                    train_descriptor_scaled,
                    dtype=torch.float32
                ).to(DEVICE),

                torch.tensor(
                    train_zc,
                    dtype=torch.float32
                ).to(DEVICE),

                torch.tensor(
                    train_zo,
                    dtype=torch.float32
                ).to(DEVICE)
            )
        )

        train_scores = torch.sigmoid(
            train_logits
        ).cpu().numpy().ravel()

    threshold = np.percentile(
        train_scores,
        THRESHOLD_PERCENTILE
    )

    print(
        f"\nTraining-only anomaly threshold: "
        f"{threshold:.6f}"
    )

    # ========================================================
    # BUILD OBSERVED TEST SAMPLES
    # ========================================================

    observed_samples = build_samples(
        df,
        "electricity_observed"
    )

    test_samples = [
        x for x in observed_samples
        if test_mask.iloc[
            x["index"]
        ]
    ]

    # ========================================================
    # TEST MATRICES
    # ========================================================

    test_context_raw = np.vstack([
        x["context"]
        for x in test_samples
    ])

    test_context = (
        context_scaler
        .transform(
            test_context_raw
        )
    )

    test_histories = np.stack([
        (
            x["history"]
            -
            e_mean
        )
        /
        e_std

        for x in test_samples
    ])

    test_histories = (
        test_histories[:, None, :]
    )

    test_observed = np.asarray([
        x["target"]
        for x in test_samples
    ])

    test_indices = np.asarray([
        x["index"]
        for x in test_samples
    ])

    y_test = (
        df.loc[
            test_indices,
            "anomaly_label"
        ]
        .astype(int)
        .values
    )

    test_types = (
        df.loc[
            test_indices,
            "anomaly_type"
        ]
        .values
    )

    # ========================================================
    # DIGITAL-TWIN INFERENCE
    # ========================================================

    inference_start = (
        time.perf_counter()
    )

    model.eval()

    with torch.no_grad():

        (
            test_pred_scaled,
            test_zc,
            test_zo
        ) = model(
            torch.tensor(
                test_context,
                dtype=torch.float32
            ).to(DEVICE),

            torch.tensor(
                test_histories,
                dtype=torch.float32
            ).to(DEVICE)
        )

    test_pred = (
        test_pred_scaled
        .cpu()
        .numpy()
        .ravel()
        *
        e_std
        +
        e_mean
    )

    test_zc = (
        test_zc
        .cpu()
        .numpy()
    )

    test_zo = (
        test_zo
        .cpu()
        .numpy()
    )

    # ========================================================
    # TEST CMTD DESCRIPTORS
    # ========================================================

    test_residual = np.abs(
        test_observed
        -
        test_pred
    )

    test_delta = np.abs(
        np.diff(
            test_residual,
            prepend=test_residual[0]
        )
    )

    test_descriptor = []

    for j, sample in enumerate(
        test_samples
    ):

        persistence = (
            persistence_descriptor(
                sample["history"]
            )
        )

        test_descriptor.append([
            test_residual[j],
            test_delta[j],
            persistence[0],
            persistence[1],
            persistence[2]
        ])

    test_descriptor = np.asarray(
        test_descriptor
    )

    test_descriptor_scaled = (
        robust_transform(
            test_descriptor,
            desc_med,
            desc_scale
        )
    )

    test_descriptor_scaled = np.clip(
        test_descriptor_scaled,
        -8,
        8
    )

    # ========================================================
    # FINAL CMTD SCORE
    # ========================================================

    with torch.no_grad():

        test_logits, test_gates = (
            fusion(
                torch.tensor(
                    test_descriptor_scaled,
                    dtype=torch.float32
                ).to(DEVICE),

                torch.tensor(
                    test_zc,
                    dtype=torch.float32
                ).to(DEVICE),

                torch.tensor(
                    test_zo,
                    dtype=torch.float32
                ).to(DEVICE)
            )
        )

        test_score = torch.sigmoid(
            test_logits
        ).cpu().numpy().ravel()

        test_gates = (
            test_gates
            .cpu()
            .numpy()
        )

    inference_time = (
        time.perf_counter()
        -
        inference_start
    )

    prediction = (
        test_score
        >
        threshold
    ).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

    m = metrics(
        y_test,
        prediction,
        test_score
    )

    # --------------------------------------------------------
    # Digital twin quality evaluated against CLEAN target
    # --------------------------------------------------------

    original_target = (
        df.loc[
            test_indices,
            "electricity_original"
        ]
        .astype(float)
        .values
    )

    twin_mae = np.mean(
        np.abs(
            original_target
            -
            test_pred
        )
    )

    twin_rmse = np.sqrt(
        np.mean(
            (
                original_target
                -
                test_pred
            ) ** 2
        )
    )

    total_training_time = (
        twin_training_time
        +
        fusion_training_time
    )

    all_results.append({
        "building_id":
            building,

        "model":
            "TwinGuard-X v2",

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

        "threshold":
            threshold,

        "twin_mae":
            twin_mae,

        "twin_rmse":
            twin_rmse,

        "training_time_s":
            total_training_time,

        "inference_time_s":
            inference_time
    })

    print("\nTwinGuard-X v2 performance:")

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
    # GATE EXPLANATIONS
    # ========================================================

    gate_names = [
        "twin_residual",
        "residual_dynamics",
        "persistence_3h",
        "persistence_6h",
        "persistence_12h"
    ]

    print(
        "\nMean CMTD explanation gates:"
    )

    for k, name in enumerate(
        gate_names
    ):

        value = (
            test_gates[:, k]
            .mean()
        )

        print(
            f"  {name:22s}: "
            f"{value:.4f}"
        )

        all_gates.append({
            "building_id":
                building,
            "component":
                name,
            "mean_gate":
                value
        })

    # ========================================================
    # RECALL BY ANOMALY TYPE
    # ========================================================

    anomaly_types = [
        "spike",
        "persistent_shift",
        "gradual_drift",
        "stuck",
        "contextual_after_hours"
    ]

    print(
        "\nRecall by anomaly type:"
    )

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

        print(
            f"  {anomaly_type:25s} "
            f"{recall:.4f} "
            f"(n={n})"
        )

        all_types.append({
            "building_id":
                building,

            "model":
                "TwinGuard-X v2",

            "anomaly_type":
                anomaly_type,

            "n":
                n,

            "recall":
                recall
        })

    # ========================================================
    # PER-HOUR EXPLANATION TABLE
    # ========================================================

    output = pd.DataFrame({
        "building_id":
            building,

        "timestamp":
            df.loc[
                test_indices,
                "timestamp"
            ].values,

        "anomaly_label":
            y_test,

        "anomaly_type":
            test_types,

        "electricity_original":
            original_target,

        "electricity_observed":
            test_observed,

        "digital_twin_expected":
            test_pred,

        "twin_residual":
            test_residual,

        "residual_dynamics":
            test_delta,

        "persistence_3h":
            test_descriptor[:, 2],

        "persistence_6h":
            test_descriptor[:, 3],

        "persistence_12h":
            test_descriptor[:, 4],

        "gate_residual":
            test_gates[:, 0],

        "gate_dynamics":
            test_gates[:, 1],

        "gate_persistence_3h":
            test_gates[:, 2],

        "gate_persistence_6h":
            test_gates[:, 3],

        "gate_persistence_12h":
            test_gates[:, 4],

        "anomaly_score":
            test_score,

        "threshold":
            threshold,

        "prediction":
            prediction
    })

    all_predictions.append(
        output
    )

    torch.save(
        model.state_dict(),
        MODELS /
        f"{building}_TwinGuardX_v2_twin.pt"
    )

    torch.save(
        fusion.state_dict(),
        MODELS /
        f"{building}_TwinGuardX_v2_fusion.pt"
    )


# ============================================================
# SAVE PERFORMANCE
# ============================================================

results = pd.DataFrame(
    all_results
)

results.to_csv(
    TABLES /
    "twinguard_x_v2_performance.csv",
    index=False
)

print("\n" + "=" * 105)
print("TWINGUARD-X v2 PERFORMANCE")
print("=" * 105)

display_columns = [
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
        display_columns
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
# MEAN PERFORMANCE
# ============================================================

metric_columns = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "specificity",
    "false_positive_rate"
]

mean_metrics = (
    results[
        metric_columns
    ]
    .mean()
)

print("\n" + "=" * 105)
print("MEAN TWINGUARD-X v2 PERFORMANCE")
print("=" * 105)

for metric, value in (
    mean_metrics.items()
):

    print(
        f"{metric:25s}: "
        f"{value:.4f}"
    )


# ============================================================
# TYPE RESULTS
# ============================================================

type_results = pd.DataFrame(
    all_types
)

type_results.to_csv(
    TABLES /
    "twinguard_x_v2_anomaly_type_recall.csv",
    index=False
)

type_mean = (
    type_results
    .groupby(
        "anomaly_type"
    )["recall"]
    .mean()
)

print("\n" + "=" * 105)
print("MEAN RECALL BY ANOMALY TYPE")
print("=" * 105)

print(
    type_mean
    .round(4)
    .to_string()
)


# ============================================================
# GATES
# ============================================================

gate_results = pd.DataFrame(
    all_gates
)

gate_results.to_csv(
    TABLES /
    "twinguard_x_v2_gate_explanations.csv",
    index=False
)


# ============================================================
# EXPLANATIONS
# ============================================================

prediction_results = pd.concat(
    all_predictions,
    ignore_index=True
)

prediction_results.to_csv(
    TABLES /
    "twinguard_x_v2_explanations.csv",
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison_rows = []

baseline_path = (
    TABLES /
    "corrected_mean_performance.csv"
)

if baseline_path.exists():

    baseline = pd.read_csv(
        baseline_path
    )

    for _, row in (
        baseline.iterrows()
    ):

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


# v1
v1_path = (
    TABLES /
    "twinguard_x_performance.csv"
)

if v1_path.exists():

    v1 = pd.read_csv(
        v1_path
    )

    comparison_rows.append({
        "model":
            "TwinGuard-X v1",

        "precision":
            v1["precision"].mean(),

        "recall":
            v1["recall"].mean(),

        "f1":
            v1["f1"].mean(),

        "auroc":
            v1["auroc"].mean(),

        "auprc":
            v1["auprc"].mean(),

        "false_positive_rate":
            v1[
                "false_positive_rate"
            ].mean()
    })


comparison_rows.append({
    "model":
        "TwinGuard-X v2",

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
    "final_model_comparison_v2.csv",
    index=False
)

print("\n" + "=" * 105)
print("BASELINES VS TWINGUARD-X")
print("=" * 105)

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

print("\n" + "=" * 105)
print("STEP 9 COMPLETE")
print("=" * 105)