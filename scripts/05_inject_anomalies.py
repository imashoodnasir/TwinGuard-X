from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

SELECTION_PATH = (
    TABLES / "selected_three_public_buildings.csv"
)

SEED = 42

print("=" * 90)
print("EIRT 2026 — CONTROLLED ANOMALY INJECTION")
print("=" * 90)

selected = pd.read_csv(SELECTION_PATH)

buildings = selected["building_id"].tolist()

# ============================================================
# Helper: find free interval
# ============================================================

def find_free_interval(
    candidate_indices,
    duration,
    occupied,
    rng,
    max_attempts=10000
):

    candidate_indices = np.array(
        candidate_indices,
        dtype=int
    )

    if len(candidate_indices) < duration:
        return None

    starts = candidate_indices[
        candidate_indices <=
        candidate_indices.max() - duration + 1
    ]

    if len(starts) == 0:
        return None

    for _ in range(max_attempts):

        start = int(
            rng.choice(starts)
        )

        interval = np.arange(
            start,
            start + duration
        )

        # Must actually exist in candidate pool
        if not np.all(
            np.isin(
                interval,
                candidate_indices
            )
        ):
            continue

        # Must not overlap previous anomaly
        if np.any(
            np.isin(
                interval,
                list(occupied)
            )
        ):
            continue

        return interval

    return None


# ============================================================
# Process each building
# ============================================================

all_event_summaries = []

for building_number, building in enumerate(
    buildings,
    start=1
):

    print("\n" + "=" * 90)
    print(f"BUILDING {building_number}: {building}")
    print("=" * 90)

    path = (
        PROCESSED /
        f"{building}_experiment.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    # --------------------------------------------------------
    # Initialize labels
    # --------------------------------------------------------

    df["electricity_original"] = (
        df["electricity"].copy()
    )

    df["electricity_injected"] = (
        df["electricity"].copy()
    )

    df["anomaly_label"] = 0
    df["anomaly_type"] = "normal"
    df["anomaly_event_id"] = ""

    # --------------------------------------------------------
    # Train/test positions
    # --------------------------------------------------------

    train_mask = (
        df["split"] == "train"
    )

    test_mask = (
        df["split"] == "test"
    )

    train_values = (
        df.loc[
            train_mask,
            "electricity"
        ]
        .astype(float)
    )

    test_indices = np.where(
        test_mask.values
    )[0]

    # Reproducible but different sequence per building
    rng = np.random.default_rng(
        SEED + building_number
    )

    # Robust training statistics
    train_median = (
        train_values.median()
    )

    train_std = (
        train_values.std()
    )

    q25 = train_values.quantile(0.25)
    q75 = train_values.quantile(0.75)

    iqr = q75 - q25

    if iqr <= 0:
        iqr = train_std

    print(
        f"\nTraining median: {train_median:.3f}"
    )

    print(
        f"Training std:    {train_std:.3f}"
    )

    print(
        f"Training IQR:    {iqr:.3f}"
    )

    occupied = set()

    event_counter = 0

    # --------------------------------------------------------
    # Function to register anomaly
    # --------------------------------------------------------

    def register_event(
        indices,
        anomaly_type,
        values
    ):

        nonlocal_dummy = None

        # event_counter handled externally
        df.loc[
            indices,
            "electricity_injected"
        ] = values

        df.loc[
            indices,
            "anomaly_label"
        ] = 1

        df.loc[
            indices,
            "anomaly_type"
        ] = anomaly_type

        return


    # ========================================================
    # A1. POINT SPIKES
    # ========================================================

    spike_events = 10

    available = np.array([
        i for i in test_indices
        if i not in occupied
    ])

    spike_positions = rng.choice(
        available,
        size=spike_events,
        replace=False
    )

    for idx in spike_positions:

        event_counter += 1

        # Random positive or negative spike
        direction = rng.choice(
            [-1, 1]
        )

        magnitude = rng.uniform(
            3.0,
            5.0
        ) * train_std

        new_value = (
            df.loc[
                idx,
                "electricity_original"
            ]
            +
            direction * magnitude
        )

        # Electricity cannot be negative
        new_value = max(
            new_value,
            0.01
        )

        register_event(
            [idx],
            "spike",
            [new_value]
        )

        event_id = (
            f"{building}_SP_{event_counter:02d}"
        )

        df.loc[
            idx,
            "anomaly_event_id"
        ] = event_id

        occupied.add(idx)

    # ========================================================
    # A2. PERSISTENT SHIFT
    # ========================================================

    for _ in range(3):

        interval = find_free_interval(
            test_indices,
            duration=12,
            occupied=occupied,
            rng=rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place persistent shift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        shift = (
            direction *
            rng.uniform(
                1.5,
                2.5
            ) *
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

        register_event(
            interval,
            "persistent_shift",
            values
        )

        event_id = (
            f"{building}_SH_{event_counter:02d}"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A3. GRADUAL DRIFT
    # ========================================================

    for _ in range(2):

        interval = find_free_interval(
            test_indices,
            duration=24,
            occupied=occupied,
            rng=rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place gradual drift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        final_magnitude = (
            direction *
            rng.uniform(
                2.0,
                3.0
            ) *
            train_std
        )

        drift = np.linspace(
            0,
            final_magnitude,
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

        register_event(
            interval,
            "gradual_drift",
            values
        )

        event_id = (
            f"{building}_DR_{event_counter:02d}"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A4. STUCK MEASUREMENT
    # ========================================================

    for _ in range(2):

        interval = find_free_interval(
            test_indices,
            duration=12,
            occupied=occupied,
            rng=rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place stuck measurement."
            )

        event_counter += 1

        # Freeze at first observed value
        stuck_value = (
            df.loc[
                interval[0],
                "electricity_original"
            ]
        )

        values = np.repeat(
            stuck_value,
            len(interval)
        )

        register_event(
            interval,
            "stuck",
            values
        )

        event_id = (
            f"{building}_ST_{event_counter:02d}"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # A5. CONTEXTUAL AFTER-HOURS ANOMALY
    # ========================================================

    # Candidate hours:
    # midnight–05:00 in test period
    after_hours = np.where(
        (
            test_mask.values
        )
        &
        (
            df["hour"].isin(
                [0, 1, 2, 3, 4, 5]
            ).values
        )
    )[0]

    for _ in range(2):

        # We use 6-hour continuous nighttime periods.
        # This is more defensible than forcing 8 hours
        # into a 6-hour after-hours definition.

        interval = find_free_interval(
            after_hours,
            duration=6,
            occupied=occupied,
            rng=rng
        )

        if interval is None:
            raise RuntimeError(
                "Could not place contextual anomaly."
            )

        event_counter += 1

        # Raise consumption substantially above
        # its normal after-hours value.
        boost = rng.uniform(
            1.5,
            2.5
        ) * train_std

        values = (
            df.loc[
                interval,
                "electricity_original"
            ].values
            +
            boost
        )

        register_event(
            interval,
            "contextual_after_hours",
            values
        )

        event_id = (
            f"{building}_CX_{event_counter:02d}"
        )

        df.loc[
            interval,
            "anomaly_event_id"
        ] = event_id

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # IMPORTANT:
    # Recompute electricity-derived features
    # using injected electricity.
    # ========================================================

    x = df[
        "electricity_injected"
    ].astype(float)

    df["electricity_lag_1h_injected"] = (
        x.shift(1)
    )

    df["electricity_lag_24h_injected"] = (
        x.shift(24)
    )

    df[
        "electricity_rolling_24h_injected"
    ] = (
        x.rolling(
            24,
            min_periods=1
        ).mean()
    )

    df[
        "electricity_rolling_std_24h_injected"
    ] = (
        x.rolling(
            24,
            min_periods=2
        ).std()
    )

    # Preserve clean training values
    # and ensure no training anomaly exists.
    assert (
        df.loc[
            train_mask,
            "anomaly_label"
        ].sum()
        == 0
    )

    # ========================================================
    # Summaries
    # ========================================================

    test_df = df[
        df["split"] == "test"
    ]

    anomaly_count = int(
        test_df["anomaly_label"].sum()
    )

    anomaly_rate = (
        anomaly_count /
        len(test_df)
    )

    print("\nInjected anomaly summary:")

    counts = (
        test_df[
            test_df["anomaly_label"] == 1
        ]["anomaly_type"]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    print(
        f"\nTotal anomalous hours: "
        f"{anomaly_count}"
    )

    print(
        f"Test observations:     "
        f"{len(test_df)}"
    )

    print(
        f"Anomaly prevalence:    "
        f"{anomaly_rate * 100:.2f}%"
    )

    # Event summary
    events = (
        test_df[
            test_df["anomaly_label"] == 1
        ]
        .groupby(
            [
                "anomaly_event_id",
                "anomaly_type"
            ]
        )
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            duration_hours=(
                "timestamp",
                "size"
            ),
            original_mean=(
                "electricity_original",
                "mean"
            ),
            injected_mean=(
                "electricity_injected",
                "mean"
            )
        )
        .reset_index()
    )

    events.insert(
        0,
        "building_id",
        building
    )

    all_event_summaries.append(
        events
    )

    # ========================================================
    # Save
    # ========================================================

    output = (
        PROCESSED /
        f"{building}_with_anomalies.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved:\n{output}"
    )

# ============================================================
# Save event manifest
# ============================================================

event_manifest = pd.concat(
    all_event_summaries,
    ignore_index=True
)

manifest_path = (
    TABLES /
    "anomaly_event_manifest.csv"
)

event_manifest.to_csv(
    manifest_path,
    index=False
)

print("\n" + "=" * 90)
print("ANOMALY EVENT MANIFEST")
print("=" * 90)

print(
    event_manifest[
        [
            "building_id",
            "anomaly_type",
            "start",
            "end",
            "duration_hours"
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved event manifest:")
print(manifest_path)

# ============================================================
# Final verification
# ============================================================

print("\n" + "=" * 90)
print("STEP 5 COMPLETE")
print("=" * 90)

print(
    "\nAll anomalies were injected into TEST data only."
)

print(
    "Original electricity values remain preserved."
)

print(
    "Ground-truth labels and anomaly types are available."
)