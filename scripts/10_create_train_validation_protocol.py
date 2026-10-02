from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

SELECTION_PATH = TABLES / "selected_three_public_buildings.csv"

SEED = 42

# Final test weeks are now LOCKED.
FINAL_TEST_WEEKS = [
    1, 8, 9, 10,
    11, 13, 21, 25
]

# Five development/validation weeks.
# These are selected from the remaining blocks and remain identical
# across all three buildings.
VALIDATION_WEEKS = [
    4, 7, 14, 18, 23
]

ALL_WEEKS = list(range(27))

TRAIN_WEEKS = [
    w for w in ALL_WEEKS
    if w not in FINAL_TEST_WEEKS
    and w not in VALIDATION_WEEKS
]

print("=" * 100)
print("EIRT 2026 — LOCKED TRAIN / VALIDATION / FINAL-TEST PROTOCOL")
print("=" * 100)

print("\nTraining weeks:")
print(TRAIN_WEEKS)

print("\nValidation weeks:")
print(VALIDATION_WEEKS)

print("\nLocked final-test weeks:")
print(FINAL_TEST_WEEKS)

print("\nNumber of blocks:")
print(f"Training:   {len(TRAIN_WEEKS)}")
print(f"Validation: {len(VALIDATION_WEEKS)}")
print(f"Final test: {len(FINAL_TEST_WEEKS)}")


# ============================================================
# CHECK FOR OVERLAP
# ============================================================

assert (
    set(TRAIN_WEEKS)
    .isdisjoint(
        VALIDATION_WEEKS
    )
)

assert (
    set(TRAIN_WEEKS)
    .isdisjoint(
        FINAL_TEST_WEEKS
    )
)

assert (
    set(VALIDATION_WEEKS)
    .isdisjoint(
        FINAL_TEST_WEEKS
    )
)

assert (
    sorted(
        TRAIN_WEEKS
        +
        VALIDATION_WEEKS
        +
        FINAL_TEST_WEEKS
    )
    ==
    ALL_WEEKS
)

print("\nProtocol overlap check: PASSED")


# ============================================================
# ANOMALY INJECTION HELPERS
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
# VALIDATION ANOMALY INJECTION
# ============================================================

def inject_validation_anomalies(
    df,
    validation_mask,
    training_mask,
    building_number
):

    df = df.copy()

    df["electricity_original"] = (
        df["electricity"]
        .astype(float)
    )

    df["electricity_validation"] = (
        df["electricity_original"]
        .copy()
    )

    df["validation_anomaly_label"] = 0
    df["validation_anomaly_type"] = "normal"
    df["validation_event_id"] = ""

    # --------------------------------------------------------
    # Statistics come ONLY from training weeks
    # --------------------------------------------------------

    train_values = (
        df.loc[
            training_mask,
            "electricity_original"
        ]
        .dropna()
    )

    train_std = float(
        train_values.std()
    )

    train_median = float(
        train_values.median()
    )

    if (
        not np.isfinite(train_std)
        or train_std <= 0
    ):
        raise RuntimeError(
            "Invalid training standard deviation."
        )

    validation_indices = np.where(
        validation_mask.values
    )[0]

    # Independent validation RNG.
    # Deliberately different from previous final-test anomaly seed.
    rng = np.random.default_rng(
        1000
        +
        SEED
        +
        building_number
    )

    occupied = set()
    event_counter = 0

    # ========================================================
    # 1. SPIKES
    # 8 isolated events
    # ========================================================

    available = np.asarray([
        i
        for i in validation_indices
        if i not in occupied
    ])

    spike_indices = rng.choice(
        available,
        size=8,
        replace=False
    )

    for idx in spike_indices:

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        magnitude = (
            rng.uniform(
                2.5,
                4.5
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
            direction
            *
            magnitude
        )

        value = max(
            0.01,
            value
        )

        df.loc[
            idx,
            "electricity_validation"
        ] = value

        df.loc[
            idx,
            "validation_anomaly_label"
        ] = 1

        df.loc[
            idx,
            "validation_anomaly_type"
        ] = "spike"

        df.loc[
            idx,
            "validation_event_id"
        ] = (
            f"VAL_SP_{event_counter:02d}"
        )

        occupied.add(
            int(idx)
        )

    # ========================================================
    # 2. PERSISTENT SHIFT
    # 2 × 12 h
    # ========================================================

    for _ in range(2):

        interval = find_interval(
            validation_indices,
            12,
            occupied,
            rng
        )

        if interval is None:

            raise RuntimeError(
                "Unable to place validation persistent shift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        shift = (
            direction
            *
            rng.uniform(
                1.25,
                2.25
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

        df.loc[
            interval,
            "electricity_validation"
        ] = values

        df.loc[
            interval,
            "validation_anomaly_label"
        ] = 1

        df.loc[
            interval,
            "validation_anomaly_type"
        ] = "persistent_shift"

        df.loc[
            interval,
            "validation_event_id"
        ] = (
            f"VAL_SH_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # 3. GRADUAL DRIFT
    # 2 × 24 h
    # ========================================================

    for _ in range(2):

        interval = find_interval(
            validation_indices,
            24,
            occupied,
            rng
        )

        if interval is None:

            raise RuntimeError(
                "Unable to place validation gradual drift."
            )

        event_counter += 1

        direction = rng.choice(
            [-1, 1]
        )

        final_shift = (
            direction
            *
            rng.uniform(
                1.75,
                2.75
            )
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
            "electricity_validation"
        ] = values

        df.loc[
            interval,
            "validation_anomaly_label"
        ] = 1

        df.loc[
            interval,
            "validation_anomaly_type"
        ] = "gradual_drift"

        df.loc[
            interval,
            "validation_event_id"
        ] = (
            f"VAL_DR_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # 4. STUCK SENSOR
    # 2 × 12 h
    # ========================================================

    for _ in range(2):

        interval = find_interval(
            validation_indices,
            12,
            occupied,
            rng
        )

        if interval is None:

            raise RuntimeError(
                "Unable to place validation stuck event."
            )

        event_counter += 1

        stuck_value = (
            df.loc[
                interval[0],
                "electricity_original"
            ]
        )

        df.loc[
            interval,
            "electricity_validation"
        ] = stuck_value

        df.loc[
            interval,
            "validation_anomaly_label"
        ] = 1

        df.loc[
            interval,
            "validation_anomaly_type"
        ] = "stuck"

        df.loc[
            interval,
            "validation_event_id"
        ] = (
            f"VAL_ST_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    # ========================================================
    # 5. CONTEXTUAL AFTER-HOURS
    # 2 × 6 h
    # ========================================================

    after_hours = np.where(
        (
            validation_mask.values
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
                "Unable to place validation contextual event."
            )

        event_counter += 1

        boost = (
            rng.uniform(
                1.25,
                2.25
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

        df.loc[
            interval,
            "electricity_validation"
        ] = values

        df.loc[
            interval,
            "validation_anomaly_label"
        ] = 1

        df.loc[
            interval,
            "validation_anomaly_type"
        ] = "contextual_after_hours"

        df.loc[
            interval,
            "validation_event_id"
        ] = (
            f"VAL_CX_{event_counter:02d}"
        )

        occupied.update(
            interval.tolist()
        )

    return df


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

summary_rows = []
type_rows = []


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
    # Reconstruct week blocks
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

    available_weeks = sorted(
        df["week_block"]
        .unique()
        .tolist()
    )

    print(
        f"\nAvailable weeks: "
        f"{available_weeks}"
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    training_mask = (
        df["week_block"]
        .isin(TRAIN_WEEKS)
    )

    validation_mask = (
        df["week_block"]
        .isin(VALIDATION_WEEKS)
    )

    final_test_mask = (
        df["week_block"]
        .isin(FINAL_TEST_WEEKS)
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assignment_count = (
        training_mask.astype(int)
        +
        validation_mask.astype(int)
        +
        final_test_mask.astype(int)
    )

    if not (
        assignment_count == 1
    ).all():

        raise RuntimeError(
            f"Split assignment error for {building}."
        )

    # --------------------------------------------------------
    # Inject validation anomalies
    # --------------------------------------------------------

    df = inject_validation_anomalies(
        df,
        validation_mask,
        training_mask,
        building_number
    )

    train_n = int(
        training_mask.sum()
    )

    val_n = int(
        validation_mask.sum()
    )

    test_n = int(
        final_test_mask.sum()
    )

    val_anomalies = int(
        df.loc[
            validation_mask,
            "validation_anomaly_label"
        ].sum()
    )

    print(
        f"\nTraining observations:   "
        f"{train_n:,}"
    )

    print(
        f"Validation observations: "
        f"{val_n:,}"
    )

    print(
        f"Final-test observations: "
        f"{test_n:,}"
    )

    print(
        f"Validation anomalies:     "
        f"{val_anomalies:,}"
    )

    print(
        f"Validation prevalence:    "
        f"{100 * val_anomalies / val_n:.2f}%"
    )

    # --------------------------------------------------------
    # Type distribution
    # --------------------------------------------------------

    distribution = (
        df.loc[
            validation_mask
            &
            (
                df[
                    "validation_anomaly_label"
                ]
                ==
                1
            ),
            "validation_anomaly_type"
        ]
        .value_counts()
    )

    print(
        "\nValidation anomaly distribution:"
    )

    print(
        distribution.to_string()
    )

    for anomaly_type, n in (
        distribution.items()
    ):

        type_rows.append({
            "building_id":
                building,

            "anomaly_type":
                anomaly_type,

            "observations":
                int(n)
        })

    # --------------------------------------------------------
    # Split label
    # --------------------------------------------------------

    df["experimental_split"] = ""

    df.loc[
        training_mask,
        "experimental_split"
    ] = "train"

    df.loc[
        validation_mask,
        "experimental_split"
    ] = "validation"

    df.loc[
        final_test_mask,
        "experimental_split"
    ] = "final_test"

    # --------------------------------------------------------
    # IMPORTANT:
    # No anomalies are injected into final test here.
    #
    # Existing final test definition remains frozen separately.
    # --------------------------------------------------------

    output_path = (
        PROCESSED /
        f"{building}_locked_protocol.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    summary_rows.append({
        "building_id":
            building,

        "training_observations":
            train_n,

        "validation_observations":
            val_n,

        "final_test_observations":
            test_n,

        "validation_anomalies":
            val_anomalies,

        "validation_prevalence":
            val_anomalies / val_n,

        "train_weeks":
            ",".join(
                map(
                    str,
                    TRAIN_WEEKS
                )
            ),

        "validation_weeks":
            ",".join(
                map(
                    str,
                    VALIDATION_WEEKS
                )
            ),

        "final_test_weeks":
            ",".join(
                map(
                    str,
                    FINAL_TEST_WEEKS
                )
            )
    })

    print(
        f"\nSaved:\n{output_path}"
    )


# ============================================================
# SAVE PROTOCOL SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    TABLES /
    "locked_experimental_protocol.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

types = pd.DataFrame(
    type_rows
)

type_path = (
    TABLES /
    "validation_anomaly_distribution.csv"
)

types.to_csv(
    type_path,
    index=False
)


# ============================================================
# SAVE MACHINE-READABLE CONFIG
# ============================================================

protocol = {
    "seed": SEED,

    "training_weeks":
        TRAIN_WEEKS,

    "validation_weeks":
        VALIDATION_WEEKS,

    "final_test_weeks":
        FINAL_TEST_WEEKS,

    "status":
        "LOCKED",

    "note":
        (
            "Final-test weeks must not be used "
            "for architecture, fusion, threshold, "
            "or hyperparameter selection."
        )
}

json_path = (
    TABLES /
    "locked_protocol.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        protocol,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("LOCKED EXPERIMENTAL PROTOCOL")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print("\n" + "=" * 100)
print("STEP 10 COMPLETE")
print("=" * 100)

print("\nSaved:")
print(summary_path)
print(type_path)
print(json_path)

print(
    "\nThe final-test partition is now LOCKED."
)

print(
    "All TwinGuard-X v3 development must use "
    "training + validation only."
)