from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

ZIP_PATH = RAW / "bdg2_v1.zip"
SELECTION_PATH = TABLES / "selected_three_public_buildings.csv"

PROCESSED.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("EIRT 2026 — THREE-BUILDING DATASET PREPARATION")
print("=" * 90)

# ============================================================
# 1. Load selected buildings
# ============================================================

selected = pd.read_csv(SELECTION_PATH)

buildings = selected["building_id"].tolist()

site_map = dict(
    zip(
        selected["building_id"],
        selected["site_id"]
    )
)

print("\nSelected buildings:")

for building in buildings:
    print(
        f"  {building:25s} -> {site_map[building]}"
    )

# ============================================================
# 2. Locate BDG2 files
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    electricity_file = next(
        n for n in names
        if n.endswith(
            "data/meters/cleaned/electricity_cleaned.csv"
        )
    )

    weather_file = next(
        n for n in names
        if n.endswith(
            "data/weather/weather.csv"
        )
    )

    # ========================================================
    # 3. Read only selected electricity columns
    # ========================================================

    print("\nReading electricity data...")

    with z.open(electricity_file) as f:

        electricity = pd.read_csv(
            f,
            usecols=["timestamp"] + buildings
        )

    print(
        f"Electricity shape: {electricity.shape}"
    )

    # ========================================================
    # 4. Read weather
    # ========================================================

    print("Reading weather data...")

    with z.open(weather_file) as f:
        weather = pd.read_csv(f)

# ============================================================
# 5. Timestamps
# ============================================================

electricity["timestamp"] = pd.to_datetime(
    electricity["timestamp"],
    errors="coerce"
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    errors="coerce"
)

electricity = electricity.dropna(
    subset=["timestamp"]
)

weather = weather.dropna(
    subset=["timestamp"]
)

# ============================================================
# 6. Keep selected sites
# ============================================================

selected_sites = list(
    dict.fromkeys(site_map.values())
)

weather = weather[
    weather["site_id"].isin(selected_sites)
].copy()

print("\nSelected weather sites:")

for site in selected_sites:

    site_df = weather[
        weather["site_id"] == site
    ]

    print(
        f"{site:10s}: "
        f"{len(site_df):,} observations | "
        f"{site_df['timestamp'].min()} -> "
        f"{site_df['timestamp'].max()}"
    )

# ============================================================
# 7. Weather variables
# ============================================================

weather_features = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed",
    "cloudCoverage"
]

weather_features = [
    c for c in weather_features
    if c in weather.columns
]

print("\nWeather features:")
for c in weather_features:
    print(" -", c)

# ============================================================
# 8. Search common six-month windows
# ============================================================

print("\n" + "=" * 90)
print("SEARCHING COMMON SIX-MONTH WINDOW")
print("=" * 90)

candidate_starts = pd.date_range(
    start="2016-01-01",
    end="2017-06-01",
    freq="MS"
)

window_rows = []

for start in candidate_starts:

    end = start + pd.DateOffset(months=6)

    elec_window = electricity[
        (electricity["timestamp"] >= start) &
        (electricity["timestamp"] < end)
    ]

    if len(elec_window) < 4000:
        continue

    building_scores = []

    for building in buildings:

        completeness = (
            elec_window[building]
            .notna()
            .mean()
        )

        building_scores.append(completeness)

    weather_scores = []

    for site in selected_sites:

        site_weather = weather[
            (weather["site_id"] == site) &
            (weather["timestamp"] >= start) &
            (weather["timestamp"] < end)
        ]

        expected_hours = int(
            (end - start).total_seconds()
            / 3600
        )

        temporal_completeness = min(
            len(site_weather) /
            expected_hours,
            1.0
        )

        if weather_features:

            variable_completeness = (
                site_weather[weather_features]
                .notna()
                .mean()
                .mean()
            )

        else:
            variable_completeness = 0

        site_score = (
            0.50 * temporal_completeness +
            0.50 * variable_completeness
        )

        weather_scores.append(site_score)

    mean_electricity = np.mean(
        building_scores
    )

    min_electricity = np.min(
        building_scores
    )

    mean_weather = np.mean(
        weather_scores
    )

    min_weather = np.min(
        weather_scores
    )

    # Reward both average quality and worst-site quality
    score = (
        0.35 * mean_electricity +
        0.25 * min_electricity +
        0.25 * mean_weather +
        0.15 * min_weather
    )

    row = {
        "start": start,
        "end_exclusive": end,
        "observations": len(elec_window),
        "mean_electricity_completeness":
            mean_electricity,
        "min_electricity_completeness":
            min_electricity,
        "mean_weather_quality":
            mean_weather,
        "min_weather_quality":
            min_weather,
        "overall_score": score
    }

    for building, value in zip(
        buildings,
        building_scores
    ):
        row[
            f"{building}_electricity_completeness"
        ] = value

    for site, value in zip(
        selected_sites,
        weather_scores
    ):
        row[
            f"{site}_weather_quality"
        ] = value

    window_rows.append(row)

windows = pd.DataFrame(window_rows)

windows = windows.sort_values(
    "overall_score",
    ascending=False
).reset_index(drop=True)

print("\nTop 10 common windows:\n")

display_cols = [
    "start",
    "end_exclusive",
    "observations",
    "mean_electricity_completeness",
    "min_electricity_completeness",
    "mean_weather_quality",
    "min_weather_quality",
    "overall_score"
]

display = windows[
    display_cols
].head(10).copy()

numeric_cols = display.select_dtypes(
    include=np.number
).columns

display[numeric_cols] = (
    display[numeric_cols].round(4)
)

print(
    display.to_string(index=False)
)

windows.to_csv(
    TABLES /
    "common_six_month_window_ranking.csv",
    index=False
)

# ============================================================
# 9. Freeze best common window
# ============================================================

best = windows.iloc[0]

START = pd.Timestamp(
    best["start"]
)

END = pd.Timestamp(
    best["end_exclusive"]
)

print("\n" + "=" * 90)
print("SELECTED COMMON WINDOW")
print("=" * 90)

print(f"Start: {START}")
print(f"End:   {END}")
print(
    f"Overall quality score: "
    f"{best['overall_score']:.4f}"
)

# ============================================================
# 10. Build each building dataset
# ============================================================

summary_rows = []

for building in buildings:

    site = site_map[building]

    print("\n" + "-" * 90)
    print(f"PREPARING: {building}")
    print("-" * 90)

    # Electricity
    b = electricity[
        (electricity["timestamp"] >= START) &
        (electricity["timestamp"] < END)
    ][
        ["timestamp", building]
    ].copy()

    b = b.rename(
        columns={
            building: "electricity"
        }
    )

    b["electricity"] = pd.to_numeric(
        b["electricity"],
        errors="coerce"
    )

    # Weather
    w = weather[
        (weather["site_id"] == site) &
        (weather["timestamp"] >= START) &
        (weather["timestamp"] < END)
    ][
        ["timestamp"] + weather_features
    ].copy()

    # Multiple weather records at same timestamp,
    # if present, are averaged.
    w = (
        w.groupby(
            "timestamp",
            as_index=False
        )[weather_features]
        .mean()
    )

    # Merge
    df = b.merge(
        w,
        on="timestamp",
        how="left"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Missingness before cleaning
    # --------------------------------------------------------

    print("\nMissingness before cleaning (%):")

    before_missing = (
        df.isna()
        .mean()
        .mul(100)
        .round(2)
    )

    print(
        before_missing.to_string()
    )

    # --------------------------------------------------------
    # Weather interpolation
    # --------------------------------------------------------

    for c in weather_features:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        # Short gaps
        df[c] = (
            df[c]
            .interpolate(
                method="linear",
                limit=6,
                limit_direction="both"
            )
        )

        # Remaining gaps
        median = df[c].median()

        if pd.notna(median):
            df[c] = df[c].fillna(
                median
            )

    # --------------------------------------------------------
    # Electricity cleaning
    # --------------------------------------------------------

    missing_electricity_before = (
        df["electricity"]
        .isna()
        .sum()
    )

    # Only very short gaps
    df["electricity"] = (
        df["electricity"]
        .interpolate(
            method="linear",
            limit=2,
            limit_direction="both"
        )
    )

    df = df.dropna(
        subset=["electricity"]
    ).copy()

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["month"] = (
        df["timestamp"].dt.month
    )

    # Cyclic encoding
    df["hour_sin"] = np.sin(
        2 * np.pi *
        df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi *
        df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi *
        df["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # Lightweight temporal operational features
    # --------------------------------------------------------

    df["electricity_lag_1h"] = (
        df["electricity"]
        .shift(1)
    )

    df["electricity_lag_24h"] = (
        df["electricity"]
        .shift(24)
    )

    df["electricity_rolling_24h"] = (
        df["electricity"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )

    df[
        "electricity_rolling_std_24h"
    ] = (
        df["electricity"]
        .rolling(
            window=24,
            min_periods=24
        )
        .std()
    )

    # Initial 24h cannot contain lagged features
    df = df.dropna().reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Chronological 70/30 split
    # --------------------------------------------------------

    split_idx = int(
        len(df) * 0.70
    )

    df["split"] = "test"

    df.loc[
        df.index < split_idx,
        "split"
    ] = "train"

    train_n = (
        df["split"] == "train"
    ).sum()

    test_n = (
        df["split"] == "test"
    ).sum()

    # --------------------------------------------------------
    # Add building identifiers
    # --------------------------------------------------------

    df.insert(
        1,
        "building_id",
        building
    )

    df.insert(
        2,
        "site_id",
        site
    )

    # --------------------------------------------------------
    # Save individual dataset
    # --------------------------------------------------------

    output = (
        PROCESSED /
        f"{building}_experiment.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        f"\nFinal observations: {len(df):,}"
    )

    print(
        f"Train: {train_n:,}"
    )

    print(
        f"Test:  {test_n:,}"
    )

    print(
        f"Saved: {output}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    meta_row = selected[
        selected["building_id"] == building
    ].iloc[0]

    summary_rows.append({
        "building_id": building,
        "site_id": site,
        "primary_use":
            meta_row["primaryspaceusage"],
        "subtype":
            meta_row["sub_primaryspaceusage"],
        "floor_area_m2":
            meta_row["sqm"],
        "year_built":
            meta_row["yearbuilt"],
        "start":
            df["timestamp"].min(),
        "end":
            df["timestamp"].max(),
        "observations":
            len(df),
        "train_observations":
            train_n,
        "test_observations":
            test_n,
        "electricity_missing_before":
            missing_electricity_before,
        "electricity_mean":
            df["electricity"].mean(),
        "electricity_std":
            df["electricity"].std(),
        "electricity_min":
            df["electricity"].min(),
        "electricity_max":
            df["electricity"].max()
    })

# ============================================================
# 11. Save dataset summary
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    TABLES /
    "three_building_dataset_summary.csv",
    index=False
)

print("\n" + "=" * 90)
print("FINAL THREE-BUILDING SUMMARY")
print("=" * 90)

summary_display = summary.copy()

for c in [
    "floor_area_m2",
    "electricity_mean",
    "electricity_std",
    "electricity_min",
    "electricity_max"
]:
    if c in summary_display.columns:

        summary_display[c] = (
            summary_display[c]
            .round(2)
        )

print(
    summary_display.to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print("STEP 4 COMPLETE")
print("=" * 90)

print(
    "\nThree clean experimental datasets "
    "have been created."
)

print(
    "\nNext step: controlled anomaly injection "
    "into TEST data only."
)