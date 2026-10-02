from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import io

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

ZIP_PATH = RAW / "bdg2_v1.zip"
METADATA_PATH = RAW / "bdg2_metadata.csv"

print("=" * 90)
print("BDG2 BUILDING SELECTION")
print("=" * 90)

# -------------------------------------------------------------------
# 1. Load metadata
# -------------------------------------------------------------------

metadata = pd.read_csv(METADATA_PATH)

print(f"\nMetadata buildings: {len(metadata):,}")

# Keep buildings with electricity
candidates = metadata[
    metadata["electricity"]
    .astype(str)
    .str.lower()
    .eq("yes")
].copy()

# Preferred building categories
preferred = [
    "Public services",
    "Education",
    "Office"
]

candidates = candidates[
    candidates["primaryspaceusage"].isin(preferred)
].copy()

print(
    f"Electricity-enabled Public/Education/Office candidates: "
    f"{len(candidates):,}"
)

print("\nCandidate categories:")

print(
    candidates["primaryspaceusage"]
    .value_counts()
    .to_string()
)

# -------------------------------------------------------------------
# 2. Inspect ZIP structure
# -------------------------------------------------------------------

print("\n" + "=" * 90)
print("ARCHIVE STRUCTURE")
print("=" * 90)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    csv_files = [
        n for n in names
        if n.lower().endswith(".csv")
    ]

    print(f"\nCSV files inside archive: {len(csv_files)}")

    print("\nRelevant archive files:")

    relevant = [
        n for n in csv_files
        if any(
            key in n.lower()
            for key in [
                "electric",
                "weather",
                "meter"
            ]
        )
    ]

    for n in relevant:
        info = z.getinfo(n)

        print(
            f"{n} "
            f"({info.file_size / 1024**2:.2f} MB)"
        )

# -------------------------------------------------------------------
# 3. Find electricity file
# -------------------------------------------------------------------

electricity_files = [
    n for n in csv_files
    if "electricity" in n.lower()
]

if not electricity_files:

    electricity_files = [
        n for n in csv_files
        if "electric" in n.lower()
    ]

if not electricity_files:

    raise RuntimeError(
        "Could not identify electricity CSV inside archive."
    )

print("\nElectricity candidate files:")

for n in electricity_files:
    print(" ", n)

# Prefer cleaned electricity file if there are several
electricity_file = sorted(
    electricity_files,
    key=lambda x: (
        "cleaned" not in x.lower(),
        len(x)
    )
)[0]

print("\nSelected electricity file:")
print(electricity_file)

# -------------------------------------------------------------------
# 4. Find weather file(s)
# -------------------------------------------------------------------

weather_files = [
    n for n in csv_files
    if "weather" in n.lower()
]

print("\nWeather candidate files:")

for n in weather_files:
    print(" ", n)

if not weather_files:
    raise RuntimeError(
        "Could not identify weather CSV inside archive."
    )

# -------------------------------------------------------------------
# 5. Read electricity data
# -------------------------------------------------------------------

print("\n" + "=" * 90)
print("READING ELECTRICITY DATA")
print("=" * 90)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    with z.open(electricity_file) as f:
        electricity = pd.read_csv(f)

print("\nElectricity shape:")
print(electricity.shape)

print("\nFirst columns:")
print(list(electricity.columns[:20]))

print("\nFirst rows:")
print(electricity.head(3).to_string())

# -------------------------------------------------------------------
# 6. Detect timestamp
# -------------------------------------------------------------------

possible_time_cols = [
    "timestamp",
    "datetime",
    "date",
    "time"
]

time_col = next(
    (
        c for c in possible_time_cols
        if c in electricity.columns
    ),
    None
)

if time_col is None:
    time_col = electricity.columns[0]

print(f"\nDetected timestamp column: {time_col}")

electricity[time_col] = pd.to_datetime(
    electricity[time_col],
    errors="coerce"
)

electricity = electricity.dropna(
    subset=[time_col]
)

electricity = electricity.sort_values(
    time_col
)

print("\nElectricity time range:")

print(
    electricity[time_col].min(),
    "to",
    electricity[time_col].max()
)

# -------------------------------------------------------------------
# 7. Determine candidate building columns
# -------------------------------------------------------------------

candidate_ids = set(
    candidates["building_id"]
    .astype(str)
)

building_columns = [
    c for c in electricity.columns
    if str(c) in candidate_ids
]

print(
    f"\nCandidate buildings found in electricity file: "
    f"{len(building_columns)}"
)

if not building_columns:
    print(
        "\nNo direct building_id matches detected."
    )

    print(
        "Example metadata IDs:",
        list(candidate_ids)[:10]
    )

    print(
        "Example electricity columns:",
        list(electricity.columns[1:11])
    )

    raise RuntimeError(
        "Building IDs do not match electricity columns."
    )

# -------------------------------------------------------------------
# 8. Electricity completeness
# -------------------------------------------------------------------

print("\nCalculating electricity completeness...")

rows = []

total_hours = len(electricity)

for building in building_columns:

    series = pd.to_numeric(
        electricity[building],
        errors="coerce"
    )

    valid = series.notna().sum()

    completeness = (
        valid / total_hours
        if total_hours
        else 0
    )

    positive = (
        (series > 0).sum() / valid
        if valid
        else 0
    )

    nonzero_std = (
        float(series.std())
        if valid > 1
        else 0
    )

    rows.append(
        {
            "building_id": building,
            "electricity_valid_hours": valid,
            "electricity_completeness": completeness,
            "positive_fraction": positive,
            "electricity_std": nonzero_std
        }
    )

quality = pd.DataFrame(rows)

# -------------------------------------------------------------------
# 9. Merge metadata
# -------------------------------------------------------------------

candidates["building_id"] = (
    candidates["building_id"].astype(str)
)

ranking = quality.merge(
    candidates,
    on="building_id",
    how="left"
)

# -------------------------------------------------------------------
# 10. Metadata completeness
# -------------------------------------------------------------------

useful_metadata = [
    "sqm",
    "yearbuilt",
    "numberoffloors",
    "occupants",
    "eui",
    "site_eui"
]

available_meta = [
    c for c in useful_metadata
    if c in ranking.columns
]

if available_meta:

    ranking["metadata_completeness"] = (
        ranking[available_meta]
        .notna()
        .mean(axis=1)
    )

else:

    ranking["metadata_completeness"] = 0

# -------------------------------------------------------------------
# 11. Category priority
# -------------------------------------------------------------------

category_score = {
    "Public services": 1.00,
    "Education": 0.90,
    "Office": 0.80
}

ranking["category_score"] = (
    ranking["primaryspaceusage"]
    .map(category_score)
    .fillna(0)
)

# -------------------------------------------------------------------
# 12. Basic electricity quality score
# -------------------------------------------------------------------

ranking["electricity_quality"] = (
    0.75 * ranking["electricity_completeness"]
    +
    0.25 * ranking["positive_fraction"]
)

ranking["preliminary_score"] = (
    0.70 * ranking["electricity_quality"]
    +
    0.15 * ranking["metadata_completeness"]
    +
    0.15 * ranking["category_score"]
)

# Remove nearly constant signals
ranking = ranking[
    ranking["electricity_std"] > 0
].copy()

ranking = ranking.sort_values(
    "preliminary_score",
    ascending=False
)

# -------------------------------------------------------------------
# 13. Print top 30
# -------------------------------------------------------------------

print("\n" + "=" * 90)
print("TOP 30 BUILDINGS — PRELIMINARY RANKING")
print("=" * 90)

display_cols = [
    "building_id",
    "site_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "yearbuilt",
    "electricity_valid_hours",
    "electricity_completeness",
    "positive_fraction",
    "metadata_completeness",
    "preliminary_score"
]

display_cols = [
    c for c in display_cols
    if c in ranking.columns
]

top30 = ranking[
    display_cols
].head(30).copy()

for col in [
    "electricity_completeness",
    "positive_fraction",
    "metadata_completeness",
    "preliminary_score"
]:
    if col in top30.columns:
        top30[col] = top30[col].round(4)

print(
    top30.to_string(
        index=False
    )
)

# -------------------------------------------------------------------
# 14. Save ranking
# -------------------------------------------------------------------

ranking_path = (
    TABLES /
    "candidate_building_ranking_preliminary.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)

print("\nFull preliminary ranking saved:")
print(ranking_path)

# -------------------------------------------------------------------
# 15. Inspect weather structure
# -------------------------------------------------------------------

print("\n" + "=" * 90)
print("WEATHER FILE INSPECTION")
print("=" * 90)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for weather_file in weather_files:

        print("\nFILE:")
        print(weather_file)

        try:

            with z.open(weather_file) as f:

                sample = pd.read_csv(
                    f,
                    nrows=5
                )

            print("Columns:")
            print(list(sample.columns))

            print("Sample:")
            print(sample.head(2).to_string())

        except Exception as e:

            print(
                "Could not inspect:",
                str(e)
            )

print("\n" + "=" * 90)
print("STEP 2 COMPLETE")
print("=" * 90)