from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"

ZIP_PATH = RAW / "bdg2_v1.zip"
RANKING_PATH = TABLES / "candidate_building_ranking_preliminary.csv"

TABLES.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("EIRT 2026 — THREE PUBLIC BUILDING SELECTION")
print("=" * 90)

# ============================================================
# 1. Load preliminary building ranking
# ============================================================

ranking = pd.read_csv(RANKING_PATH)

print(f"\nBuildings in preliminary ranking: {len(ranking):,}")

# Keep Public Services only
public = ranking[
    ranking["primaryspaceusage"] == "Public services"
].copy()

print(f"Public Services buildings: {len(public):,}")

# Basic electricity quality requirements
public = public[
    (public["electricity_completeness"] >= 0.95) &
    (public["positive_fraction"] >= 0.90) &
    (public["electricity_std"] > 0)
].copy()

print(
    f"After electricity quality filtering: {len(public):,}"
)

# ============================================================
# 2. Read weather data
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    names = z.namelist()

    weather_file = next(
        n for n in names
        if n.endswith("data/weather/weather.csv")
    )

    print("\nReading weather data...")

    with z.open(weather_file) as f:
        weather = pd.read_csv(f)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    errors="coerce"
)

weather = weather.dropna(
    subset=["timestamp", "site_id"]
)

print(f"Weather observations: {len(weather):,}")

# ============================================================
# 3. Weather quality by site
# ============================================================

weather_features = [
    "airTemperature",
    "dewTemperature",
    "seaLvlPressure",
    "windSpeed"
]

weather_features = [
    c for c in weather_features
    if c in weather.columns
]

site_rows = []

for site, group in weather.groupby("site_id"):

    start = group["timestamp"].min()
    end = group["timestamp"].max()

    if weather_features:

        completeness = (
            group[weather_features]
            .notna()
            .mean()
            .mean()
        )

    else:
        completeness = 0

    site_rows.append({
        "site_id": site,
        "weather_start": start,
        "weather_end": end,
        "weather_observations": len(group),
        "weather_completeness": completeness
    })

weather_quality = pd.DataFrame(site_rows)

print("\nWeather quality by site:\n")

weather_display = weather_quality.copy()

weather_display["weather_completeness"] = (
    weather_display["weather_completeness"]
    .round(4)
)

print(
    weather_display
    .sort_values(
        "weather_completeness",
        ascending=False
    )
    .to_string(index=False)
)

# ============================================================
# 4. Merge weather quality into candidate ranking
# ============================================================

public = public.merge(
    weather_quality,
    on="site_id",
    how="left"
)

public = public.dropna(
    subset=["weather_completeness"]
)

# Require reasonably complete weather
public = public[
    public["weather_completeness"] >= 0.90
].copy()

print(
    f"\nCandidates after weather filtering: "
    f"{len(public):,}"
)

# ============================================================
# 5. Calculate final building score
# ============================================================

public["selection_score"] = (
    0.50 * public["electricity_completeness"]
    +
    0.25 * public["positive_fraction"]
    +
    0.20 * public["weather_completeness"]
    +
    0.05 * public["metadata_completeness"]
)

public = public.sort_values(
    "selection_score",
    ascending=False
)

# ============================================================
# 6. Best candidate from each site
# ============================================================

best_per_site = (
    public
    .sort_values(
        "selection_score",
        ascending=False
    )
    .groupby(
        "site_id",
        as_index=False
    )
    .first()
)

best_per_site = best_per_site.sort_values(
    "selection_score",
    ascending=False
)

print("\n" + "=" * 90)
print("BEST PUBLIC-SERVICE BUILDING FROM EACH SITE")
print("=" * 90)

cols = [
    "building_id",
    "site_id",
    "sub_primaryspaceusage",
    "sqm",
    "yearbuilt",
    "electricity_completeness",
    "positive_fraction",
    "electricity_std",
    "weather_completeness",
    "metadata_completeness",
    "selection_score"
]

cols = [
    c for c in cols
    if c in best_per_site.columns
]

display = best_per_site[cols].copy()

for c in [
    "electricity_completeness",
    "positive_fraction",
    "weather_completeness",
    "metadata_completeness",
    "selection_score"
]:
    if c in display.columns:
        display[c] = display[c].round(4)

print(
    display.to_string(index=False)
)

# ============================================================
# 7. Select top three DIFFERENT sites
# ============================================================

selected = best_per_site.head(3).copy()

if len(selected) < 3:

    raise RuntimeError(
        "Fewer than three suitable sites were found."
    )

print("\n" + "=" * 90)
print("SELECTED THREE BUILDINGS")
print("=" * 90)

selected_display = selected[cols].copy()

for c in [
    "electricity_completeness",
    "positive_fraction",
    "weather_completeness",
    "metadata_completeness",
    "selection_score"
]:
    if c in selected_display.columns:
        selected_display[c] = (
            selected_display[c].round(4)
        )

print(
    selected_display.to_string(index=False)
)

# ============================================================
# 8. Check subtype diversity
# ============================================================

print("\nSubtypes:")

for _, row in selected.iterrows():

    print(
        f"{row['building_id']} | "
        f"{row['site_id']} | "
        f"{row.get('sub_primaryspaceusage', 'Unknown')}"
    )

# ============================================================
# 9. Save selections
# ============================================================

output = (
    TABLES /
    "selected_three_public_buildings.csv"
)

selected.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\n" + "=" * 90)
print("STEP 3 COMPLETE")
print("=" * 90)