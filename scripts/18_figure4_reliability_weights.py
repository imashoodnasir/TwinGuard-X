from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EIRT 2026 — FIGURE 4
# Reliability-weight stability across five runs
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
    / "five_run_v3_weight_stability.csv"
)

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "figure4_reliability_weight_stability.png"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("Shape:", df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", repr(c))

print("\nContents:")
print(df.to_string(index=False))

# ------------------------------------------------------------
# Identify mean and SD columns
# ------------------------------------------------------------

mean_candidates = [
    "weight_mean",
    "mean_weight",
    "mean"
]

sd_candidates = [
    "weight_sd",
    "weight_std",
    "sd",
    "std"
]

mean_col = next(
    (c for c in mean_candidates if c in df.columns),
    None
)

sd_col = next(
    (c for c in sd_candidates if c in df.columns),
    None
)

# ------------------------------------------------------------
# Support formatted "mean ± SD" column if necessary
# ------------------------------------------------------------

if mean_col is None:

    pm_candidates = [
        "weight_mean_pm_sd",
        "mean_pm_sd"
    ]

    pm_col = next(
        (c for c in pm_candidates if c in df.columns),
        None
    )

    if pm_col is None:
        raise KeyError(
            "Could not identify reliability-weight mean/SD columns."
        )

    means = []
    sds = []

    for value in df[pm_col].astype(str):

        parts = (
            value
            .replace("+/-", "±")
            .split("±")
        )

        means.append(
            float(parts[0].strip())
        )

        sds.append(
            float(parts[1].strip())
            if len(parts) > 1
            else 0.0
        )

    df["_mean"] = means
    df["_sd"] = sds

else:

    df["_mean"] = df[mean_col].astype(float)

    if sd_col is not None:
        df["_sd"] = df[sd_col].astype(float)
    else:
        df["_sd"] = 0.0


# ------------------------------------------------------------
# Ordering exactly matching the methodology
# ------------------------------------------------------------

building_order = [
    "Robin_public_Cami",
    "Bear_public_Valorie",
    "Hog_public_Gerard"
]

component_order = [
    "contextual_residual",
    "residual_dynamics",
    "persistence_3h",
    "persistence_6h",
    "persistence_12h"
]

component_labels = [
    r"$w_b^{C*}$",
    r"$w_b^{D*}$",
    r"$w_b^{P,3*}$",
    r"$w_b^{P,6*}$",
    r"$w_b^{P,12*}$"
]

building_labels = [
    "Robin",
    "Bear",
    "Hog"
]

# ------------------------------------------------------------
# Construct matrices
# ------------------------------------------------------------

mean_matrix = np.full(
    (
        len(building_order),
        len(component_order)
    ),
    np.nan
)

sd_matrix = np.full_like(
    mean_matrix,
    np.nan
)

for i, building in enumerate(building_order):

    for j, component in enumerate(component_order):

        subset = df[
            (df["building_id"] == building)
            &
            (df["component"] == component)
        ]

        if not subset.empty:

            mean_matrix[i, j] = (
                subset["_mean"].iloc[0]
            )

            sd_matrix[i, j] = (
                subset["_sd"].iloc[0]
            )

print("\nMean matrix:")
print(mean_matrix)

print("\nSD matrix:")
print(sd_matrix)

# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.0, 2.65)
)

im = ax.imshow(
    mean_matrix,
    cmap="viridis",
    aspect="auto",
    vmin=0,
    vmax=1
)

# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

ax.set_xticks(
    np.arange(len(component_labels))
)

ax.set_xticklabels(
    component_labels,
    fontsize=11
)

ax.set_yticks(
    np.arange(len(building_labels))
)

ax.set_yticklabels(
    building_labels,
    fontsize=9
)

# ------------------------------------------------------------
# Cell annotations
# Explicit text colors for maximum readability
# ------------------------------------------------------------

for i in range(mean_matrix.shape[0]):
    for j in range(mean_matrix.shape[1]):

        mean = mean_matrix[i, j]
        sd = sd_matrix[i, j]

        if np.isnan(mean):
            label = "N/A"
            text_color = "white"
        else:
            label = f"{mean:.3f}\n± {sd:.3f}"

            # Black text ONLY on sufficiently bright cells.
            # Everything else gets white text.
            if mean >= 0.45:
                text_color = "black"
            else:
                text_color = "white"

        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=text_color
        )

# ------------------------------------------------------------
# Cell borders
# ------------------------------------------------------------

ax.set_xticks(
    np.arange(
        -0.5,
        len(component_order),
        1
    ),
    minor=True
)

ax.set_yticks(
    np.arange(
        -0.5,
        len(building_order),
        1
    ),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=1.0,
    alpha=0.70
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

# ------------------------------------------------------------
# Color bar
# ------------------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.025
)

cbar.set_label(
    "Reliability weight",
    fontsize=9
)

cbar.ax.tick_params(
    labelsize=8
)

# ------------------------------------------------------------
# Clean appearance
# ------------------------------------------------------------

for spine in ax.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(
    left=0.12,
    right=0.91,
    top=0.94,
    bottom=0.18
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()

print(f"\nSaved:\n{OUTPUT}")