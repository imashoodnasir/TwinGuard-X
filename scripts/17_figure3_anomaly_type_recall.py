from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EIRT 2026 — FIGURE 3
# Five-run anomaly-type recall
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
    / "five_run_anomaly_type_recall_mean_sd.csv"
)

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "figure3_anomaly_type_recall.png"

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("Shape:", df.shape)
print("\nColumns:")
for c in df.columns:
    print(" ", repr(c))

print("\nFirst rows:")
print(df.head().to_string(index=False))

# ------------------------------------------------------------
# Normalize names only for matching
# ------------------------------------------------------------

def normalize_model(x):
    x = str(x).strip()

    aliases = {
        "IsolationForest": "Isolation Forest",
        "Isolation Forest": "Isolation Forest",
        "LOF": "LOF",
        "Autoencoder": "Autoencoder",
        "TwinGuard-X v3": "TwinGuard-X v3",
        "TwinGuard-X": "TwinGuard-X v3"
    }

    return aliases.get(x, x)


df["model"] = df["model"].apply(
    normalize_model
)

# ------------------------------------------------------------
# Detect mean and SD columns
# ------------------------------------------------------------

mean_candidates = [
    "recall_mean",
    "mean_recall",
    "mean"
]

sd_candidates = [
    "recall_sd",
    "recall_std",
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

# Support formatted column if needed
if mean_col is None:

    pm_candidates = [
        "recall_mean_pm_sd",
        "mean_pm_sd"
    ]

    pm_col = next(
        (c for c in pm_candidates if c in df.columns),
        None
    )

    if pm_col is None:
        raise KeyError(
            "Could not identify recall mean/SD columns."
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
# Desired ordering
# ------------------------------------------------------------

anomaly_order = [
    "spike",
    "persistent_shift",
    "gradual_drift",
    "stuck",
    "contextual_after_hours"
]

anomaly_labels = [
    "Spike",
    "Persistent\nshift",
    "Gradual\ndrift",
    "Stuck",
    "Contextual\nafter-hours"
]

model_order = [
    "Isolation Forest",
    "LOF",
    "Autoencoder",
    "TwinGuard-X v3"
]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = np.arange(
    len(anomaly_order)
)

width = 0.19

fig, ax = plt.subplots(
    figsize=(7.2, 3.35)
)

for i, model in enumerate(model_order):

    means = []
    sds = []

    for anomaly in anomaly_order:

        subset = df[
            (df["model"] == model)
            &
            (df["anomaly_type"] == anomaly)
        ]

        if subset.empty:
            means.append(np.nan)
            sds.append(0.0)
        else:
            means.append(
                subset["_mean"].iloc[0]
            )

            sds.append(
                subset["_sd"].iloc[0]
            )

    offset = (
        i - (len(model_order) - 1) / 2
    ) * width

    ax.bar(
        x + offset,
        means,
        width,
        yerr=sds,
        capsize=2.5,
        label=model,
        edgecolor="black",
        linewidth=0.5
    )

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

ax.set_ylabel(
    "Recall",
    fontsize=10
)

ax.set_ylim(
    0,
    1.08
)

ax.set_xticks(x)

ax.set_xticklabels(
    anomaly_labels,
    fontsize=8.5
)

ax.set_yticks(
    np.arange(0, 1.01, 0.2)
)

ax.tick_params(
    axis="y",
    labelsize=8
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.35
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=4,
    frameon=False,
    fontsize=8,
    columnspacing=1.2,
    handlelength=1.4
)

plt.subplots_adjust(
    left=0.09,
    right=0.99,
    bottom=0.22,
    top=0.80
)

plt.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()

print(f"\nSaved:\n{OUTPUT}")