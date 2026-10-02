from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EIRT 2026 — FIGURE 2
# Five-run final-test model comparison
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
    / "five_run_FINAL_mean_sd.csv"
)

OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "figure2_overall_performance.png"

# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

print("Input columns:")
for c in df.columns:
    print(" ", c)

# ------------------------------------------------------------
# Helper: robustly obtain mean / SD columns
# ------------------------------------------------------------

def get_metric_columns(data, metric):
    mean_candidates = [
        f"{metric}_mean",
        f"{metric}_mean_value"
    ]

    sd_candidates = [
        f"{metric}_sd",
        f"{metric}_std",
        f"{metric}_sample_sd"
    ]

    mean_col = next(
        (c for c in mean_candidates if c in data.columns),
        None
    )

    sd_col = next(
        (c for c in sd_candidates if c in data.columns),
        None
    )

    # If CSV contains formatted "mean ± sd" column
    pm_col = f"{metric}_mean_pm_sd"

    if mean_col is None and pm_col in data.columns:

        means = []
        sds = []

        for value in data[pm_col].astype(str):
            parts = value.replace("+/-", "±").split("±")

            means.append(float(parts[0].strip()))

            if len(parts) > 1:
                sds.append(float(parts[1].strip()))
            else:
                sds.append(0.0)

        return np.array(means), np.array(sds)

    if mean_col is None:
        raise KeyError(
            f"Could not find mean column for metric: {metric}"
        )

    means = data[mean_col].to_numpy(dtype=float)

    if sd_col is None:
        sds = np.zeros(len(data))
    else:
        sds = data[sd_col].to_numpy(dtype=float)

    return means, sds


# ------------------------------------------------------------
# Model order
# ------------------------------------------------------------

model_order = [
    "Isolation Forest",
    "LOF",
    "Autoencoder",
    "TwinGuard-X v3"
]

df = (
    df.set_index("model")
      .reindex(model_order)
      .reset_index()
)

# Short display names
display_names = [
    "Isolation\nForest",
    "LOF",
    "Autoencoder",
    "TwinGuard-X\nv3"
]

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

metrics = [
    ("f1", "F1"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("false_positive_rate", "FPR")
]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(10.0, 2.65)
)

x = np.arange(len(model_order))

for ax, (metric, title) in zip(axes, metrics):

    means, sds = get_metric_columns(
        df,
        metric
    )

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=3,
        width=0.67,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
        pad=5
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        display_names,
        fontsize=7.5
    )

    ax.set_ylim(0, 1.0)

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

    # Numerical mean above each bar
    for bar, mean, sd in zip(
        bars,
        means,
        sds
    ):

        y = min(
            mean + sd + 0.025,
            0.96
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

axes[0].set_ylabel(
    "Performance",
    fontsize=9
)

plt.subplots_adjust(
    left=0.06,
    right=0.995,
    top=0.88,
    bottom=0.25,
    wspace=0.28
)

plt.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()

print(f"\nSaved:\n{OUTPUT}")