from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
    / "five_run_per_seed_model_mean.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "tables"
    / "five_run_experiment"
)

REFERENCE = "TwinGuard-X v3"

BASELINES = [
    "Isolation Forest",
    "LOF",
    "Autoencoder",
]

METRICS = [
    "f1",
    "auprc",
    "false_positive_rate",
]

EXPECTED_SEEDS = [42, 43, 44, 45, 46]


# ============================================================
# HELPERS
# ============================================================

def holm_adjust(p_values):
    """
    Holm step-down adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0

    for i, p in enumerate(sorted_p):
        adjusted = (m - i) * p
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial_from_differences(differences):
    """
    Matched-pairs rank-biserial correlation.

    Positive value:
        TwinGuard-X has larger values than baseline.

    Negative value:
        TwinGuard-X has smaller values than baseline.

    Zero differences are removed.
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isclose(d, 0.0)]

    if len(d) == 0:
        return 0.0

    abs_d = np.abs(d)

    ranks = pd.Series(abs_d).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = ranks[d > 0].sum()
    negative_rank_sum = ranks[d < 0].sum()

    total_rank_sum = ranks.sum()

    return (
        positive_rank_sum - negative_rank_sum
    ) / total_rank_sum


def effect_label(value):
    a = abs(value)

    if a < 0.10:
        return "negligible"
    elif a < 0.30:
        return "small"
    elif a < 0.50:
        return "moderate"
    else:
        return "large"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 120)
print("EIRT 2026 — STEP 17")
print("PAIRED STATISTICAL ANALYSIS")
print("=" * 120)

print(f"\nInput:\n{INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file does not exist:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nShape:", df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", c)

required = {
    "seed",
    "model",
    *METRICS,
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# VALIDATE EXPERIMENT STRUCTURE
# ============================================================

available_seeds = sorted(
    df["seed"].astype(int).unique().tolist()
)

print("\nSeeds:", available_seeds)

if available_seeds != EXPECTED_SEEDS:
    raise RuntimeError(
        f"Expected seeds {EXPECTED_SEEDS}, "
        f"found {available_seeds}"
    )

available_models = df["model"].unique().tolist()

print("\nModels:")
for model in available_models:
    print(" ", model)

for model in [REFERENCE] + BASELINES:
    if model not in available_models:
        raise RuntimeError(
            f"Model not found: {model}"
        )

counts = (
    df.groupby(["seed", "model"])
    .size()
)

if not (counts == 1).all():
    raise RuntimeError(
        "Expected exactly one mean-performance row "
        "for each seed-model combination."
    )


# ============================================================
# PAIRED COMPARISONS
# ============================================================

results = []

for metric in METRICS:

    print("\n" + "=" * 120)
    print(f"METRIC: {metric}")
    print("=" * 120)

    metric_rows = []

    reference_df = (
        df[df["model"] == REFERENCE]
        [["seed", metric]]
        .rename(columns={metric: "reference_value"})
    )

    for baseline in BASELINES:

        baseline_df = (
            df[df["model"] == baseline]
            [["seed", metric]]
            .rename(columns={metric: "baseline_value"})
        )

        paired = (
            reference_df
            .merge(
                baseline_df,
                on="seed",
                how="inner",
                validate="one_to_one",
            )
            .sort_values("seed")
            .reset_index(drop=True)
        )

        if len(paired) != 5:
            raise RuntimeError(
                f"{metric}, {baseline}: expected "
                f"5 paired observations, found "
                f"{len(paired)}."
            )

        ref = paired["reference_value"].to_numpy(float)
        base = paired["baseline_value"].to_numpy(float)

        difference = ref - base

        # ----------------------------------------------------
        # WILCOXON SIGNED-RANK TEST
        # ----------------------------------------------------

        if np.all(np.isclose(difference, 0.0)):
            statistic = 0.0
            p_raw = 1.0
        else:
            test = wilcoxon(
                ref,
                base,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

            statistic = float(test.statistic)
            p_raw = float(test.pvalue)

        # ----------------------------------------------------
        # EFFECT SIZE
        # ----------------------------------------------------

        rbc = rank_biserial_from_differences(
            difference
        )

        row = {
            "metric": metric,
            "reference": REFERENCE,
            "baseline": baseline,
            "n_pairs": len(paired),

            "reference_mean": ref.mean(),
            "baseline_mean": base.mean(),

            "mean_difference_reference_minus_baseline":
                difference.mean(),

            "difference_sd":
                difference.std(ddof=1),

            "wilcoxon_statistic":
                statistic,

            "p_raw":
                p_raw,

            "rank_biserial":
                rbc,

            "effect_magnitude":
                effect_label(rbc),
        }

        metric_rows.append(row)

        print(
            f"{REFERENCE} vs {baseline:18s} | "
            f"mean diff={difference.mean(): .4f} | "
            f"W={statistic:.4f} | "
            f"p={p_raw:.6f} | "
            f"RBC={rbc:.4f}"
        )

    # --------------------------------------------------------
    # HOLM CORRECTION WITHIN THIS METRIC
    # --------------------------------------------------------

    adjusted = holm_adjust(
        [row["p_raw"] for row in metric_rows]
    )

    for row, p_holm in zip(
        metric_rows,
        adjusted,
    ):
        row["p_holm"] = float(p_holm)
        row["significant_0_05"] = bool(
            p_holm < 0.05
        )

        results.append(row)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df["metric_label"] = (
    results_df["metric"]
    .replace({
        "f1": "F1",
        "auprc": "AUPRC",
        "false_positive_rate": "FPR",
    })
)

results_df["direction"] = np.where(
    results_df[
        "mean_difference_reference_minus_baseline"
    ] > 0,
    "TwinGuard-X higher",
    np.where(
        results_df[
            "mean_difference_reference_minus_baseline"
        ] < 0,
        "TwinGuard-X lower",
        "No difference",
    )
)


# ============================================================
# MANUSCRIPT-READY TABLE
# ============================================================

manuscript = results_df[
    [
        "metric_label",
        "baseline",
        "mean_difference_reference_minus_baseline",
        "wilcoxon_statistic",
        "p_raw",
        "p_holm",
        "rank_biserial",
        "effect_magnitude",
        "direction",
    ]
].copy()

manuscript = manuscript.rename(
    columns={
        "metric_label": "metric",
        "mean_difference_reference_minus_baseline":
            "mean_difference",
        "wilcoxon_statistic": "W",
        "rank_biserial": "rank_biserial_effect",
    }
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 120)
print("FINAL PAIRED STATISTICAL RESULTS")
print("=" * 120)

print(
    manuscript.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print("\n" + "=" * 120)
print("COMPACT MANUSCRIPT VIEW")
print("=" * 120)

compact = manuscript[
    [
        "metric",
        "baseline",
        "mean_difference",
        "p_holm",
        "rank_biserial_effect",
        "effect_magnitude",
    ]
].copy()

print(
    compact.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE
# ============================================================

results_file = (
    OUTPUT_DIR
    / "five_run_paired_statistical_analysis.csv"
)

manuscript_file = (
    OUTPUT_DIR
    / "five_run_paired_statistical_analysis_manuscript.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)

manuscript.to_csv(
    manuscript_file,
    index=False,
)


# ============================================================
# SAVE PAIRED VALUES FOR REPRODUCIBILITY
# ============================================================

paired_rows = []

for metric in METRICS:

    for baseline in BASELINES:

        ref = (
            df[df["model"] == REFERENCE]
            [["seed", metric]]
            .rename(columns={metric: "reference_value"})
        )

        base = (
            df[df["model"] == baseline]
            [["seed", metric]]
            .rename(columns={metric: "baseline_value"})
        )

        paired = (
            ref.merge(
                base,
                on="seed",
                validate="one_to_one",
            )
            .sort_values("seed")
        )

        for _, row in paired.iterrows():

            paired_rows.append({
                "metric": metric,
                "baseline": baseline,
                "seed": int(row["seed"]),
                "reference_value":
                    row["reference_value"],
                "baseline_value":
                    row["baseline_value"],
                "difference":
                    row["reference_value"]
                    - row["baseline_value"],
            })


paired_df = pd.DataFrame(
    paired_rows
)

paired_file = (
    OUTPUT_DIR
    / "five_run_paired_statistical_values.csv"
)

paired_df.to_csv(
    paired_file,
    index=False,
)


print("\n" + "=" * 120)
print("STEP 17 COMPLETE")
print("=" * 120)

print("\nSaved:")
print(results_file)
print(manuscript_file)
print(paired_file)

print("""
Interpretation note:
- Tests are paired because models are evaluated under the
  corresponding seed-specific experimental runs.
- Holm correction is applied separately within each metric.
- Positive rank-biserial values mean TwinGuard-X has larger
  metric values; negative values mean TwinGuard-X has smaller
  values.
- For FPR, a negative difference/effect favors TwinGuard-X.
- With only five paired runs, p-values have limited resolution;
  effect sizes and consistency of paired differences should be
  interpreted alongside statistical significance.
""")