import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
})

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
DIMENSIONS = [
    "model", "persona_format", "question_framing",
    "system_prompt", "temperature", "few_shot",
]


def load_results(filename="pilot_results.json"):
    with open(RESULTS_DIR / filename) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=["score"])
    return df


def compute_partisan_gaps(df):
    grouped = (
        df.groupby(["spec_id", "item", "party"])["score"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["spec_id", "item"],
        columns="party",
        values="score",
    ).reset_index()
    pivot["gap"] = pivot["Democrat"] - pivot["Republican"]
    pivot["gap_positive"] = pivot["gap"] > 0
    return pivot


def specification_curve(gaps_df, item_key):
    FIGURES_DIR.mkdir(exist_ok=True)
    item_gaps = (
        gaps_df[gaps_df["item"] == item_key]
        .sort_values("gap")
        .reset_index(drop=True)
    )
    n = len(item_gaps)
    pct = item_gaps["gap_positive"].mean() * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5), height_ratios=[4, 1],
        gridspec_kw={"hspace": 0.08}, sharex=True,
    )

    colors = np.where(item_gaps["gap"] > 0, "#4a90d9", "#d94a4a")
    ax1.bar(range(n), item_gaps["gap"], color=colors, width=1.0, edgecolor="none")
    ax1.axhline(y=0, color="black", linewidth=0.6)
    median = item_gaps["gap"].median()
    ax1.axhline(
        y=median, color="#2c3e50", linewidth=1.0, linestyle="--", alpha=0.6,
    )
    ax1.set_ylabel("Partisan Gap (D $-$ R)")
    title = item_key.replace("_", " ").title()
    ax1.set_title(f"Specification Curve: {title}", fontweight="normal")
    ax1.text(
        0.97, 0.93, f"{pct:.0f}% positive",
        transform=ax1.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white",
            edgecolor="#cccccc", alpha=0.9,
        ),
    )

    ax2.bar(range(n), [1] * n, color=colors, width=1.0, edgecolor="none")
    ax2.set_yticks([])
    ax2.set_xlabel("Specification (sorted by gap size)")
    ax2.set_ylabel("Sign")

    for ax in (ax1, ax2):
        ax.set_xlim(-0.5, n - 0.5)

    fig.savefig(FIGURES_DIR / f"spec_curve_{item_key}.png")
    plt.close()
    return pct


def variance_decomposition(df):
    FIGURES_DIR.mkdir(exist_ok=True)

    results = {}
    for item in df["item"].unique():
        item_df = df[df["item"] == item]
        ss_total = ((item_df["score"] - item_df["score"].mean()) ** 2).sum()
        if ss_total == 0:
            results[item] = {d: 0.0 for d in DIMENSIONS}
            continue

        eta_sq = {}
        for dim in DIMENSIONS:
            groups = item_df.groupby(dim)["score"]
            grand_mean = item_df["score"].mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for _, g in groups
            )
            eta_sq[dim] = ss_between / ss_total
        results[item] = eta_sq

    results_df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results_df.index))
    n_dims = len(DIMENSIONS)
    width = 0.8 / n_dims
    palette = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085"]

    for i, dim in enumerate(DIMENSIONS):
        offset = (i - n_dims / 2 + 0.5) * width
        label = dim.replace("_", " ").title()
        ax.bar(
            x + offset, results_df[dim], width,
            label=label, color=palette[i], edgecolor="none",
        )

    ax.set_ylabel(r"$\eta^2$ (Variance Explained)")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [item.replace("_", " ").title() for item in results_df.index]
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
    )
    ax.set_title("Variance Decomposition by Prompt Dimension", fontweight="normal")

    fig.savefig(FIGURES_DIR / "variance_decomposition.png")
    plt.close()
    return results_df


def summary_stats(df):
    gaps = compute_partisan_gaps(df)
    stats = {}
    for item in gaps["item"].unique():
        item_gaps = gaps[gaps["item"] == item]
        stats[item] = {
            "n_specs": len(item_gaps),
            "pct_positive_gap": item_gaps["gap_positive"].mean() * 100,
            "median_gap": item_gaps["gap"].median(),
            "mean_gap": item_gaps["gap"].mean(),
            "std_gap": item_gaps["gap"].std(),
            "min_gap": item_gaps["gap"].min(),
            "max_gap": item_gaps["gap"].max(),
        }
    return pd.DataFrame(stats).T


def run_analysis(filename="pilot_results.json"):
    df = load_results(filename)
    n_total = len(df) + df["score"].isna().sum()
    parse_fail = n_total - len(df)

    print(f"Total responses: {n_total}")
    print(f"Parsed successfully: {len(df)} ({len(df)/n_total*100:.1f}%)")
    if parse_fail > 0:
        print(f"Parse failures: {parse_fail} ({parse_fail/n_total*100:.1f}%)")
    print()

    stats = summary_stats(df)
    print("=== Partisan Gap Summary ===")
    print(stats.to_string(float_format="%.2f"))
    print()

    gaps = compute_partisan_gaps(df)
    for item in df["item"].unique():
        pct = specification_curve(gaps, item)
        print(f"{item}: {pct:.0f}% of specifications preserve expected partisan direction")

    print("\n=== Variance Decomposition (eta-squared) ===")
    var_df = variance_decomposition(df)
    print(var_df.to_string(float_format="%.4f"))
    print()

    print(f"Figures saved to {FIGURES_DIR}/")
    return stats, var_df
