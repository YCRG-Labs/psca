import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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


def _compute_spec_gaps_with_pvalues(df):
    rows = []
    for (spec_id, item), group in df.groupby(["spec_id", "item"]):
        d = group[group["party"] == "Democrat"]["score"]
        r = group[group["party"] == "Republican"]["score"]
        gap = d.mean() - r.mean()
        if len(d) > 1 and len(r) > 1:
            _, p_val = scipy_stats.ttest_ind(d, r)
        else:
            p_val = 1.0
        rows.append({
            "spec_id": spec_id, "item": item,
            "gap": gap, "p_value": p_val,
        })
    return pd.DataFrame(rows)


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


def _compute_eta_sq(item_df):
    ss_total = ((item_df["score"] - item_df["score"].mean()) ** 2).sum()
    if ss_total == 0:
        return {d: 0.0 for d in DIMENSIONS}
    eta_sq = {}
    for dim in DIMENSIONS:
        groups = item_df.groupby(dim)["score"]
        grand_mean = item_df["score"].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for _, g in groups
        )
        eta_sq[dim] = ss_between / ss_total
    return eta_sq


def bootstrap_ci(df, n_boot=1000, ci=0.95, seed=42):
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    results = {}
    for item in df["item"].unique():
        item_df = df[df["item"] == item]
        spec_ids = item_df["spec_id"].unique()
        observed = _compute_eta_sq(item_df)

        boot_samples = {d: [] for d in DIMENSIONS}
        for _ in range(n_boot):
            resampled_ids = rng.choice(spec_ids, size=len(spec_ids), replace=True)
            boot_df = pd.concat(
                [item_df[item_df["spec_id"] == sid] for sid in resampled_ids],
                ignore_index=True,
            )
            boot_eta = _compute_eta_sq(boot_df)
            for d in DIMENSIONS:
                boot_samples[d].append(boot_eta[d])

        item_results = {}
        for d in DIMENSIONS:
            samples = np.array(boot_samples[d])
            item_results[d] = {
                "eta_sq": observed[d],
                "ci_lo": np.quantile(samples, alpha),
                "ci_hi": np.quantile(samples, 1 - alpha),
            }
        results[item] = item_results

        print(f"\n=== Bootstrap 95% CI: {item} ===")
        for d in DIMENSIONS:
            r = item_results[d]
            print(f"  {d:20s}: η² = {r['eta_sq']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    items = list(results.keys())
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 5), sharey=True)
    if len(items) == 1:
        axes = [axes]
    palette = ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085"]

    for ax, item in zip(axes, items):
        vals = [results[item][d]["eta_sq"] for d in DIMENSIONS]
        los = [results[item][d]["eta_sq"] - results[item][d]["ci_lo"] for d in DIMENSIONS]
        his = [results[item][d]["ci_hi"] - results[item][d]["eta_sq"] for d in DIMENSIONS]
        x = np.arange(len(DIMENSIONS))
        ax.bar(x, vals, color=palette, edgecolor="none",
               yerr=[los, his], capsize=4, error_kw={"linewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_", " ").title() for d in DIMENSIONS],
            rotation=35, ha="right", fontsize=9,
        )
        ax.set_title(item.replace("_", " ").title(), fontweight="normal")
        ax.set_ylabel(r"$\eta^2$")

    plt.suptitle("Variance Decomposition with 95% Bootstrap CI", fontweight="normal", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_decomposition_ci.png")
    plt.close()

    return results


def variance_decomposition(df):
    FIGURES_DIR.mkdir(exist_ok=True)

    results = {}
    for item in df["item"].unique():
        item_df = df[df["item"] == item]
        results[item] = _compute_eta_sq(item_df)

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


def sobol_analysis(df, problem, n_base, calc_second_order=False):
    from SALib.analyze import sobol

    FIGURES_DIR.mkdir(exist_ok=True)
    results = {}

    for item in df["item"].unique():
        item_df = df[df["item"] == item]
        gaps = compute_partisan_gaps(item_df)
        gaps = gaps.sort_values("spec_id")
        Y = gaps["gap"].values

        expected_n = n_base * (2 * problem["num_vars"] + 2) if calc_second_order else n_base * (problem["num_vars"] + 2)
        if len(Y) != expected_n:
            print(f"WARNING: {item} has {len(Y)} specs, expected {expected_n}. Skipping Sobol.")
            continue

        Si = sobol.analyze(
            problem, Y,
            calc_second_order=calc_second_order,
            conf_level=0.95,
        )
        results[item] = Si

        s1 = pd.DataFrame({
            "dimension": problem["names"],
            "S1": Si["S1"],
            "S1_conf": Si["S1_conf"],
            "ST": Si["ST"],
            "ST_conf": Si["ST_conf"],
        })

        print(f"\n=== Sobol Indices: {item} ===")
        print(s1.to_string(index=False, float_format="%.4f"))

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(s1))
        w = 0.35
        ax.bar(x - w/2, s1["S1"], w, label="First-order ($S_1$)",
               color="#2980b9", edgecolor="none",
               yerr=s1["S1_conf"], capsize=3)
        ax.bar(x + w/2, s1["ST"], w, label="Total-order ($S_T$)",
               color="#c0392b", edgecolor="none",
               yerr=s1["ST_conf"], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d.replace("_", " ").title() for d in s1["dimension"]],
            rotation=30, ha="right",
        )
        ax.set_ylabel("Sobol Index")
        ax.set_title(
            f"Sobol Sensitivity Indices: {item.replace('_', ' ').title()}",
            fontweight="normal",
        )
        ax.legend()
        fig.savefig(FIGURES_DIR / f"sobol_{item}.png")
        plt.close()

    return results


def permutation_inference(df, n_permutations=500, seed=42):
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    obs = _compute_spec_gaps_with_pvalues(df)
    results = {}

    for item in obs["item"].unique():
        item_obs = obs[obs["item"] == item]
        obs_median = item_obs["gap"].median()
        obs_share_pos_sig = (
            (item_obs["gap"] > 0) & (item_obs["p_value"] < 0.05)
        ).mean()

        item_df = df[df["item"] == item].copy()

        null_medians = np.zeros(n_permutations)
        null_shares = np.zeros(n_permutations)

        for b in range(n_permutations):
            perm_df = item_df.copy()
            for sid in perm_df["spec_id"].unique():
                mask = perm_df["spec_id"] == sid
                parties = perm_df.loc[mask, "party"].values.copy()
                rng.shuffle(parties)
                perm_df.loc[mask, "party"] = parties

            perm_gaps = _compute_spec_gaps_with_pvalues(perm_df)
            perm_item = perm_gaps[perm_gaps["item"] == item]
            null_medians[b] = perm_item["gap"].median()
            null_shares[b] = (
                (perm_item["gap"] > 0) & (perm_item["p_value"] < 0.05)
            ).mean()

        p_median = (np.abs(null_medians) >= np.abs(obs_median)).mean()
        p_share = (null_shares >= obs_share_pos_sig).mean()

        results[item] = {
            "obs_median_gap": obs_median,
            "obs_share_pos_sig": obs_share_pos_sig,
            "p_median": p_median,
            "p_share_pos_sig": p_share,
            "null_medians": null_medians,
        }

        print(f"\n=== Permutation Inference: {item} ===")
        print(f"  Observed median gap: {obs_median:.3f}")
        print(f"  Observed share positive & significant: {obs_share_pos_sig:.3f}")
        print(f"  p-value (median): {p_median:.4f}")
        print(f"  p-value (share): {p_share:.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_medians, bins=30, color="#bdc3c7", edgecolor="white", alpha=0.8)
        ax.axvline(obs_median, color="#c0392b", linewidth=2, label=f"Observed ({obs_median:.2f})")
        ax.set_xlabel("Median Partisan Gap")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Permutation Null: {item.replace('_', ' ').title()} (p = {p_median:.3f})",
            fontweight="normal",
        )
        ax.legend()
        fig.savefig(FIGURES_DIR / f"permutation_{item}.png")
        plt.close()

    return results


def flipped_spec_analysis(df, item_key="gun_control"):
    FIGURES_DIR.mkdir(exist_ok=True)
    gaps = compute_partisan_gaps(df)
    item_gaps = gaps[gaps["item"] == item_key].copy()
    item_gaps["flipped"] = item_gaps["gap"] <= 0

    flipped_ids = item_gaps[item_gaps["flipped"]]["spec_id"].tolist()
    if not flipped_ids:
        print(f"No flipped specifications for {item_key}")
        return None

    item_df = df[df["item"] == item_key]
    all_specs = item_df.drop_duplicates("spec_id")
    flipped_specs = item_df[item_df["spec_id"].isin(flipped_ids)].drop_duplicates("spec_id")

    print(f"\n=== Flipped Specification Analysis: {item_key} ===")
    print(f"Total specs: {len(all_specs)}, Flipped: {len(flipped_specs)}")

    for dim in DIMENSIONS:
        all_dist = all_specs[dim].value_counts(normalize=True)
        flip_dist = flipped_specs[dim].value_counts(normalize=True)
        overrep = (flip_dist / all_dist).dropna().sort_values(ascending=False)
        top = overrep.index[0] if len(overrep) > 0 else "N/A"
        ratio = overrep.iloc[0] if len(overrep) > 0 else 0
        if ratio > 1.5:
            print(f"  {dim}: '{top}' overrepresented {ratio:.1f}x in flipped specs")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, dim in enumerate(DIMENSIONS):
        ax = axes[i]
        all_counts = all_specs[dim].value_counts(normalize=True).sort_index()
        flip_counts = flipped_specs[dim].value_counts(normalize=True).reindex(all_counts.index, fill_value=0)

        x = np.arange(len(all_counts))
        w = 0.35
        ax.bar(x - w/2, all_counts.values, w, label="All", color="#bdc3c7", edgecolor="none")
        ax.bar(x + w/2, flip_counts.values, w, label="Flipped", color="#c0392b", edgecolor="none")
        ax.set_xticks(x)
        labels = [str(v)[:12] for v in all_counts.index]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(dim.replace("_", " ").title(), fontsize=10, fontweight="normal")
        ax.set_ylabel("Proportion")
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(
        f"Flipped Specs vs All: {item_key.replace('_', ' ').title()}",
        fontweight="normal", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"flipped_analysis_{item_key}.png")
    plt.close()
    return flipped_ids


def anes_benchmark(df, anes_path=None):
    if anes_path is None:
        anes_path = Path("data/anes_timeseries_2024.csv")
    if not anes_path.exists():
        print(f"ANES data not found at {anes_path}")
        print("Download from https://electionstudies.org/data-center/2024-time-series-study/")
        return None

    FIGURES_DIR.mkdir(exist_ok=True)
    anes = pd.read_csv(anes_path, low_memory=False)

    PARTY_VAR = "V241227x"
    ITEM_MAP = {
        "gov_spending": {"var": "V241239", "scale": 7, "valid_range": (1, 7), "flip": False},
        "immigration": {"var": "V241747", "scale": 5, "valid_range": (1, 5), "flip": True},
        "gun_control": {"var": "V242325", "scale": 3, "valid_range": (1, 3), "flip": True},
    }

    anes_party = anes[PARTY_VAR]
    anes["party_simple"] = np.where(
        anes_party.between(1, 3), "Democrat",
        np.where(anes_party.between(5, 7), "Republican", "Independent")
    )
    anes = anes[anes["party_simple"].isin(["Democrat", "Republican"])]

    results = {}
    for item_key, info in ITEM_MAP.items():
        var = info["var"]
        if var not in anes.columns:
            continue
        lo, hi = info["valid_range"]
        valid = anes[anes[var].between(lo, hi)].copy()
        if len(valid) == 0:
            continue

        if info.get("flip"):
            valid[var] = (lo + hi) - valid[var]

        anes_gap = (
            valid[valid["party_simple"] == "Democrat"][var].mean()
            - valid[valid["party_simple"] == "Republican"][var].mean()
        )

        llm_df = df[df["item"] == item_key] if item_key in df["item"].values else None
        if llm_df is None:
            continue

        llm_gaps = compute_partisan_gaps(llm_df)
        llm_median = llm_gaps["gap"].median()
        llm_mean = llm_gaps["gap"].mean()

        scale_factor = info["scale"] / 5.0 if info["scale"] != 5 else 1.0

        results[item_key] = {
            "anes_gap": anes_gap,
            "anes_scale": info["scale"],
            "llm_median_gap": llm_median,
            "llm_mean_gap": llm_mean,
            "llm_gap_rescaled": llm_mean * scale_factor,
        }

        print(f"\n=== ANES Benchmark: {item_key} ===")
        print(f"  ANES D-R gap (1-{info['scale']} scale): {anes_gap:.3f}")
        print(f"  LLM median gap (1-5 scale): {llm_median:.3f}")
        print(f"  LLM mean gap (1-5 scale): {llm_mean:.3f}")

    if results:
        items = list(results.keys())
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(items))
        w = 0.3
        anes_vals = [results[i]["anes_gap"] for i in items]
        llm_vals = [results[i]["llm_gap_rescaled"] for i in items]
        ax.bar(x - w/2, anes_vals, w, label="ANES 2024", color="#2980b9", edgecolor="none")
        ax.bar(x + w/2, llm_vals, w, label="LLM (rescaled)", color="#c0392b", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("_", " ").title() for i in items])
        ax.set_ylabel("Partisan Gap (D $-$ R)")
        ax.set_title("ANES vs LLM Partisan Gaps", fontweight="normal")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
        fig.savefig(FIGURES_DIR / "anes_benchmark.png")
        plt.close()

    return results


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
