import asyncio
import argparse
import time

from config import (
    PROFILES, ANES_ITEMS, REPEATS,
    COST_PER_1M_INPUT, COST_PER_1M_OUTPUT,
    AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
)
from sampler import generate_specifications, generate_saltelli_specifications
from prompts import build_prompt
from runner import run_batch, save_results
from analysis import (
    run_analysis, sobol_analysis, permutation_inference,
    flipped_spec_analysis, anes_benchmark, bootstrap_ci, load_results,
    fisher_rz_dimension_test, derive_coverage_threshold,
    profile_jackknife, system_decomposition, hierarchical_system_decomp,
)


def build_tasks(specifications, items=None, repeats=REPEATS):
    if items is None:
        items = list(ANES_ITEMS.keys())

    tasks = []
    for spec in specifications:
        for item_key in items:
            for profile in PROFILES:
                for r in range(repeats):
                    prompt = build_prompt(spec, profile, item_key)
                    tasks.append({
                        "spec_id": spec["spec_id"],
                        "profile_id": profile["id"],
                        "party": profile["party"],
                        "item": item_key,
                        "repeat": r,
                        "persona_format": spec["persona_format"],
                        "question_framing": spec["question_framing"],
                        "system_prompt": spec["system_prompt"],
                        "few_shot": spec["few_shot"],
                        "prompt": prompt,
                    })
    return tasks


def estimate_cost(tasks):
    model_counts = {}
    for t in tasks:
        m = t["prompt"]["model"]
        model_counts[m] = model_counts.get(m, 0) + 1

    total = 0.0
    print("\n--- Cost Estimate ---")
    for model, count in sorted(model_counts.items()):
        input_cost = (count * AVG_INPUT_TOKENS / 1_000_000) * COST_PER_1M_INPUT.get(model, 3.0)
        output_cost = (count * AVG_OUTPUT_TOKENS / 1_000_000) * COST_PER_1M_OUTPUT.get(model, 15.0)
        subtotal = input_cost + output_cost
        total += subtotal
        print(f"  {model}: {count:,} calls → ${subtotal:.2f}")
    print(f"  TOTAL: ${total:.2f}")
    return total


async def run_lhs(args):
    only_models = getattr(args, "only_models", None)
    spec_id_offset = getattr(args, "spec_id_offset", 0)
    print(f"Generating {args.n_specs} LHS specifications (seed={args.seed}"
          f"{', only_models=' + str(only_models) if only_models else ''}"
          f"{', offset=' + str(spec_id_offset) if spec_id_offset else ''})...")
    specs = generate_specifications(
        args.n_specs, seed=args.seed,
        only_models=only_models, spec_id_offset=spec_id_offset,
    )

    items = args.items or list(ANES_ITEMS.keys())
    print(f"Items: {', '.join(items)}")
    print(f"Profiles: {len(PROFILES)}")
    print(f"Repeats: {args.repeats}")

    tasks = build_tasks(specs, items=args.items, repeats=args.repeats)
    print(f"Total API calls: {len(tasks):,}")
    estimate_cost(tasks)

    if args.dry_run:
        print("\n[dry run]")
        return

    print(f"\nRunning with max {args.max_concurrent} concurrent requests...")
    start = time.time()
    results = await run_batch(tasks, max_concurrent=args.max_concurrent)
    elapsed = time.time() - start
    print(f"\nCompleted {len(results):,} calls in {elapsed:.0f}s")

    path = save_results(results, args.output)
    print(f"Results saved to {path}")

    print("\n" + "=" * 60)
    run_analysis(args.output)


async def run_saltelli(args):
    n_base = args.saltelli_n
    calc_second = args.second_order
    print(f"Generating Saltelli specifications (N={n_base}, second_order={calc_second})...")
    specs, problem, total = generate_saltelli_specifications(
        n_base=n_base, calc_second_order=calc_second, seed=args.seed,
    )
    print(f"Total specifications: {total}")

    items = args.items or list(ANES_ITEMS.keys())
    print(f"Items: {', '.join(items)}")

    tasks = build_tasks(specs, items=items, repeats=args.repeats)
    print(f"Total API calls: {len(tasks):,}")
    estimate_cost(tasks)

    if args.dry_run:
        print("\n[dry run]")
        return

    print(f"\nRunning with max {args.max_concurrent} concurrent requests...")
    start = time.time()
    results = await run_batch(tasks, max_concurrent=args.max_concurrent)
    elapsed = time.time() - start
    print(f"\nCompleted {len(results):,} calls in {elapsed:.0f}s")

    path = save_results(results, args.output)
    print(f"Results saved to {path}")

    print("\n" + "=" * 60)
    run_analysis(args.output)

    print("\n" + "=" * 60)
    df = load_results(args.output)
    sobol_analysis(df, problem, n_base, calc_second_order=calc_second)


async def run_permutation(args):
    print(f"Running permutation inference on {args.output} ({args.n_permutations} permutations)...")
    df = load_results(args.output)
    permutation_inference(df, n_permutations=args.n_permutations, seed=args.seed)


async def run_flipped(args):
    df = load_results(args.output)
    for item in (args.items or ["gun_control"]):
        flipped_spec_analysis(df, item_key=item)


async def run_anes(args):
    df = load_results(args.output)
    from pathlib import Path
    anes_benchmark(df, anes_path=Path(args.anes_path) if args.anes_path else None)


async def main():
    parser = argparse.ArgumentParser(
        description="Prompt Specification Curves"
    )
    sub = parser.add_subparsers(dest="command")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--items", nargs="+", default=None, choices=list(ANES_ITEMS.keys()))
    shared.add_argument("--repeats", type=int, default=REPEATS)
    shared.add_argument("--max_concurrent", type=int, default=15)
    shared.add_argument("--seed", type=int, default=42)
    shared.add_argument("--dry_run", action="store_true")
    shared.add_argument("--output", default="pilot_results.json")
    shared.add_argument("--exclude_models", nargs="+", default=None,
                        help="Model identifiers to exclude from analysis (e.g. gemini-2.5-flash)")

    lhs = sub.add_parser("lhs", parents=[shared], help="LHS sampling run")
    lhs.add_argument("--n_specs", type=int, default=100)
    lhs.add_argument("--only_models", nargs="+", default=None,
                     help="Restrict the model dimension to these systems "
                          "(useful for incremental v2 catch-up runs)")
    lhs.add_argument("--spec_id_offset", type=int, default=0,
                     help="Offset spec_ids so a follow-up run does not collide "
                          "with an earlier file")

    salt = sub.add_parser("saltelli", parents=[shared], help="Saltelli sampling for Sobol indices")
    salt.add_argument("--saltelli_n", type=int, default=256)
    salt.add_argument("--second_order", action="store_true")

    perm = sub.add_parser("permutation", parents=[shared], help="Permutation inference on existing results")
    perm.add_argument("--n_permutations", type=int, default=500)

    flip = sub.add_parser("flipped", parents=[shared], help="Analyze flipped specifications")

    sob = sub.add_parser("sobol", parents=[shared], help="Run Sobol analysis on existing Saltelli results")
    sob.add_argument("--saltelli_n", type=int, default=256)
    sob.add_argument("--second_order", action="store_true")

    ci = sub.add_parser("bootstrap", parents=[shared], help="Bootstrap CIs on eta-squared")
    ci.add_argument("--n_boot", type=int, default=5000)

    ab = sub.add_parser("anes", parents=[shared], help="Benchmark against ANES 2024")
    ab.add_argument("--anes_path", default=None)

    fisher = sub.add_parser("fisher", parents=[shared],
                            help="Fisher r-to-z test for top-2 dimension dominance")

    threshold = sub.add_parser("threshold", parents=[shared],
                               help="Derive empirical coverage threshold under permutation null")
    threshold.add_argument("--n_permutations", type=int, default=10000)

    profile = sub.add_parser("profile_sensitivity", parents=[shared],
                             help="Leave-one-profile-out jackknife on amplification ratio")

    sysdecomp = sub.add_parser("system_decomp", parents=[shared],
                               help="Decompose deployed-system eta-squared (jackknife + open vs proprietary)")

    hsysd = sub.add_parser("hierarchical_decomp", parents=[shared],
                           help="Hierarchical decomposition of system variance (access -> provider -> family -> size)")

    analyze = sub.add_parser("analyze", parents=[shared], help="Re-run analysis on existing results")

    legacy = sub.add_parser("run", parents=[shared], help="Legacy: same as lhs")
    legacy.add_argument("--n_specs", type=int, default=100)

    args = parser.parse_args()
    exclude = getattr(args, "exclude_models", None)
    if exclude:
        print(f"[load_results: excluding models {exclude}]")

    if args.command is None or args.command == "analyze":
        df = load_results(
            args.output if hasattr(args, "output") else "pilot_results.json",
            exclude_models=exclude,
        )
        run_analysis_df(df)
    elif args.command in ("lhs", "run"):
        await run_lhs(args)
    elif args.command == "saltelli":
        await run_saltelli(args)
    elif args.command == "permutation":
        df = load_results(args.output, exclude_models=exclude)
        permutation_inference(df, n_permutations=args.n_permutations, seed=args.seed)
    elif args.command == "flipped":
        df = load_results(args.output, exclude_models=exclude)
        for item in (args.items or ["gun_control"]):
            flipped_spec_analysis(df, item_key=item)
    elif args.command == "sobol":
        from sampler import get_saltelli_problem
        df = load_results(args.output, exclude_models=exclude)
        problem = get_saltelli_problem()
        sobol_analysis(df, problem, args.saltelli_n, calc_second_order=args.second_order)
    elif args.command == "bootstrap":
        df = load_results(args.output, exclude_models=exclude)
        bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    elif args.command == "anes":
        from pathlib import Path
        df = load_results(args.output, exclude_models=exclude)
        anes_benchmark(df, anes_path=Path(args.anes_path) if args.anes_path else None)
    elif args.command == "fisher":
        df = load_results(args.output, exclude_models=exclude)
        fisher_rz_dimension_test(df)
    elif args.command == "threshold":
        df = load_results(args.output, exclude_models=exclude)
        derive_coverage_threshold(df, n_permutations=args.n_permutations, seed=args.seed)
    elif args.command == "profile_sensitivity":
        df = load_results(args.output, exclude_models=exclude)
        profile_jackknife(df)
    elif args.command == "system_decomp":
        df = load_results(args.output, exclude_models=exclude)
        system_decomposition(df)
    elif args.command == "hierarchical_decomp":
        df = load_results(args.output, exclude_models=exclude)
        hierarchical_system_decomp(df)


def run_analysis_df(df):
    """run_analysis but on a pre-filtered dataframe."""
    from analysis import summary_stats, compute_partisan_gaps, specification_curve, variance_decomposition
    print(f"Total responses: {len(df)}")
    stats = summary_stats(df)
    print("\n=== Partisan Gap Summary ===")
    print(stats.to_string(float_format="%.2f"))
    gaps = compute_partisan_gaps(df)
    for item in df["item"].unique():
        pct = specification_curve(gaps, item)
        print(f"{item}: {pct:.0f}% of specifications preserve expected partisan direction")
    print("\n=== Variance Decomposition (eta-squared) ===")
    var_df = variance_decomposition(df)
    print(var_df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    asyncio.run(main())
