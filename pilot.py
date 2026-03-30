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
    print(f"Generating {args.n_specs} LHS specifications (seed={args.seed})...")
    specs = generate_specifications(args.n_specs, seed=args.seed)

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

    lhs = sub.add_parser("lhs", parents=[shared], help="LHS sampling run")
    lhs.add_argument("--n_specs", type=int, default=100)

    salt = sub.add_parser("saltelli", parents=[shared], help="Saltelli sampling for Sobol indices")
    salt.add_argument("--saltelli_n", type=int, default=512)
    salt.add_argument("--second_order", action="store_true")

    perm = sub.add_parser("permutation", parents=[shared], help="Permutation inference on existing results")
    perm.add_argument("--n_permutations", type=int, default=500)

    flip = sub.add_parser("flipped", parents=[shared], help="Analyze flipped specifications")

    ci = sub.add_parser("bootstrap", parents=[shared], help="Bootstrap CIs on eta-squared")
    ci.add_argument("--n_boot", type=int, default=5000)

    ab = sub.add_parser("anes", parents=[shared], help="Benchmark against ANES 2024")
    ab.add_argument("--anes_path", default=None)

    analyze = sub.add_parser("analyze", parents=[shared], help="Re-run analysis on existing results")

    legacy = sub.add_parser("run", parents=[shared], help="Legacy: same as lhs")
    legacy.add_argument("--n_specs", type=int, default=100)

    args = parser.parse_args()

    if args.command is None or args.command == "analyze":
        run_analysis(args.output if hasattr(args, "output") else "pilot_results.json")
    elif args.command in ("lhs", "run"):
        await run_lhs(args)
    elif args.command == "saltelli":
        await run_saltelli(args)
    elif args.command == "permutation":
        await run_permutation(args)
    elif args.command == "flipped":
        await run_flipped(args)
    elif args.command == "bootstrap":
        df = load_results(args.output)
        bootstrap_ci(df, n_boot=args.n_boot, seed=args.seed)
    elif args.command == "anes":
        await run_anes(args)


if __name__ == "__main__":
    asyncio.run(main())
