import asyncio
import argparse
import time

from config import (
    PROFILES, ANES_ITEMS, REPEATS,
    COST_PER_1M_INPUT, COST_PER_1M_OUTPUT,
    AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
)
from sampler import generate_specifications
from prompts import build_prompt
from runner import run_batch, save_results
from analysis import run_analysis


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


async def main():
    parser = argparse.ArgumentParser(
        description="Prompt Specification Curves — Pilot Pipeline"
    )
    parser.add_argument("--n_specs", type=int, default=100)
    parser.add_argument("--items", nargs="+", default=None,
                        choices=list(ANES_ITEMS.keys()))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--output", default="pilot_results.json")
    args = parser.parse_args()

    if args.analyze_only:
        run_analysis(args.output)
        return

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
        print("\n[dry run — no API calls made]")
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


if __name__ == "__main__":
    asyncio.run(main())
