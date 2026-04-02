import asyncio
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from config import PROFILES, ANES_ITEMS, REPEATS
from sampler import generate_saltelli_specifications
from prompts import build_prompt
from runner import run_batch, save_results, RESULTS_DIR


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--items", nargs="+", default=None, choices=list(ANES_ITEMS.keys()))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max_concurrent", type=int, default=15)
    parser.add_argument("--saltelli_n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(RESULTS_DIR / args.input) as f:
        existing = json.load(f)

    df = pd.DataFrame(existing)
    valid = df.dropna(subset=["score"])
    dead_specs = set(df["spec_id"].unique()) - set(valid["spec_id"].unique())
    print(f"Dead specs to patch: {len(dead_specs)}")

    if not dead_specs:
        print("Nothing to patch.")
        return

    specs, problem, total = generate_saltelli_specifications(
        n_base=args.saltelli_n, calc_second_order=False, seed=args.seed,
    )
    patch_specs = [s for s in specs if s["spec_id"] in dead_specs]
    print(f"Regenerated {len(patch_specs)} specifications to rerun")

    items = args.items or list(set(df["item"].unique()))

    tasks = []
    for spec in patch_specs:
        for item_key in items:
            for profile in PROFILES:
                for r in range(args.repeats):
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

    print(f"Patch API calls: {len(tasks):,}")

    if args.dry_run:
        print("[dry run]")
        return

    print(f"Running with max {args.max_concurrent} concurrent requests...")
    start = time.time()
    patch_results = await run_batch(tasks, max_concurrent=args.max_concurrent)
    elapsed = time.time() - start
    print(f"Completed {len(patch_results):,} calls in {elapsed:.0f}s")

    old_by_spec = {}
    for r in existing:
        old_by_spec.setdefault(r["spec_id"], []).append(r)

    for r in patch_results:
        old_by_spec.setdefault(r["spec_id"], []).append(r)

    merged = []
    for sid in sorted(old_by_spec.keys()):
        rows = old_by_spec[sid]
        has_valid = any(r["score"] is not None for r in rows)
        if has_valid:
            merged.extend([r for r in rows if r["score"] is not None])
        else:
            merged.extend(rows)

    output = args.input.replace(".json", "_patched.json")
    path = save_results(merged, output)
    print(f"Merged results saved to {path}")

    valid_count = sum(1 for r in merged if r["score"] is not None)
    spec_count = len(set(r["spec_id"] for r in merged if r["score"] is not None))
    print(f"Total valid responses: {valid_count}")
    print(f"Total valid specs: {spec_count}")


if __name__ == "__main__":
    asyncio.run(main())
