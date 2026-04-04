import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ANES_ITEMS, PROFILES, SYSTEM_PROMPTS
from runner import run_batch, save_results, parse_response

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

MODELS = ["gpt-5.4-nano", "gpt-5.4", "claude-sonnet-4-6", "gemini-2.5-flash", "llama-3.3-70b", "mistral-small"]
ITEM = "gun_control"
ORDERINGS = ["standard", "reversed", "shuffled_1", "shuffled_2", "shuffled_3"]
REPEATS = 3


def build_forced_choice_prompt(profile, item_key, ordering, seed=None):
    item = ANES_ITEMS[item_key]
    scale = list(item["scale"].items())

    if ordering == "standard":
        order = scale
    elif ordering == "reversed":
        order = list(reversed(scale))
    else:
        rng = random.Random(seed)
        order = scale.copy()
        rng.shuffle(order)

    value_to_letter = {}
    choices = []
    for i, (val, label) in enumerate(order):
        letter = chr(97 + i)
        value_to_letter[letter] = val
        choices.append(f"  ({letter}) {label}")

    question = (
        f"{item['text']}\n\n"
        f"Which comes closest to your view?\n" +
        "\n".join(choices) +
        "\n\nRespond with only the letter."
    )

    persona = (
        f"You are a {profile['age']}-year-old {profile['race']} "
        f"{profile['gender']} {profile['party']} from "
        f"{profile['area']}, {profile['state']} with a "
        f"{profile['education']} education."
    )

    return {
        "system": SYSTEM_PROMPTS["neutral"],
        "user": f"{persona}\n\n{question}",
        "model": None,
        "temperature": 0.3,
    }, value_to_letter


async def main():
    tasks = []
    mappings = {}

    for model in MODELS:
        for ordering in ORDERINGS:
            for profile in PROFILES:
                for r in range(REPEATS):
                    seed = hash(f"{ordering}_{r}") % 10000 if ordering.startswith("shuffled") else None
                    prompt, letter_map = build_forced_choice_prompt(profile, ITEM, ordering, seed=seed)
                    prompt["model"] = model

                    task_id = len(tasks)
                    mappings[task_id] = letter_map
                    tasks.append({
                        "spec_id": task_id,
                        "profile_id": profile["id"],
                        "party": profile["party"],
                        "item": ITEM,
                        "repeat": r,
                        "persona_format": "bare",
                        "question_framing": f"forced_{ordering}",
                        "system_prompt": "neutral",
                        "few_shot": 0,
                        "model": model,
                        "ordering": ordering,
                        "prompt": prompt,
                    })

    print(f"Total calls: {len(tasks)}")
    print(f"Models: {len(MODELS)}, Orderings: {len(ORDERINGS)}, Profiles: {len(PROFILES)}, Repeats: {REPEATS}")

    start = time.time()
    results = await run_batch(tasks, max_concurrent=15)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.0f}s")

    task_ordering = {t["spec_id"]: t["ordering"] for t in tasks}
    for r in results:
        tid = r["spec_id"]
        r["ordering"] = task_ordering.get(tid, "unknown")
        letter_map = mappings[tid]
        raw = r.get("raw_response", "")
        if raw and not raw.startswith("ERROR"):
            raw_lower = raw.strip().lower()
            for char in raw_lower:
                if char in letter_map:
                    r["score"] = letter_map[char]
                    break

    Path("results").mkdir(exist_ok=True)
    with open("results/ordering_test.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results).dropna(subset=["score"])
    print(f"\nValid responses: {len(df)}/{len(results)}")

    gaps = df.groupby(["model", "ordering", "party"])["score"].mean().reset_index()
    pivot = gaps.pivot_table(index=["model", "ordering"], columns="party", values="score").reset_index()
    pivot["gap"] = pivot["Democrat"] - pivot["Republican"]

    print("\n=== Partisan Gap by Model x Ordering ===")
    print(pivot[["model", "ordering", "gap"]].to_string(index=False, float_format="%.2f"))

    fig, ax = plt.subplots(figsize=(10, 5))
    heat = pivot.pivot_table(index="model", columns="ordering", values="gap")
    im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto",
                   vmin=heat.values.min() - 0.5, vmax=heat.values.max() + 0.5)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=9)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Partisan Gap (D - R)")
    ax.set_title("Gun Control: Forced Choice Gap by Option Ordering", fontweight="normal")
    fig.savefig("figures/ordering_test.png")
    plt.close()
    print("\nSaved: figures/ordering_test.png")


if __name__ == "__main__":
    asyncio.run(main())
