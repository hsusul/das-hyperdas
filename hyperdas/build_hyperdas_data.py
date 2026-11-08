"""Turn Arithmetic-in-the-Wild prompts into HyperDAS (RAVEL-format) examples.

One example = base prompt, source prompt, a natural-language instruction naming
the variable to localize, and the answer expected after the intervention.
  causal  : source differs in that variable  -> answer follows the source value
  isolate : source shares that variable only -> answer stays the base answer
Only prompts Llama-3.1-8B answers correctly are used, for base, source and the
counterfactual answer prompt, and base prompts are split disjointly train/test.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

ROOT = Path("/workspace/arithmetic-wild")
sys.path.insert(0, str(ROOT / "src"))

TASKS = {
    "months": ("month", "months"),
    "weekdays": ("day", "days"),
    "hours": ("hour", "hours"),
}
INSTRUCTIONS = {
    "input": "Localize the starting {unit} in this {task} question.",
    "offset": "Localize the number of {units} being added in this {task} question.",
    "output": "Localize the resulting {unit} in this {task} question.",
}


def correct_prompts(task: str) -> dict[tuple[str, str], dict]:
    data = json.load(open(ROOT / "datasets" / "Llama-3.1-8B" / task / "filtered_dataset.json"))
    rows = {}
    # base prompts plus their counterfactuals; the filter kept only pairs the model gets right
    for row in data["input"] + [r for group in data["counterfactual_inputs"] for r in group]:
        rows[(row["input"], row["offset"])] = row
    return rows


def answer(row: dict) -> str:
    return row["raw_output"].strip()


def example(task, var, kind, base, source, target):
    unit, units = TASKS[task]
    return {
        "task": task,
        "variable": var,
        "edit_instruction": INSTRUCTIONS[var].format(unit=unit, units=units, task=task),
        "input_prefix": base["raw_input"],
        "input_suffix": "",
        "counterfactual_input_prefix": source["raw_input"],
        "counterfactual_input_suffix": "",
        "target": answer(base),
        "counterfactual_target": target,
        "attribute_type": kind,
    }


def build(task: str, rng: random.Random, per_base: int):
    cm = importlib.import_module(f"tasks.{task}.causal_models")
    rows = correct_prompts(task)
    by_premod: dict[int, list] = {}
    for r in rows.values():
        by_premod.setdefault(cm.compute_premod(r["offset"], r["input"]), []).append(r)
    keys = list(rows)
    out = []
    for (c, n), base in rows.items():
        premod = cm.compute_premod(n, c)
        for _ in range(per_base):
            src = rows[rng.choice(keys)]
            c2, n2 = src["input"], src["offset"]
            # input: answer becomes (c2 + n); needs a correct prompt for that pair
            if c2 != c and (c2, n) in rows:
                out.append(example(task, "input", "causal", base, src, answer(rows[(c2, n)])))
            if n2 != n and (c, n2) in rows:
                out.append(example(task, "offset", "causal", base, src, answer(rows[(c, n2)])))
            if cm.compute_premod(n2, c2) != premod:
                out.append(example(task, "output", "causal", base, src, answer(src)))
        # isolate: source keeps the localized variable and changes the rest
        same_c = [rows[k] for k in keys if k[0] == c and k[1] != n]
        same_n = [rows[k] for k in keys if k[1] == n and k[0] != c]
        same_p = [r for r in by_premod[premod] if (r["input"], r["offset"]) != (c, n)]
        for var, pool in (("input", same_c), ("offset", same_n), ("output", same_p)):
            for _ in range(max(1, per_base // 2) if pool else 0):
                out.append(example(task, var, "isolate", base, rng.choice(pool), answer(base)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/hyperdas_data")
    parser.add_argument("--per-base", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()
    rng = random.Random(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, test = [], []
    for task in TASKS:
        examples = build(task, rng, args.per_base)
        bases = sorted({e["input_prefix"] for e in examples})
        rng.shuffle(bases)
        test_bases = set(bases[: int(len(bases) * args.test_frac)])
        for e in examples:
            (test if e["input_prefix"] in test_bases else train).append(e)
        print(task, len(examples), "examples")
    for name, rows in (("train", train), ("test", test)):
        rng.shuffle(rows)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts = {}
        for r in rows:
            k = (r["variable"], r["attribute_type"])
            counts[k] = counts.get(k, 0) + 1
        print(name, len(rows), counts)


if __name__ == "__main__":
    main()
