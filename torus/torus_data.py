"""HyperDAS (RAVEL-format) examples for the two torus variables, day and hour, from template 0.

causal  day : source has another day  -> answer is (source day, base hour + n)
causal  hour: source has another hour -> answer is (base day, source hour + n)
isolate     : source shares the localized variable -> answer stays the base answer
Only pairs whose counterfactual answer prompt the model also gets right are kept.
"""

import json
import random
from pathlib import Path

INSTRUCTIONS = {
    "day": "Localize the day of the week in this time question.",
    "hour": "Localize the hour in this time question.",
}


def main(per_base: int = 10, test_frac: float = 0.15) -> None:
    rng = random.Random(0)
    prompts = [r for r in json.load(open("/workspace/prompts/torus.json")) if r["template"] == 0]
    by_key = {(r["day"], r["hour"], r["n"]): r for r in prompts}

    def example(var, kind, base, source, target):
        return {"task": "torus", "variable": var, "edit_instruction": INSTRUCTIONS[var],
                "input_prefix": base["raw_input"], "input_suffix": "",
                "counterfactual_input_prefix": source["raw_input"], "counterfactual_input_suffix": "",
                "target": base["answer"], "counterfactual_target": target, "attribute_type": kind}

    rows = []
    for (d, h, n), base in by_key.items():
        for _ in range(per_base):
            src = rng.choice(prompts)
            if src["day"] != d and (src["day"], h, n) in by_key:
                rows.append(example("day", "causal", base, src, by_key[(src["day"], h, n)]["answer"]))
            if src["hour"] != h and (d, src["hour"], n) in by_key:
                rows.append(example("hour", "causal", base, src, by_key[(d, src["hour"], n)]["answer"]))
        same_day = [r for r in prompts if r["day"] == d and r["hour"] != h]
        same_hour = [r for r in prompts if r["hour"] == h and r["day"] != d]
        for var, pool in (("day", same_day), ("hour", same_hour)):
            for _ in range(per_base // 2 if pool else 0):
                rows.append(example(var, "isolate", base, rng.choice(pool), base["answer"]))

    bases = sorted({r["input_prefix"] for r in rows})
    rng.shuffle(bases)
    test_bases = set(bases[: int(len(bases) * test_frac)])
    out = Path("/workspace/torus_data")
    out.mkdir(exist_ok=True)
    for name, split in (("train", [r for r in rows if r["input_prefix"] not in test_bases]),
                        ("test", [r for r in rows if r["input_prefix"] in test_bases])):
        rng.shuffle(split)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        counts = {}
        for r in split:
            counts[(r["variable"], r["attribute_type"])] = counts.get((r["variable"], r["attribute_type"]), 0) + 1
        print(name, len(split), counts)


if __name__ == "__main__":
    main()
