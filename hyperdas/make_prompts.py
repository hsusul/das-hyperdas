"""Every value x offset prompt for each task and template, kept only if Llama-3.1-8B answers it.

Templates: the original arithmetic-wild one (t0) plus rephrasings in the style of their
rephrased_templates notebook. Output: /workspace/prompts/<task>.json, one row per correct prompt.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {20: "twenty", 30: "thirty", 40: "forty"}


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, ones = n - n % 10, n % 10
    return TENS[tens] + (f"-{ONES[ones]}" if ones else "")


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

TASKS = {
    "months": {
        "values": MONTHS, "offsets": range(1, 25), "index": lambda v: MONTHS.index(v) + 1,
        "answer": lambda s: MONTHS[(s % 12) - 1],
        "templates": [
            "Q: What month is {offset} months after {input}?\nA:",
            "Let's do some calendar math. {offset} months after {input} is what month? Answer:",
            "Question: It is {input}. In {offset} months, what month will it be? Answer: It will be",
        ],
    },
    "weekdays": {
        "values": DAYS, "offsets": range(1, 15), "index": lambda v: DAYS.index(v) + 1,
        "answer": lambda s: DAYS[(s % 7) - 1],
        "templates": [
            "Q: What day is {offset} days after {input}?\nA:",
            "Let's do some weekday math. {offset} days after {input} is what day? Answer:",
            "Question: Today is {input}. In {offset} days, what day will it be? Answer: It will be",
            "Q: If today is {input}, what day of the week will it be in {offset} days?\nA:",
            "Today is {input}. {offset} days from now, it will be",
        ],
    },
    "hours": {
        "values": [f"{h:02d}" for h in range(24)], "offsets": range(1, 49), "index": int,
        "answer": lambda s: f"{s % 24:02d}:00",
        "templates": [
            "Q: In 24-hour time, it is now {input}:00. What time will it be in {offset} hours?\nA: In 24-hour time, it will be ",
        ],
    },
}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=args.device).eval()
    out = Path("/workspace/prompts")
    out.mkdir(exist_ok=True)
    for task in args.tasks:
        spec = TASKS[task]
        rows = []
        for t, template in enumerate(spec["templates"]):
            for value in spec["values"]:
                for offset in spec["offsets"]:
                    premod = spec["index"](value) + offset
                    rows.append({"task": task, "template": t, "input": value, "offset": offset, "premod": premod,
                                 "answer": spec["answer"](premod),
                                 "raw_input": template.format(input=value, offset=words(offset))})
        kept = []
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]
            enc = tok([r["raw_input"] for r in batch], return_tensors="pt", padding=True).to(args.device)
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
            for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                if tok.decode(g).strip().startswith(r["answer"]):
                    kept.append(r)
        json.dump(kept, open(out / f"{task}.json", "w"))
        by_t = {t: sum(r["template"] == t for r in kept) for t in range(len(spec["templates"]))}
        print(f"{task}: {len(kept)}/{len(rows)} correct; per template {by_t}")


if __name__ == "__main__":
    main()
