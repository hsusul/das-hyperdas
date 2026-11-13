"""Time-of-week prompts (weekday x hour), kept only if Llama-3.1-8B answers them exactly.

The concept is a torus: two independent cycles, the weekday (period 7) and the hour (period 24).
"It is Monday at 14:00. In two hours it will be" -> " Monday at 16:00". Offsets stay within the same
day (hour + n <= 23) so the answer's day is always the prompt's day. A one-shot example fixes the format.
Output: /workspace/prompts/torus.json
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
N_WORDS = {1: "one hour", 2: "two hours", 3: "three hours"}
TEMPLATES = [
    ("It is Friday at 09:00. In one hour it will be Friday at 10:00.\n"
     "It is {day} at {hour:02d}:00. In {n} it will be", "{day} at {ans:02d}:00"),
    ("Q: It is Wednesday, 17:00. What will the day and time be in two hours?\nA: Wednesday, 19:00\n"
     "Q: It is {day}, {hour:02d}:00. What will the day and time be in {n}?\nA:", "{day}, {ans:02d}:00"),
]


@torch.no_grad()
def main() -> None:
    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rows = []
    for t, (template, answer) in enumerate(TEMPLATES):
        for day in DAYS:
            for hour in range(24):
                for n in N_WORDS:
                    if hour + n > 23:
                        continue
                    rows.append({"task": "torus", "template": t, "day": day, "hour": hour, "n": n,
                                 "raw_input": template.format(day=day, hour=hour, n=N_WORDS[n]),
                                 "answer": answer.format(day=day, ans=hour + n)})
    kept = []
    for i in range(0, len(rows), 64):
        batch = rows[i:i + 64]
        enc = tok([r["raw_input"] for r in batch], return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        for r, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            if tok.decode(g).strip().startswith(r["answer"]):
                kept.append(r)
    Path("/workspace/prompts").mkdir(exist_ok=True)
    json.dump(kept, open("/workspace/prompts/torus.json", "w"))
    per_t = {t: sum(r["template"] == t for r in kept) for t in range(len(TEMPLATES))}
    combos = len({(r["day"], r["hour"]) for r in kept})
    print(f"torus: {len(kept)}/{len(rows)} correct; per template {per_t}; {combos} of 168 (day, hour) combos covered")


if __name__ == "__main__":
    main()
