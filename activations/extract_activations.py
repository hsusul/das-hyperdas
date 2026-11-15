"""Per-prompt Llama-3.1-8B activations for every prompt set used in the experiments.

Last prompt token, residual stream. Layer 18 output (hidden_states[19]) for every task, which is
where the DAS and HyperDAS subspaces live; the weekday x hour task also gets the outputs of layers
10, 14, 22 and 26 used in its layer comparison. Stored as float16 with the prompt metadata, one
.npz per task, in the same row order as the prompt JSON files.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
PROMPTS = {
    "months": REPO / "results/point_clouds_200/prompts/months.json",
    "weekdays": REPO / "results/point_clouds_200/prompts/weekdays.json",
    "hours": REPO / "results/point_clouds_200/prompts/hours.json",
    "torus": REPO / "results/torus/torus.json",
}
LAYERS = {"months": [19], "weekdays": [19], "hours": [19], "torus": [11, 15, 19, 23, 27]}


@torch.no_grad()
def main() -> None:
    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = REPO / "activations"
    for task, path in PROMPTS.items():
        rows = json.load(open(path))
        acts = {layer: [] for layer in LAYERS[task]}
        for i in range(0, len(rows), 64):
            enc = tok([r["raw_input"] for r in rows[i:i + 64]], return_tensors="pt", padding=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            for layer in LAYERS[task]:
                acts[layer].append(hs[layer][:, -1].float().cpu())
        arrays = {f"layer{layer - 1}_out": torch.cat(a).numpy().astype(np.float16) for layer, a in acts.items()}
        meta = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "raw_input"}
        np.savez_compressed(out / f"{task}.npz", **arrays, prompts=np.array([r["raw_input"] for r in rows]), **meta)
        print(task, len(rows), "prompts,", list(arrays), flush=True)


if __name__ == "__main__":
    main()
