"""Raw vs DAS-subspace concept geometry for Llama-3.1-8B (Arithmetic in the Wild).

For each cyclic task, collect last-token residual activations on the prompts
Llama answers correctly, mean-pool them per concept value, and compute the
hypo_free_geometry topology vector (PH + Ollivier-Ricci) on:
  raw      - full 4096-d residual (every layer, both hook sites)
  das      - projection onto the released layer-18 DAS subspace
  random   - projection onto random orthonormal subspaces of the same size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/workspace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "arithmetic-wild" / "src"))

from concept_geometry import GEOMETRY_FEATURE_COLUMNS, extract_geometry_features  # noqa: E402
from utils import load_subspace_hf  # noqa: E402

TASKS = ["months", "weekdays", "hours"]
LAYER = 18
# (DAS variable, hook site, grouping key). input/offset subspaces live before the
# MLP (post_attn_resid); output lives at the block output.
VARIABLES = {
    "input": ("post_attn", "input"),
    "output": ("block_out", "premod"),
}


def load_prompts(task: str) -> list[dict]:
    path = ROOT / "arithmetic-wild" / "datasets" / "Llama-3.1-8B" / task / "filtered_dataset.json"
    data = json.load(open(path))
    seen, rows = set(), []
    flat = data["input"] + [r for group in data["counterfactual_inputs"] for r in group]
    for row in flat:
        if row["raw_input"] not in seen:
            seen.add(row["raw_input"])
            rows.append(row)
    return rows


@torch.no_grad()
def collect(model, tok, prompts: list[str], batch_size: int = 32):
    """Return {site: [n_layers, n_prompts, d]} last-token activations."""
    n_layers = model.config.num_hidden_layers
    post_attn: dict[int, torch.Tensor] = {}

    def hook(layer_idx):
        def fn(module, args):
            post_attn[layer_idx] = args[0]
        return fn

    handles = [
        model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(hook(i))
        for i in range(n_layers)
    ]
    out = {"post_attn": [[] for _ in range(n_layers)], "block_out": [[] for _ in range(n_layers)]}
    tok.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            enc = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True).to(model.device)
            res = model(**enc, output_hidden_states=True)
            for i in range(n_layers):
                out["post_attn"][i].append(post_attn[i][:, -1].float().cpu())
                out["block_out"][i].append(res.hidden_states[i + 1][:, -1].float().cpu())
    finally:
        for h in handles:
            h.remove()
    return {site: torch.stack([torch.cat(v) for v in layers]).numpy() for site, layers in out.items()}


def mean_pool(acts: np.ndarray, keys: list) -> tuple[np.ndarray, list]:
    order = sorted(set(keys), key=lambda k: (isinstance(k, str), k))
    keys = np.asarray(keys, dtype=object)
    return np.stack([acts[keys == k].mean(0) for k in order]), order


def features(cloud: np.ndarray, name: str) -> dict:
    try:
        return extract_geometry_features(cloud, name).features
    except Exception as exc:  # small clouds can yield degenerate graphs
        print(f"  feature extraction failed for {name}: {exc}")
        return {c: np.nan for c in GEOMETRY_FEATURE_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "das_geometry"))
    parser.add_argument("--n-random", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    rng = np.random.default_rng(0)
    rows = []
    for task in TASKS:
        data = load_prompts(task)
        print(f"{task}: {len(data)} unique prompts")
        acts = collect(model, tok, [r["raw_input"] for r in data])
        for var, (site, key) in VARIABLES.items():
            keys = [r[key] for r in data]
            W = load_subspace_hf(task, var, layer=LAYER).float().cpu().numpy()  # (4096, k)
            k = W.shape[1]
            orth_err = float(np.abs(W.T @ W - np.eye(k)).max())

            pooled_by_layer = []
            for layer in range(acts[site].shape[0]):
                pooled, order = mean_pool(acts[site][layer], keys)
                pooled_by_layer.append(pooled)
                rows.append({"task": task, "variable": var, "space": "raw", "layer": layer,
                             "site": site, "n_points": len(order),
                             **features(pooled, f"{task}-{var}-raw-L{layer}")})
            pooled = pooled_by_layer[LAYER]
            np.savez(out / f"{task}_{var}_L{LAYER}.npz", raw=pooled, das=pooled @ W,
                     keys=np.asarray(order, dtype=object), W=W)
            rows.append({"task": task, "variable": var, "space": "das", "layer": LAYER,
                         "site": site, "n_points": len(order), "subspace_dim": k,
                         "orth_err": orth_err, **features(pooled @ W, f"{task}-{var}-das")})
            for seed in range(args.n_random):
                Q, _ = np.linalg.qr(rng.standard_normal((W.shape[0], k)))
                rows.append({"task": task, "variable": var, "space": "random", "layer": LAYER,
                             "site": site, "n_points": len(order), "subspace_dim": k,
                             "seed": seed, **features(pooled @ Q, f"{task}-{var}-rand{seed}")})
            print(f"  {var}: {len(order)} points, k={k}, orth_err={orth_err:.2e}")
        pd.DataFrame(rows).to_csv(out / "topology_vectors.csv", index=False)

    print(f"wrote {out / 'topology_vectors.csv'}")


if __name__ == "__main__":
    main()
