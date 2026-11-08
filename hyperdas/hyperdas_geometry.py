"""Concept geometry inside the subspaces a trained HyperDAS picks (Llama-3.1-8B, layer 18).

HyperDAS's ReflectSelect module gives every example its own basis W_i = H(rv_i) R,
a Householder reflection of one learned 4096 x k basis R. Per task and variable we
collect W_i over held-out causal pairs, take the top-k eigenvectors of the mean
projector as the consensus subspace U, and report how well the W_i agree with it.
Mean-pooled last-token activations (hidden_states[19], the output of layer 18) are
then measured raw, in U, in the released DAS subspace, and in random subspaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace")
import train_hyperdas_tok  # noqa: F401,E402  single-BOS tokenizer
sys.path.insert(0, "/workspace/HyperDAS")

from transformers import AutoTokenizer  # noqa: E402

from concept_geometry import GEOMETRY_FEATURE_COLUMNS, extract_geometry_features  # noqa: E402
from src.hyperdas.data_utils import get_ravel_collate_fn  # noqa: E402
from src.hyperdas.llama3.model import RavelInterpreterHypernetwork  # noqa: E402

M = "meta-llama/Llama-3.1-8B"
TASKS = ["months", "weekdays", "hours"]
GROUP_KEY = {"input": "input", "output": "premod"}


def features(cloud, name):
    try:
        return extract_geometry_features(cloud, name).features
    except Exception as exc:
        print(f"  feature extraction failed for {name}: {exc}")
        return {c: np.nan for c in GEOMETRY_FEATURE_COLUMNS}


def strip(row: dict) -> dict:
    """Same prompt handling as train_hyperdas.split_trailing_space."""
    row = dict(row)
    for side in ("input", "counterfactual_input"):
        if row[f"{side}_prefix"].endswith(" "):
            row[f"{side}_prefix"], row[f"{side}_suffix"] = row[f"{side}_prefix"].rstrip(" "), " "
    return row


@torch.no_grad()
def subspaces(h, tok, rows, batch_size=16):
    """Per-example reflected bases for these rows, shape [n, 4096, k]."""
    das = h.interpreter.das_module
    R = das.rotate_layer.weight.float()
    collate = get_ravel_collate_fn(tok, add_space_before_target=True,
                                   source_suffix_visibility=True, base_suffix_visibility=True)
    das.save_rv, das.rvs = True, []
    for i in range(0, len(rows), batch_size):
        b = {k: v.cuda() for k, v in collate(rows[i:i + batch_size]).items()}
        h.forward(**b)
    rv = torch.cat(das.rvs).float()
    das.save_rv, das.rvs = False, []
    # H R = R - 2 rv (rv^T R)
    return R.unsqueeze(0) - 2 * rv.unsqueeze(2) * (rv @ R).unsqueeze(1)


@torch.no_grad()
def accuracy(h, tok, rows, batch_size=16) -> float:
    """Fraction of rows whose whole target is predicted exactly under the learned intervention."""
    collate = get_ravel_collate_fn(tok, add_space_before_target=True,
                                   source_suffix_visibility=True, base_suffix_visibility=True)
    hits = []
    for i in range(0, len(rows), batch_size):
        b = {k: v.cuda() for k, v in collate(rows[i:i + batch_size]).items()}
        logits = h.forward(**b).logits
        pred, lab = logits[:, :-1].argmax(-1), b["labels"][:, 1:]
        mask = lab != -100
        hits += ((pred == lab) | ~mask).all(-1).tolist()
    return float(np.mean(hits))


def consensus(W: torch.Tensor):
    k = W.shape[-1]
    P = torch.einsum("nik,njk->ij", W, W) / len(W)
    evals, evecs = torch.linalg.eigh(P)
    U = evecs[:, -k:]
    agreement = (torch.einsum("nik,il->nkl", W, U) ** 2).sum((1, 2)) / k  # 1 = same subspace
    return U, agreement, evals[-k:].flip(0)


@torch.no_grad()
def last_token_states(target, tok, prompts, layer, batch_size=32):
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok([tok.bos_token + p for p in prompts[i:i + batch_size]], return_tensors="pt", padding=True).to("cuda")
        hs = target(**enc, output_hidden_states=True).hidden_states[layer]
        out.append(hs[:, -1].float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/results/hyperdas_full/final_model")
    parser.add_argument("--data", default="/workspace/hyperdas_data/test.jsonl")
    parser.add_argument("--das-dir", default="/workspace/das_geometry")
    parser.add_argument("--out", default="/workspace/results/hyperdas_geometry")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--das-dim", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--variables", nargs="+", default=["input", "output"], help="variables to measure geometry for")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--subspace-module", default="ReflectSelect", choices=["ReflectSelect", "DAS"],
                        help="DAS: one learned basis, used directly as the subspace")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(M)
    tok.pad_token, tok.padding_side, tok.pad_token_id = tok.eos_token, "left", tok.eos_token_id
    h = RavelInterpreterHypernetwork(model_name_or_path=M, num_editing_heads=32, intervention_layer=args.layer,
                                     subspace_module=args.subspace_module, das_dimension=args.das_dim,
                                     chop_editor_at_layer=2).cuda()
    h.load_model(args.model_dir)
    h.eval()

    rows = [strip(json.loads(line)) for line in open(args.data)]
    rng = np.random.default_rng(0)
    acc_rows = []
    for task in args.tasks:
        for var in args.variables:
            for kind in ["causal", "isolate"]:
                sel = [r for r in rows if r["task"] == task and r["variable"] == var and r["attribute_type"] == kind][:256]
                acc_rows.append({"task": task, "variable": var, "kind": kind, "n": len(sel), "acc": accuracy(h, tok, sel)})
                print(acc_rows[-1])
    pd.DataFrame(acc_rows).to_csv(out / "accuracy_by_variable.csv", index=False)
    table, agreement_rows = [], []
    for task in args.tasks:
        data = json.load(open(f"/workspace/arithmetic-wild/datasets/Llama-3.1-8B/{task}/filtered_dataset.json"))
        seen, prompts = set(), []
        for r in data["input"] + [r for g in data["counterfactual_inputs"] for r in g]:
            if r["raw_input"] not in seen:
                seen.add(r["raw_input"])
                prompts.append(r)
        acts = last_token_states(h.interpreter.target_model, tok, [p["raw_input"] for p in prompts], args.layer)
        for var, key in ((v, GROUP_KEY[v]) for v in args.variables):
            pair_rows = [r for r in rows if r["task"] == task and r["variable"] == var and r["attribute_type"] == "causal"]
            pair_rows = [pair_rows[i] for i in rng.permutation(len(pair_rows))[: args.pairs]]
            if args.subspace_module == "DAS":
                W = h.interpreter.das_module.rotate_layer.weight.detach().float().unsqueeze(0)
            else:
                W = subspaces(h, tok, pair_rows)
            U, agree, top = consensus(W)
            agreement_rows.append({"task": task, "variable": var, "n_pairs": len(W),
                                   "agreement_mean": agree.mean().item(), "agreement_min": agree.min().item(),
                                   "projector_top_eig": top.tolist()})
            keys = np.asarray([p[key] for p in prompts], dtype=object)
            order = sorted(set(keys.tolist()))
            pooled = np.stack([acts[keys == k].mean(0) for k in order])
            U = U.cpu().numpy()
            spaces = {"raw": pooled, "hyperdas": pooled @ U}
            das_file = Path(args.das_dir) / f"{task}_{var}_L18.npz"
            if das_file.exists():
                spaces["das"] = pooled @ np.load(das_file, allow_pickle=True)["W"]
            for space, cloud in spaces.items():
                table.append({"task": task, "variable": var, "space": space, "n_points": len(order),
                              **features(cloud, f"{task}-{var}-{space}")})
            for seed in range(args.n_random):
                Q, _ = np.linalg.qr(rng.standard_normal((pooled.shape[1], U.shape[1])))
                table.append({"task": task, "variable": var, "space": "random", "seed": seed,
                              "n_points": len(order), **features(pooled @ Q, f"{task}-{var}-rand{seed}")})
            np.savez(out / f"{task}_{var}_L18.npz", raw=pooled, hyperdas=pooled @ U, U=U,
                     keys=np.asarray(order, dtype=object))
            print(f"{task}/{var}: {len(order)} points, agreement {agree.mean().item():.3f}")
        pd.DataFrame(table).to_csv(out / "topology_vectors.csv", index=False)
    json.dump(agreement_rows, open(out / "subspace_agreement.json", "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
