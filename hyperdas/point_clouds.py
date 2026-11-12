"""200-point concept clouds (one point per correct prompt) in raw, DAS, HyperDAS and random subspaces.

Activations: last prompt token, output of layer 18 (hidden_states[19]), Llama-3.1-8B. Each task's cloud
is 200 prompts drawn at random from the correct prompts in /workspace/prompts (all templates), repeated
for several draws. For every cloud we compute the hypo_free_geometry topology vector and three
Spearman correlations between pairwise distances and
  magnitude  : |sum_a - sum_b|
  units_digit: circular distance of the sums' units digits (base-10 structure)
  cycle      : circular distance of the starting concept on its own cycle (Dec next to Jan).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, "/workspace")
from concept_geometry import GEOMETRY_FEATURE_COLUMNS, extract_geometry_features  # noqa: E402

PERIOD = {"months": 12, "weekdays": 7, "hours": 24}
INDEX = {
    "months": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"],
    "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}


def concept_index(task: str, value: str) -> int:
    return int(value) if task == "hours" else INDEX[task].index(value)


@torch.no_grad()
def activations(prompts: list[str], layer: int) -> np.ndarray:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i in range(0, len(prompts), 64):
        enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True).to("cuda")
        out.append(model(**enc, output_hidden_states=True).hidden_states[layer][:, -1].float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def measure(job: dict) -> dict:
    cloud, s, c, period = job["cloud"], job["sums"], job["cycle"], job["period"]
    d = pdist(cloud)
    mag = pdist(s[:, None], lambda a, b: abs(a - b)[0])
    digit = pdist(s[:, None], lambda a, b: min(abs(a - b)[0] % 10, 10 - abs(a - b)[0] % 10))
    cyc = pdist(c[:, None], lambda a, b: min(abs(a - b)[0] % period, period - abs(a - b)[0] % period))
    try:
        feats = extract_geometry_features(cloud, job["name"]).features
    except Exception as exc:  # degenerate graphs
        print(f"features failed for {job['name']}: {exc}", flush=True)
        feats = {k: np.nan for k in GEOMETRY_FEATURE_COLUMNS}
    return {**job["meta"], **feats,
            "rho_magnitude": spearmanr(d, mag)[0],
            "rho_units_digit": spearmanr(d, digit)[0],
            "rho_cycle": spearmanr(d, cyc)[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--hyperdas-glob", default="/workspace/results/hd_{task}_seed*_stage2/final_model/rotation.npy")
    parser.add_argument("--out", default="/workspace/results/point_clouds")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs, counts = [], {}
    for task in PERIOD:
        rows = json.load(open(f"/workspace/prompts/{task}.json"))
        counts[task] = len(rows)
        X = activations([r["raw_input"] for r in rows], args.layer)
        np.save(out / f"{task}_acts.npy", X)
        sums = np.array([r["premod"] for r in rows], dtype=float)
        cycle = np.array([concept_index(task, r["input"]) for r in rows], dtype=float)
        spaces = {"raw": None,
                  "das": np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"]}
        for f in sorted(Path("/").glob(args.hyperdas_glob.format(task=task).lstrip("/"))):
            seed = f.parts[-3].split("_seed")[1].split("_")[0]
            spaces[f"hyperdas_s{seed}"] = np.load(f)
        k = spaces["das"].shape[1]
        rng = np.random.default_rng(0)
        for r in range(args.n_random):
            spaces[f"random{r}"] = np.linalg.qr(rng.standard_normal((X.shape[1], k)))[0]
        for draw in range(args.draws):
            idx = np.random.default_rng(100 + draw).choice(len(rows), size=min(args.n_points, len(rows)), replace=False)
            for space, W in spaces.items():
                cloud = X[idx] if W is None else X[idx] @ W
                jobs.append({"cloud": cloud, "sums": sums[idx], "cycle": cycle[idx], "period": PERIOD[task],
                             "name": f"{task}-{space}-d{draw}",
                             "meta": {"task": task, "space": space, "draw": draw, "n_points": len(idx),
                                      "kind": "random" if space.startswith("random") else
                                              "hyperdas" if space.startswith("hyperdas") else space}})
        print(f"{task}: {len(rows)} correct prompts, spaces {[s for s in spaces if not s.startswith('random')]}", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(measure, jobs, chunksize=2))
    df = pd.DataFrame(results)
    df.to_csv(out / "point_clouds.csv", index=False)
    cols = ["rho_magnitude", "rho_units_digit", "rho_cycle", "h1_count_finite", "h1_max_persistence",
            "ricci_fraction_negative"]
    summary = df.groupby(["task", "kind"])[cols].agg(["mean", "std"]).round(3)
    summary.to_csv(out / "summary.csv")
    print(json.dumps(counts))
    print(summary.to_string())


if __name__ == "__main__":
    main()
