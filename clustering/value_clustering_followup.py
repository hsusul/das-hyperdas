"""Follow-ups to value_clustering.py (run after it, on its cached activations):
  torus hour clustered by the answer hour (hour + n) instead of the prompt hour, and
  sums clustered on template-0 prompts only (the template DAS/HyperDAS were trained on)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_clustering import ROOT, score

p = ROOT / "repo/results"
acts = dict(np.load(ROOT / "value_acts.npz"))
rng = np.random.default_rng(0)
rows_out = []


def run(task, var, X, labels, subs):
    k, dim = len(np.unique(labels)), next(iter(subs.values())).shape[1]
    spaces = {"raw": X, **{n: X @ W for n, W in subs.items()}}
    for r in range(10):
        spaces[f"random{r}"] = X @ np.linalg.qr(rng.standard_normal((X.shape[1], dim)))[0]
    for name, Y in spaces.items():
        rows_out.append({"task": task, "variable": var, "k": k, "n": len(labels), "space": name,
                         "kind": "random" if name.startswith("random") else "hyperdas" if name.startswith("hyperdas") else name,
                         **score(Y, labels, k)})


torus = json.load(open(p / "torus/torus.json"))
hour_subs = {f"hyperdas_s{s}": np.load(p / f"torus/rotations/hour_seed{s}.npy") for s in range(2)}
run("torus", "answer_hour", acts["torus"], np.array([r["hour"] + r["n"] for r in torus]), hour_subs)
t0 = np.array([r["template"] == 0 for r in torus])
run("torus_t0", "answer_hour", acts["torus"][t0], np.array([r["hour"] + r["n"] for r in torus])[t0], hour_subs)
for t in ("months", "weekdays", "hours"):
    rows = json.load(open(p / f"point_clouds_200/prompts/{t}.json"))
    m = np.array([r["template"] == 0 for r in rows])
    subs = {"das": np.load(p / f"das_geometry/{t}_output_L18.npz", allow_pickle=True)["W"]}
    subs.update({f"hyperdas_s{s}": np.load(p / f"point_clouds_200/rotations/{t}_seed{s}.npy") for s in range(3)})
    run(f"{t}_t0", "sum", acts[t][m], np.array([r["premod"] for r in rows])[m], subs)
df = pd.DataFrame(rows_out)
df.to_csv(ROOT / "results/value_clustering_followup.csv", index=False)
print(df.groupby(["task", "variable", "kind"])[["n", "k", "ari_kmeans", "ari_ward"]].mean().round(3).to_string())
