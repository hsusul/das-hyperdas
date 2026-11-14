"""Do concept values form clusters in the causal subspaces? Unsupervised clustering scored by ARI.

Points are individual correct prompts (last prompt token, output of layer 18, Llama-3.1-8B). Each
(concept, variable) is clustered with k = the number of distinct values of that variable, using
k-means (20 inits) and Ward agglomerative clustering, and scored against the true values with the
adjusted Rand index. Spaces:
  raw          : full 4096-d residual
  raw_pca      : raw projected on its top-k principal components, k = the subspace size (size-matched)
  das          : released DAS output subspace (arithmetic tasks only)
  hyperdas_s*  : HyperDAS subspaces, one per training seed, for that variable
  random       : random orthonormal subspaces of the same size (10 draws)
Nothing is tuned on the labels apart from k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/workspace")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@torch.no_grad()
def activations(prompts: list[str]) -> np.ndarray:
    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = []
    for i in range(0, len(prompts), 64):
        enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True).to("cuda")
        out.append(model(**enc, output_hidden_states=True).hidden_states[19][:, -1].float().cpu())
    return torch.cat(out).numpy()


def score(X: np.ndarray, labels: np.ndarray, k: int) -> dict:
    km = [adjusted_rand_score(labels, KMeans(k, n_init=20, random_state=s).fit_predict(X)) for s in range(3)]
    ward = adjusted_rand_score(labels, AgglomerativeClustering(k).fit_predict(X))
    return {"ari_kmeans": float(np.mean(km)), "ari_ward": float(ward)}


def main() -> None:
    p = ROOT / "repo/results"
    tasks = {t: json.load(open(p / f"point_clouds_200/prompts/{t}.json")) for t in ("months", "weekdays", "hours")}
    tasks["torus"] = json.load(open(p / "torus/torus.json"))
    cache = ROOT / "value_acts.npz"
    if cache.exists():
        acts = dict(np.load(cache))
    else:
        acts = {t: activations([r["raw_input"] for r in rows]) for t, rows in tasks.items()}
        np.savez(cache, **acts)

    # (task, variable, label per prompt, subspaces for that variable)
    cases = []
    for t in ("months", "weekdays", "hours"):
        rows = tasks[t]
        subs = {"das": np.load(p / f"das_geometry/{t}_output_L18.npz", allow_pickle=True)["W"]}
        for s in range(3):
            subs[f"hyperdas_s{s}"] = np.load(p / f"point_clouds_200/rotations/{t}_seed{s}.npy")
        premod = np.array([r["premod"] for r in rows])
        cases.append((t, "sum", premod, subs))
        cases.append((t, "sum_units_digit", premod % 10, subs))
    rows = tasks["torus"]
    for var, labels in (("day", np.array([DAYS.index(r["day"]) for r in rows])),
                        ("hour", np.array([r["hour"] for r in rows]))):
        subs = {f"hyperdas_s{s}": np.load(p / f"torus/rotations/{var}_seed{s}.npy") for s in range(2)}
        cases.append(("torus", var, labels, subs))

    rng = np.random.default_rng(0)
    results = []
    for task, var, labels, subs in cases:
        X = acts[task]
        k = len(np.unique(labels))
        dim = next(iter(subs.values())).shape[1]
        Xc = X - X.mean(0)
        spaces = {"raw": X, "raw_pca": Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:dim].T}
        spaces.update({name: X @ W for name, W in subs.items()})
        for r in range(10):
            spaces[f"random{r}"] = X @ np.linalg.qr(rng.standard_normal((X.shape[1], dim)))[0]
        for name, Y in spaces.items():
            results.append({"task": task, "variable": var, "k": k, "n": len(labels), "dim": dim,
                            "space": name, "kind": "random" if name.startswith("random") else
                            "hyperdas" if name.startswith("hyperdas") else name, **score(Y, labels, k)})
        print(task, var, "done", flush=True)
    df = pd.DataFrame(results)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "value_clustering.csv", index=False)
    summary = df.groupby(["task", "variable", "kind"])[["ari_kmeans", "ari_ward"]].agg(["mean", "std"]).round(3)
    summary.to_csv(out / "value_clustering_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
