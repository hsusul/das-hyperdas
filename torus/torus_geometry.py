"""Is time-of-week (weekday x hour) a torus inside Llama-3.1-8B, and where?

A torus has two independent loops (two persistent H1 bars) and one enclosed void (one persistent H2
bar); a circle has one H1 bar and no H2. Clouds are last-prompt-token activations, either mean-pooled
per (day, hour) or 200 individual prompts (3 draws). Spaces: raw residual at several layers, and at
the output of layer 18 the HyperDAS day subspace, hour subspace, their union, and random subspaces of
the union's size. Each cloud is preprocessed like concept_geometry (PCA to 95% variance, unit diameter),
then Vietoris-Rips persistence up to H2 (ripser). Persistence is compared with 20 draws of the repo's
low-rank Gaussian null (same mean and covariance spectrum, no nonlinear structure). A sampled Clifford
torus and circle are run through the same steps as positive controls.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, "/workspace")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LAYERS = [11, 15, 19, 23, 27]  # hidden_states indices: outputs of layers 10, 14, 18, 22, 26
N_NULL = 20


def preprocess(cloud: np.ndarray) -> np.ndarray:
    from concept_geometry import normalize_point_cloud_diameter, reduce_point_cloud_pca

    projected, _, _ = reduce_point_cloud_pca(cloud, variance_threshold=0.95)
    return normalize_point_cloud_diameter(projected)[0]


def persistence(cloud: np.ndarray) -> dict:
    from ripser import ripser

    dgms = ripser(preprocess(cloud), maxdim=2)["dgms"]
    out = {}
    for dim, k in ((1, 3), (2, 2)):
        d = dgms[dim]
        p = np.sort((d[:, 1] - d[:, 0])[np.isfinite(d[:, 1])])[::-1] if len(d) else np.array([])
        p = np.pad(p, (0, max(0, k - len(p))))
        for i in range(k):
            out[f"h{dim}_top{i + 1}"] = float(p[i])
    return out


def measure(job: dict) -> dict:
    from null_cloud import fit_low_rank_gaussian, sample_low_rank_gaussian

    cloud = job["cloud"]
    obs = persistence(cloud)
    fit = fit_low_rank_gaussian(cloud)
    nulls = pd.DataFrame([persistence(sample_low_rank_gaussian(fit, seed=s)) for s in range(N_NULL)])
    row = {**job["meta"], **obs}
    for key in ("h1_top1", "h1_top2", "h2_top1"):
        row[f"{key}_null_mean"] = float(nulls[key].mean())
        row[f"{key}_p"] = float((1 + (nulls[key] >= obs[key]).sum()) / (N_NULL + 1))
    if job.get("day") is not None:
        d = pdist(cloud)
        day = pdist(job["day"][:, None], lambda a, b: min(abs(a - b)[0] % 7, 7 - abs(a - b)[0] % 7))
        hour = pdist(job["hour"][:, None], lambda a, b: min(abs(a - b)[0] % 24, 24 - abs(a - b)[0] % 24))
        row["rho_day"], row["rho_hour"] = spearmanr(d, day)[0], spearmanr(d, hour)[0]
    return row


def activations(prompts: list[str]) -> dict[int, np.ndarray]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "meta-llama/Llama-3.1-8B"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            for layer in LAYERS:
                out[layer].append(hs[layer][:, -1].float().cpu())
    return {layer: torch.cat(v).numpy() for layer, v in out.items()}


def controls(rng: np.random.Generator) -> list[dict]:
    """Clifford torus and circle sampled on the same (day, hour) grid, in 64 dims with noise."""
    d, h = np.meshgrid(np.arange(7), np.arange(23), indexing="ij")
    a, b = 2 * np.pi * d.ravel() / 7, 2 * np.pi * h.ravel() / 24
    embed = np.linalg.qr(rng.standard_normal((64, 4)))[0]
    torus = np.c_[np.cos(a), np.sin(a), np.cos(b), np.sin(b)] @ embed.T
    circle = np.c_[np.cos(b), np.sin(b), 0 * a, 0 * a] @ embed.T
    jobs = []
    for name, cloud in (("control_torus", torus), ("control_circle", circle)):
        cloud = cloud + 0.05 * rng.standard_normal(cloud.shape)
        jobs.append({"cloud": cloud, "day": d.ravel().astype(float), "hour": h.ravel().astype(float),
                     "meta": {"cloud": "control", "space": name, "layer": -1, "draw": -1, "n_points": len(cloud)}})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/results/torus_geometry")
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--reuse-acts", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = json.load(open("/workspace/prompts/torus.json"))
    if args.reuse_acts:
        acts = {layer: np.load(out / f"acts_L{layer}.npy") for layer in LAYERS}
    else:
        acts = activations([r["raw_input"] for r in rows])
        for layer, X in acts.items():
            np.save(out / f"acts_L{layer}.npy", X)
    day = np.array([DAYS.index(r["day"]) for r in rows], dtype=float)
    hour = np.array([r["hour"] for r in rows], dtype=float)
    keys = sorted({(int(a), int(b)) for a, b in zip(day, hour)})
    pool_index = [np.flatnonzero((day == a) & (hour == b)) for a, b in keys]
    pool_day = np.array([k[0] for k in keys], dtype=float)
    pool_hour = np.array([k[1] for k in keys], dtype=float)

    spaces: dict[str, tuple[int, np.ndarray | None]] = {f"raw_L{layer}": (layer, None) for layer in LAYERS}
    rng = np.random.default_rng(0)
    for seed_dir in sorted(Path("/workspace/results").glob("hd_torus_day_seed*_stage2")):
        seed = seed_dir.name.split("seed")[1].split("_")[0]
        hour_dir = Path(str(seed_dir).replace("_day_", "_hour_"))
        Ud = np.load(seed_dir / "final_model/rotation.npy")
        Uh = np.load(hour_dir / "final_model/rotation.npy")
        spaces[f"hyperdas_day_s{seed}"] = (19, Ud)
        spaces[f"hyperdas_hour_s{seed}"] = (19, Uh)
        spaces[f"hyperdas_union_s{seed}"] = (19, np.linalg.qr(np.c_[Ud, Uh])[0])
        overlap = float((np.linalg.svd(Ud.T @ Uh, compute_uv=False) ** 2).mean())
        print(f"seed {seed}: day/hour subspace overlap {overlap:.3f}", flush=True)
    k = next((W.shape[1] for name, (_, W) in spaces.items() if name.startswith("hyperdas_union")), 16)
    for r in range(args.n_random):
        spaces[f"random{r}"] = (19, np.linalg.qr(rng.standard_normal((acts[19].shape[1], k)))[0])

    jobs = controls(rng)
    for name, (layer, W) in spaces.items():
        X = acts[layer] if W is None else acts[layer] @ W
        pooled = np.stack([X[i].mean(0) for i in pool_index])
        jobs.append({"cloud": pooled, "day": pool_day, "hour": pool_hour,
                     "meta": {"cloud": "pooled", "space": name, "layer": layer, "draw": -1, "n_points": len(pooled)}})
        for draw in range(args.draws):
            idx = np.random.default_rng(100 + draw).choice(len(rows), size=min(args.n_points, len(rows)), replace=False)
            jobs.append({"cloud": X[idx], "day": day[idx], "hour": hour[idx],
                         "meta": {"cloud": "prompts", "space": name, "layer": layer, "draw": draw, "n_points": len(idx)}})
    print(f"{len(rows)} prompts, {len(keys)} (day, hour) points, {len(jobs)} clouds", flush=True)

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(measure, jobs, chunksize=1))
    df = pd.DataFrame(results)
    df["kind"] = [sp if sp.startswith(("raw", "control")) else
                  sp.rsplit("_s", 1)[0] if sp.startswith("hyperdas") else "random" for sp in df.space]
    df.to_csv(out / "torus_persistence.csv", index=False)
    cols = ["h1_top1", "h1_top2", "h1_top3", "h2_top1", "h2_top2", "h1_top2_p", "h2_top1_p", "rho_day", "rho_hour"]
    summary = df.groupby(["cloud", "kind"])[cols].mean().round(3)
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
