"""How is the hour coded? Spearman correlation of pairwise distance with circular hour distance (mod 24),
linear hour distance |h - h'|, the units-digit distance of the hour, and the answer hour (h + n) distances,
for 200-prompt clouds (3 draws) in raw and HyperDAS spaces at the output of layer 18, plus random subspaces."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

out = Path("/workspace/results/torus_geometry")
rows = json.load(open("/workspace/prompts/torus.json"))
X = np.load(out / "acts_L19.npy")
hour = np.array([r["hour"] for r in rows], dtype=float)
ans = hour + np.array([r["n"] for r in rows], dtype=float)
spaces = {"raw": None}
for f in sorted(Path("/workspace/results").glob("hd_torus_*_seed*_stage2/final_model/rotation.npy")):
    spaces[f.parts[-3].replace("hd_torus_", "").replace("_stage2", "")] = np.load(f)
rng = np.random.default_rng(0)
for r in range(10):
    spaces[f"random{r}"] = np.linalg.qr(rng.standard_normal((X.shape[1], 8)))[0]


def metric(f):
    return lambda v: pdist(v[:, None], lambda a, b: f(a[0], b[0]))


codes = {
    "hour_circular": metric(lambda a, b: min(abs(a - b) % 24, 24 - abs(a - b) % 24)),
    "hour_linear": metric(lambda a, b: abs(a - b)),
    "hour_units_digit": metric(lambda a, b: min(abs(a - b) % 10, 10 - abs(a - b) % 10)),
}
res = []
for draw in range(3):
    idx = np.random.default_rng(100 + draw).choice(len(rows), size=200, replace=False)
    for name, W in spaces.items():
        d = pdist(X[idx] if W is None else X[idx] @ W)
        row = {"space": "random" if name.startswith("random") else name, "draw": draw}
        for code, fn in codes.items():
            row[f"rho_{code}"] = spearmanr(d, fn(hour[idx]))[0]
            row[f"rho_answer_{code}"] = spearmanr(d, fn(ans[idx]))[0]
        res.append(row)
df = pd.DataFrame(res)
df.to_csv(out / "hour_code.csv", index=False)
print(df.groupby("space").mean(numeric_only=True).drop(columns="draw").round(3).T.to_string())
