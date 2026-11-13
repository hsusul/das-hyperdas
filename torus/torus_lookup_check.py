"""Lookup-table check: are the per-value means roughly equidistant (a simplex) instead of a cycle?

For each variable (day, hour), pool activations per value, then report the coefficient of variation of
the pairwise distances between value means (0 = perfectly equidistant), and how much closer adjacent
values (h, h+1) are than the average pair (ratio < 1 means neighbors are closer, i.e. ordered structure).
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
out = Path("/workspace/results/torus_geometry")
rows = json.load(open("/workspace/prompts/torus.json"))
X = np.load(out / "acts_L19.npy")
vals = {"day": np.array([DAYS.index(r["day"]) for r in rows]), "hour": np.array([r["hour"] for r in rows])}
period = {"day": 7, "hour": 24}
spaces = {"raw": None}
for f in sorted(Path("/workspace/results").glob("hd_torus_*_seed0_stage2/final_model/rotation.npy")):
    spaces[f.parts[-3].replace("hd_torus_", "").replace("_stage2", "")] = np.load(f)
report = {}
for var, v in vals.items():
    levels = np.unique(v)
    for name, W in spaces.items():
        Y = X if W is None else X @ W
        means = np.stack([Y[v == lv].mean(0) for lv in levels])
        D = squareform(pdist(means))
        off = D[np.triu_indices(len(levels), 1)]
        adj = [D[i, i + 1] for i in range(len(levels) - 1)]
        wrap = D[0, -1] if levels[-1] - levels[0] == period[var] - 1 else np.nan
        report[f"{var} in {name}"] = {"cv_pairwise": round(float(off.std() / off.mean()), 3),
                                      "adjacent_over_mean": round(float(np.mean(adj) / off.mean()), 3),
                                      "wrap_over_mean": None if np.isnan(wrap) else round(float(wrap / off.mean()), 3)}
json.dump(report, open(out / "lookup_check.json", "w"), indent=1)
for k, r in report.items():
    print(f"{k:22s} {r}")
