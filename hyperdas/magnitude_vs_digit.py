"""Which structure do pairwise distances between per-sum points follow: magnitude |a-b| or the units digit?

Spearman correlation of pairwise Euclidean distance with |a - b| (magnitude) and with the circular units-digit
distance min(|a-b| mod 10, 10 - |a-b| mod 10) (base-10). Random same-size subspaces give a reference.
"""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

runs = {"months": "warmup_geometry_months", "weekdays": "warmup_geometry_weekdays_long", "hours": "warmup_geometry_hours16"}
rng = np.random.default_rng(0)
rows = []
for task, run in runs.items():
    h = np.load(f"/workspace/results/{run}/{task}_output_L18.npz", allow_pickle=True)
    W = np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"]
    s = np.array([int(k) for k in h["keys"]], dtype=float)[:, None]
    mag = pdist(s, lambda a, b: abs(a - b)[0])
    digit = pdist(s, lambda a, b: min(abs(a - b)[0] % 10, 10 - abs(a - b)[0] % 10))
    clouds = {"raw": h["raw"], "das": h["raw"] @ W, "hyperdas": h["hyperdas"]}
    for i in range(20):
        Q, _ = np.linalg.qr(rng.standard_normal((h["raw"].shape[1], W.shape[1])))
        clouds[f"random{i}"] = h["raw"] @ Q
    for name, c in clouds.items():
        d = pdist(c)
        rows.append({"task": task, "space": name, "rho_magnitude": spearmanr(d, mag)[0], "rho_units_digit": spearmanr(d, digit)[0]})
df = pd.DataFrame(rows)
df["kind"] = df.space.where(~df.space.str.startswith("random"), "random")
out = df.groupby(["task", "kind"])[["rho_magnitude", "rho_units_digit"]].agg(["mean", "std"]).round(3)
print(out.to_string())
df.to_csv("/workspace/results/magnitude_vs_digit.csv", index=False)
