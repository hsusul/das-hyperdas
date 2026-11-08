"""Raw vs DAS vs HyperDAS subspace topology for the output variable, plus DAS/HyperDAS subspace overlap."""
import sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/results"
runs = {"months": "warmup_geometry_months", "weekdays": "warmup_geometry_weekdays_long", "hours": "warmup_geometry_hours16"}
cols = ["h1_count_finite", "h1_max_persistence", "h1_dominance_ratio", "h0_max_persistence", "ricci_mean", "ricci_fraction_negative"]
rows = []
for task, run in runs.items():
    d = pd.read_csv(f"{root}/{run}/topology_vectors.csv")
    acc = pd.read_csv(f"{root}/{run}/accuracy_by_variable.csv").set_index("kind")["acc"]
    rand = d[d.space == "random"][cols]
    U = np.load(f"{root}/{run}/{task}_output_L18.npz", allow_pickle=True)["U"]
    W = np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"]
    cos = np.linalg.svd(U.T @ W, compute_uv=False)
    print(f"== {task}: HyperDAS causal {acc['causal']:.2f} isolate {acc['isolate']:.2f}; "
          f"overlap with DAS (mean cos^2 of principal angles) {np.mean(cos**2):.2f}; cosines {np.round(cos, 2)}")
    for space in ("raw", "das", "hyperdas"):
        v = d[d.space == space][cols].iloc[0]
        z = (v - rand.mean()) / rand.std().replace(0, np.nan)
        rows.append({"task": task, "space": space, **{c: round(v[c], 3) for c in cols},
                     **{f"z_{c}": round(z[c], 2) for c in ("h1_max_persistence", "ricci_fraction_negative")}})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(f"{root}/warmup_summary.csv", index=False)
