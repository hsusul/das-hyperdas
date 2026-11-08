"""2-D PCA of the per-sum mean activations: raw, released DAS subspace, HyperDAS subspace (layer 18 output)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

runs = {"months": "warmup_geometry_months", "weekdays": "warmup_geometry_weekdays_long", "hours": "warmup_geometry_hours16"}
fig, axes = plt.subplots(len(runs), 3, figsize=(11, 3.6 * len(runs)))
for row, (task, run) in zip(axes, runs.items()):
    h = np.load(f"/workspace/results/{run}/{task}_output_L18.npz", allow_pickle=True)
    W = np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"]
    keys = list(h["keys"])
    for ax, (name, cloud) in zip(row, (("raw", h["raw"]), ("DAS", h["raw"] @ W), ("HyperDAS", h["hyperdas"]))):
        xy = PCA(2).fit_transform(cloud)
        ax.plot(*xy.T, color="0.8", lw=0.8, zorder=0)
        ax.scatter(*xy.T, c=np.arange(len(keys)), cmap="viridis", s=26)
        for (x, y), k in zip(xy, keys):
            ax.annotate(str(k), (x, y), fontsize=6)
        ax.set_title(f"{task} sum - {name}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("/workspace/results/warmup_clouds_pca.png", dpi=110)
print("wrote /workspace/results/warmup_clouds_pca.png")
