"""2-D PCA views of the mean-pooled concept clouds, raw vs DAS subspace (layer 18)."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

src = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/results/das_geometry")
files = sorted(src.glob("*_L18.npz"))
fig, axes = plt.subplots(len(files), 2, figsize=(8, 3.6 * len(files)))
for row, f in zip(axes, files):
    d = np.load(f, allow_pickle=True)
    keys = list(d["keys"])
    for ax, space in zip(row, ("raw", "das")):
        xy = PCA(2).fit_transform(d[space])
        ax.plot(*np.r_[xy, xy[:1]].T, color="0.8", lw=0.8, zorder=0)
        ax.scatter(*xy.T, c=np.arange(len(keys)), cmap="twilight", s=30)
        for (x, y), k in zip(xy, keys):
            ax.annotate(str(k)[:3], (x, y), fontsize=7)
        ax.set_title(f"{f.stem.replace('_L18', '')} - {space}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(src / "clouds_pca.png", dpi=110)
print("wrote", src / "clouds_pca.png")
