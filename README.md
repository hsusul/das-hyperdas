# findings

Concept geometry inside causally found subspaces (DAS and HyperDAS) vs raw activations, Llama-3.1-8B.

The question: does the shape of a concept (months, weekdays, hours) look different in the subspace the model actually computes with than in the raw residual stream?

## Setup

- Model: `meta-llama/Llama-3.1-8B`, bf16, one A100 80GB on RunPod.
- Tasks and prompts: [goodfire-ai/arithmetic-wild](https://github.com/goodfire-ai/arithmetic-wild) (commit `d03024a`), e.g. `Q: What month is three months after July?\nA:`. Only prompts the model answers correctly are used.
- DAS subspaces: released ones from [Tal535/Arithmetic-In-The-Wild_DAS](https://huggingface.co/datasets/Tal535/Arithmetic-In-The-Wild_DAS), layer 18, last token.
  - `input` (starting concept): after attention, before the MLP. 8-d (16-d for hours).
  - `output` (the sum before the modulo, `premod`): block output. 8-d (16-d for hours).
- Topology vector: `extract_geometry_features` from `concept_geometry.py` in the hypo_free_geometry repo (PCA to 95% variance, unit diameter, H0/H1 persistence + Ollivier-Ricci summaries).

## DAS (done)

`das/das_geometry.py` mean-pools last-token activations per concept value (per starting concept for `input`, per sum for `output`) and computes the topology vector on:

- raw 4096-d activations, every layer
- the DAS subspace at layer 18
- 20 random orthonormal subspaces of the same size (control)

Outputs are in `results/das_geometry/`: `topology_vectors.csv` (all rows), one `.npz` per task/variable with the pooled clouds and the subspace, and `clouds_pca.png` (2-D PCA, raw vs DAS).

First look, not yet a claim:

- The sum (`output`, grouped by `premod`) is an open arc in both raw and DAS space for months, weekdays and hours. It does not wrap at 12, 7 or 24, which fits the paper's account that the model adds plain integers and only maps back to the cycle later.
- For `input`, the wrap-around step (Dec to Jan, Sun to Mon, 23 to 00) is about as long as an ordinary neighbor step in both raw and DAS space (ratio to the median neighbor step: months 1.27 raw / 1.07 DAS, weekdays 1.10 / 1.10, hours 1.91 / 2.14). So far there is no raw-vs-DAS difference in where the cycle closes.
- DAS vs random-subspace z-scores are mixed (e.g. hours `output` H1 max persistence z = 6.8, months `output` z = 0.0). With 7 to 68 points per cloud these numbers are noisy.

## HyperDAS (in progress)

- `hyperdas/build_hyperdas_data.py` turns the arithmetic-wild prompts into HyperDAS's RAVEL format: base prompt, source prompt, an instruction such as "Localize the starting month in this months question.", and the expected answer. `causal` examples change the named variable; `isolate` examples keep it fixed and expect the base answer.
- `hyperdas/train_hyperdas.py` trains one hypernetwork for all three tasks and variables at layer 18 (hidden_states index 19), subspace dim 8.
- `hyperdas/hyperdas_llama31.patch` is the two-line patch to [jiudingsun01/HyperDAS](https://github.com/jiudingsun01/HyperDAS) needed for Llama-3.1 with transformers 4.44.2:
  - the `_update_causal_mask` call signature changed;
  - the low-rank rotation weight was left uninitialized (`torch.empty`), which gave NaN losses.

## Reproduce

On a CUDA box with access to Llama-3.1-8B:

```bash
git clone https://github.com/goodfire-ai/arithmetic-wild.git && (cd arithmetic-wild && uv sync && uv pip install ripser GraphRicciCurvature networkx pandas scikit-learn matplotlib pyyaml)
# put concept_geometry.py from hypo_free_geometry next to das_geometry.py's ROOT (/workspace)
arithmetic-wild/.venv/bin/python das/das_geometry.py
arithmetic-wild/.venv/bin/python das/plot_das_clouds.py

git clone https://github.com/jiudingsun01/HyperDAS.git && (cd HyperDAS && git apply ../hyperdas/hyperdas_llama31.patch)
# separate env: python 3.10, torch 2.4.1 (cu124), transformers 4.44.2, datasets 2.19.1, accelerate 0.34.0, matplotlib, seaborn, wandb
arithmetic-wild/.venv/bin/python hyperdas/build_hyperdas_data.py --per-base 10
python hyperdas/train_hyperdas.py --epochs 3
```

Paths in the scripts assume everything lives under `/workspace`.
