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
- `hyperdas/hyperdas_geometry.py` pulls a subspace out of the trained model and runs the topology pipeline in it. ReflectSelect gives each example its own basis (a Householder reflection of one learned basis), so per task and variable it averages the projectors over 256 held-out pairs, keeps the top 8 directions, and reports how well the per-example bases agree with that consensus.
- `hyperdas/after_training.sh` runs the analysis when training ends and stops the pod after 45 idle minutes.

Fixes needed to get HyperDAS working on Llama-3.1 with our prompts:

1. `hyperdas/hyperdas_llama31.patch` (applied to [jiudingsun01/HyperDAS](https://github.com/jiudingsun01/HyperDAS)):
   - the `_update_causal_mask` call signature changed in transformers 4.44.2;
   - the low-rank rotation weight was left uninitialized (`torch.empty`), which gave NaN losses.
2. In `train_hyperdas.py` / `train_hyperdas_tok.py`:
   - The collate function prepends a BOS token and Llama-3's tokenizer adds another; with two, the model got months right only 63% of the time with no intervention. The tokenizer is patched to stop adding its own.
   - Hours prompts end with a space and collate adds another before the target, which shifted the label by one token (0% correct with no intervention). Trailing spaces are stripped from prompts.
   - With empty suffixes, collate runs `mask[-0:] = 0`, which blanks the whole source mask, so the hypernetwork could never select a source token. Both suffix-visibility flags are on, which skips that code. `check_base_acc.py` is the no-intervention sanity check.

The first full run (before fixes 2) finished but learned nothing (causal accuracy ~5%). A rerun with all fixes is in progress.

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
