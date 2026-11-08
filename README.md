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

## HyperDAS

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

### Result so far: HyperDAS does not learn this task (yet)

| Run | Data | Causal acc | Isolate acc |
|---|---|---|---|
| First full run, before the fixes above | all 3 variables, 33k pairs, 3 epochs | ~5% | ~35% |
| Mixed, after fixes (stopped at step 2000) | all 3 variables | ~5% | ~40% |
| Output only, after fixes (`results/hyperdas_output_only/`) | `output` only, 3 epochs | months 9%, weekdays 17%, hours 4% | 20-49% |

For comparison, `results/patching_check.txt` swaps only the released 8-d DAS output subspace at the same site (output of layer 18, last token) with plain transformers hooks: 84% causal for months, 97% for weekdays. So a working 8-d subspace exists there and the labels are right; HyperDAS is not finding it.

Two things this showed:

- At the output of layer 18, only the `output` variable can be swapped. Swapping the whole last-token vector moves `input`/`offset` answers only 8-22% of the time, because the MLP has already combined them into the sum. The paper localizes those two right after attention (`post_attn_resid`), which HyperDAS's layer-output hook cannot reach. Training on all three variables at this site asked for something that is not there.
- In the output-only model, every example gets the same reflection vector (`subspace_agreement.json`, agreement 1.000): the hypernetwork ignores its inputs. The geometry rows in `results/hyperdas_output_only/topology_vectors.csv` therefore describe an untrained-looking subspace and should not be read as a HyperDAS result.

### Diagnosis trials (`results/hyperdas_trials/`)

All on the `output` variable, layer 18 output, subspace dim 8 unless noted. Causal/isolate accuracy on held-out pairs, final eval, HyperDAS's default (soft) inference mode.

| Trial | Setup | Causal | Isolate | Reading |
|---|---|---|---|---|
| `forced_intervention.py` | HyperDAS's own intervention code, token choice set by hand (last base token <- last source token), released DAS basis | months 84%, weekdays 97% | - | Intervention code is fine |
| `frozen_das` | months; released DAS basis loaded and frozen; only token selection trains | 84% (by step 100) | 87% | Token selection learns fine |
| `dim128` | all tasks, ReflectSelect, dim 128, 1 epoch | 6% | 11% | Bigger subspace does not help |
| `dasmodule` | all tasks, plain DAS module, stopped at step 250 | 4% | 96% (no-op) | - |
| `months_random` | months, plain DAS module, random init, 3 epochs | 13% | 7% | Subspace learning fails from random init |
| `months_pca` | months, plain DAS module, init inside top PCs of the 32 per-sum mean activations, 3 epochs | 87% | 94% | Works, but the init used the answer grouping |
| `per_prompt_pca_months` | same, but PCs of the 190 individual prompts (no labels), as arithmetic-wild does | 10% | 9% | Fails without the label-informed init |
| `reflect_all_tasks_per_prompt_pca` | all tasks, ReflectSelect, per-prompt PCA init, 3 epochs | 6% | 40% | Fails |

Conclusion so far: on these tasks HyperDAS learns which tokens to swap, but it does not learn the subspace itself unless it starts almost inside the answer subspace. Plain DAS with a fixed last-token position (the released subspaces) does find it. The `months_pca` result should not be read as HyperDAS discovering the subspace independently.

Not tried: more epochs or data for the from-scratch runs, a separate (higher or lower) learning rate for the rotation, and hooking `post_attention_layernorm` for the `input`/`offset` variables.

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
