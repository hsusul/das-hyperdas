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

These runs showed that HyperDAS learns which tokens to swap, but does not learn the subspace when both are learned from scratch at once. The `months_pca` result should not be read as HyperDAS discovering the subspace independently. The warm-up recipe below fixes this.


### HyperDAS working: warm-up recipe (`results/hyperdas_warmup/`)

What finally isolated the problem (`hyperdas/train_hyperdas.py --force-last-token`): with the token choice fixed to base last prompt token <- source last token, HyperDAS's own training loop learns the subspace from a random init (months: 77% causal, 87% isolate). So the subspace training works; learning tokens and subspace jointly from scratch is what fails, since each needs the other to be roughly right first. Leaving the rotation out of the global grad clip did not help (11%).

Recipe, per task, output variable, plain DAS module, random init, no DAS weights or labels used for initialization:

1. Warm-up: train with the token choice fixed to last <- last (1 epoch; 8 for weekdays, which has ~800 pairs).
2. Release: continue with HyperDAS choosing tokens itself (2 epochs; 6 for weekdays).

The only prior is the warm-up position, which is where DAS always intervenes.

Hours needed one more fix: its prompts end with a space, and that space token is where DAS intervenes and where the answer is produced. Earlier runs stripped it (to avoid a doubled space before the label), which moved the intervention one token too early. Now the space goes into the suffix so the prompt keeps it and the label is not shifted (`check_hours_labels.py`: 126/128 correct with no intervention). Hours uses 16 dims, as DAS did.

Held-out accuracy after training, HyperDAS choosing tokens (`*/accuracy_by_variable.csv`):

| Task | HyperDAS causal | HyperDAS isolate | DAS IIA (released) | Overlap with DAS subspace |
|---|---|---|---|---|
| months (8-d) | 0.86 | 0.98 | 0.85 | 0.51 |
| weekdays (8-d) | 0.98 | 1.00 | 0.97 (plain hooks) | 0.52 |
| hours (16-d) | 0.94 | 0.98 | 0.94 | 0.44 |

Overlap is the mean squared cosine of the principal angles between the two subspaces: 4-6 directions are shared closely, the rest are not. Both subspaces pass the causal test at the same level, so the causally valid subspace is not unique.

Geometry inside the HyperDAS subspace (`warmup_summary.csv`, `warmup_clouds_pca.png`, `magnitude_vs_digit.csv`):

- In raw and DAS space the per-sum points lie on a smooth arc ordered by size. In the HyperDAS subspace they look more coiled, with sums sharing a units digit closer together.
- Measured as the Spearman correlation between pairwise distance and the circular units-digit distance (min(|a-b| mod 10, 10 - |a-b| mod 10)):

| Task | raw | DAS | HyperDAS | random subspaces (20) |
|---|---|---|---|---|
| months | 0.07 | 0.14 | 0.23 | 0.08 +/- 0.05 |
| hours | 0.10 | 0.16 | 0.23 | 0.09 +/- 0.03 |
| weekdays | -0.01 | 0.04 | 0.06 | 0.05 +/- 0.06 |

  Magnitude (|a-b|) still dominates in every space (rho 0.56-0.92). The units-digit signal is modest but above random subspaces for HyperDAS on months (~2.8 sd) and hours (~5.3 sd), and weaker for DAS. Weekdays sums only run 2-21, which leaves little room for a units-digit pattern. This fits the Arithmetic in the Wild account of base-10 features, but it rests on one seed per task and 17-68 points per cloud.
- The PH/curvature topology vectors differ between DAS and HyperDAS subspaces (e.g. months: 13 vs 27 H1 bars, 16% vs 43% negatively curved edges), but not in a consistent direction across tasks.

Not done: more seeds, the `input`/`offset` variables (they need a hook after attention, `post_attention_layernorm`, which HyperDAS's layer-output hook cannot reach), and the instruction-conditioned ReflectSelect module (forcing tokens hits a NameError in its code path, so the warm-up was only run with the plain DAS module).

## 200 points per concept (`results/point_clouds_200/`)

Each concept is now a cloud of 200 individual prompts (one point per prompt, no averaging), drawn at random from all prompts Llama-3.1-8B answers correctly. Three draws per task, so every cloud has the same size.

- Prompts (`hyperdas/make_prompts.py`): every value x offset combination for each template. Months: 3 templates, 510 correct. Weekdays: 5 templates, 226 correct (3 templates gave only 175). Hours: the original template, 700 correct. Templates follow arithmetic-wild's rephrased-templates notebook.
- Subspaces: raw (4096-d), released DAS output subspace, HyperDAS output subspaces from 3 seeds per task (warm-up recipe, trained on the original template only), and 20 random subspaces of the same size.
- HyperDAS held-out causal accuracy per seed (`train_logs/`): months 0.83 / 0.85 / 0.86, weekdays 0.99 / 0.99 / 0.98, hours 0.97 / 0.90 / 0.97.
- Measured per cloud (`hyperdas/point_clouds.py`): the topology vector, plus Spearman correlations between pairwise distances and (a) |sum difference| (magnitude), (b) the circular units-digit distance of the sums (base-10), and (c) the circular distance of the starting concept on its own cycle (Dec next to Jan).

Mean over draws (and seeds for HyperDAS), from `summary.csv`:

| Task | Space | rho magnitude | rho units digit | rho cycle |
|---|---|---|---|---|
| months | raw | 0.16 | 0.05 | 0.07 |
| | DAS | 0.39 | 0.18 | 0.13 |
| | HyperDAS | 0.41 | 0.26 +/- 0.02 | 0.05 |
| | random | 0.12 +/- 0.06 | 0.03 +/- 0.02 | 0.05 +/- 0.03 |
| weekdays | raw | 0.15 | 0.06 | 0.06 |
| | DAS | 0.54 | 0.29 | 0.16 |
| | HyperDAS | 0.47 | 0.32 +/- 0.01 | 0.08 |
| | random | 0.12 +/- 0.06 | 0.05 +/- 0.03 | 0.04 +/- 0.02 |
| hours | raw | 0.44 | 0.18 | 0.19 |
| | DAS | 0.48 | 0.26 | 0.19 |
| | HyperDAS | 0.51 | 0.34 +/- 0.02 | 0.15 |
| | random | 0.27 +/- 0.07 | 0.10 +/- 0.03 | 0.13 +/- 0.03 |

What this shows:

- Units-digit (base-10) structure in the sum subspaces is now clear for all three tasks, in the same order everywhere: HyperDAS > DAS > raw, with random subspaces at the bottom (months HyperDAS is about 15 sd above random). With 12-68 averaged points it was only visible for months and hours. It is stable across HyperDAS seeds (sd 0.01-0.02).
- The HyperDAS subspaces carry less information about the starting concept's cycle than DAS (rho cycle 0.05-0.15 vs 0.13-0.19), consistent with an output subspace that holds the sum, not the input.
- The subspaces themselves are not unique. HyperDAS seeds overlap each other about as much as they overlap DAS (mean squared cosine of principal angles: months 0.57-0.60 between seeds vs 0.49-0.51 with DAS; weekdays 0.55-0.57 vs 0.50-0.54; hours 0.38-0.40 vs 0.39-0.43). All of them pass the causal test, and the base-10 geometry is the same across them even though the subspaces differ.
- Raw clouds mix several templates, which adds variance unrelated to the sum (raw rho magnitude only 0.15-0.16 for months/weekdays).
- PH counts (e.g. raw ~230 H1 bars vs 90-200 in the subspaces) mostly track the dimension left after PCA and are not compared directly.

Caveats: the DAS and HyperDAS subspaces were trained on the original template only and were not causally re-tested on the new templates; one model; output variable only.

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
