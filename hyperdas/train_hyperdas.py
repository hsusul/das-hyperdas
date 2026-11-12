"""Train HyperDAS on the arithmetic tasks with Llama-3.1-8B (wraps HyperDAS/train.py)."""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset
from tokenizers import processors
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/HyperDAS")
os.chdir("/workspace/HyperDAS")

# HyperDAS's collate prepends bos_token itself; Llama-3's tokenizer adds another,
# which drops months accuracy to ~63% even without an intervention.
_from_pretrained = AutoTokenizer.from_pretrained


def _single_bos_tokenizer(*args, **kwargs):
    tok = _from_pretrained(*args, **kwargs)
    tok._tokenizer.post_processor = processors.TemplateProcessing(single="$A", pair="$A $B", special_tokens=[])
    return tok


AutoTokenizer.from_pretrained = _single_bos_tokenizer
from train import run_experiment  # noqa: E402
from src.hyperdas.llama3.model import RavelInterpreterHypernetwork  # noqa: E402


def split_trailing_space(r: dict) -> dict:
    """Hours prompts end with a space, which is the token DAS intervenes on. Collate adds a space
    before the target unless the suffix already ends with one, so move that space into the suffix:
    the prompt keeps its last token and the label is not shifted."""
    for side in ("input", "counterfactual_input"):
        if r[f"{side}_prefix"].endswith(" "):
            r[f"{side}_prefix"], r[f"{side}_suffix"] = r[f"{side}_prefix"].rstrip(" "), " "
    return r


def to_disk(jsonl: Path, limit: int | None, variables: list[str], tasks: list[str]) -> str:
    rows = [r for r in map(json.loads, open(jsonl)) if r["variable"] in variables and r["task"] in tasks][:limit]
    for r in rows:
        split_trailing_space(r)
    out = jsonl.with_suffix("")
    Dataset.from_list(rows).save_to_disk(str(out))
    return str(out)


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/workspace/hyperdas_data")
parser.add_argument("--save-dir", default="/workspace/results/hyperdas")
parser.add_argument("--layer", type=int, default=19, help="hidden_states index; 19 = output of layer 18")
parser.add_argument("--das-dim", type=int, default=8)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=4e-5)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--eval-steps", type=int, default=200)
parser.add_argument("--no-sparsity", action="store_true", help="turn off the source-token selection sparsity loss")
parser.add_argument("--causal-weight", type=float, default=3.5)
parser.add_argument("--variables", nargs="+", default=["input", "offset", "output"])
parser.add_argument("--subspace-module", default="ReflectSelect", choices=["ReflectSelect", "DAS", "MaskSelect", "BoundlessDAS"])
parser.add_argument("--tasks", nargs="+", default=["months", "weekdays", "hours"])
parser.add_argument("--init-das", default=None, help="npz with a DAS basis W to load into the DAS module")
parser.add_argument("--freeze-das", action="store_true", help="keep that basis fixed; only token selection trains")
parser.add_argument("--init-pca", action="store_true",
                    help="start the basis at random inside the top PCs (90%% variance) of the training prompts' "
                         "activations at the intervention layer, as arithmetic-wild does for DAS")
parser.add_argument("--force-last-token", action="store_true",
                    help="fix token selection to base last prompt token <- source last token; only the subspace learns")
parser.add_argument("--rotate-lr", type=float, default=None, help="learning rate for the rotation (HyperDAS default 1e-3)")
parser.add_argument("--clip-without-rotation", action="store_true",
                    help="leave the rotation out of the global grad-norm clip (4.0 over every trainable parameter)")
parser.add_argument("--load-from", default=None, help="final_model dir of an earlier run to continue from")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    import random

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

_save_model = RavelInterpreterHypernetwork.save_model


def _save_with_rotation(self, save_dir):
    """Also write the learned basis as a small .npy (the DAS module's subspace is just this matrix)."""
    _save_model(self, save_dir)
    import numpy as np

    W = self.interpreter.das_module.rotate_layer.weight.detach().float().cpu().numpy()
    np.save(os.path.join(save_dir, "rotation.npy"), W)


RavelInterpreterHypernetwork.save_model = _save_with_rotation

if args.init_das or args.init_pca:
    import numpy as np
    import torch

    _init = RavelInterpreterHypernetwork.__init__

    @torch.no_grad()
    def prompt_activations(target_model):
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        tok.pad_token, tok.padding_side = tok.eos_token, "left"
        rows = [json.loads(line) for line in open(Path(args.data) / "train.jsonl")]
        prompts = sorted({r["input_prefix"].rstrip(" ") for r in rows if r["task"] in args.tasks})
        target_model.to("cuda")  # run_experiment moves everything to cuda right after __init__ anyway
        acts = []
        for i in range(0, len(prompts), 32):
            enc = tok([tok.bos_token + p for p in prompts[i:i + 32]], return_tensors="pt", padding=True).to("cuda")
            hs = target_model(**enc, output_hidden_states=True).hidden_states[args.layer]
            acts.append(hs[:, -1].float().cpu())
        print(f"random-pca init from {len(prompts)} prompts")
        return torch.cat(acts).numpy()

    def _init_from_das(self, *a, **k):
        _init(self, *a, **k)
        if args.init_das:
            W = torch.tensor(np.load(args.init_das, allow_pickle=True)["W"], dtype=torch.float32)
        else:  # arithmetic-wild's random-pca init, over individual training prompts (no labels)
            X = prompt_activations(self.interpreter.target_model)
            _, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
            n_pc = max(int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.9)) + 1, args.das_dim)
            Q, _ = np.linalg.qr(Vt[:n_pc].T @ np.random.default_rng(0).standard_normal((n_pc, args.das_dim)))
            W = torch.tensor(Q, dtype=torch.float32)
        rotate = self.interpreter.das_module.rotate_layer
        with torch.no_grad():
            rotate.weight = W
        if args.freeze_das:
            rotate.requires_grad_(False)

    RavelInterpreterHypernetwork.__init__ = _init_from_das

os.makedirs(args.save_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.save_dir, "args.json"), "w"), indent=2)
if args.force_last_token or args.rotate_lr or args.clip_without_rotation:
    import torch

    _init2 = RavelInterpreterHypernetwork.__init__
    rotation_params: set[int] = set()

    def _init_options(self, *a, **k):
        _init2(self, *a, **k)
        if args.rotate_lr:
            self.rotate_lr = args.rotate_lr
        rotation_params.update(id(p) for p in self.interpreter.das_module.rotate_layer.parameters())

    RavelInterpreterHypernetwork.__init__ = _init_options

if args.force_last_token:
    _forward = RavelInterpreterHypernetwork.forward

    def _forward_last_token(self, *a, **k):
        if k.get("intervention_weight") is None:
            labels, n = k["labels"], k["labels"].shape[0]
            base_len, source_len = labels.shape[1], k["source_input_ids"].shape[1]
            last_prompt = (labels != -100).float().argmax(-1) - 1
            W = torch.zeros(n, source_len + 1, base_len, device=labels.device, dtype=torch.bfloat16)
            W[:, -1, :] = 1  # every base position keeps itself...
            rows = torch.arange(n, device=labels.device)
            W[rows, -1, last_prompt] = 0  # ...except the last prompt token, which takes the source's last token
            W[rows, source_len - 1, last_prompt] = 1
            k["intervention_weight"] = W
        return _forward(self, *a, **k)

    RavelInterpreterHypernetwork.forward = _forward_last_token

if args.clip_without_rotation:
    _clip = torch.nn.utils.clip_grad_norm_

    def _clip_others(parameters, *a, **k):
        return _clip([p for p in parameters if id(p) not in rotation_params], *a, **k)

    torch.nn.utils.clip_grad_norm_ = _clip_others

run_experiment(
    log_wandb=False,
    load_trained_from=args.load_from,
    intervention_layer=args.layer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    train_path=to_disk(Path(args.data) / "train.jsonl", args.limit, args.variables, args.tasks),
    test_path=to_disk(Path(args.data) / "test.jsonl", args.limit and args.limit // 4, args.variables, args.tasks),
    das_dimension=args.das_dim,
    subspace_module=args.subspace_module,
    n_epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    eval_per_steps=args.eval_steps,
    save_dir=args.save_dir,
    save_model=True,
    checkpoint_per_steps=None,  # only the final model; each checkpoint is ~5GB
    # our suffixes are empty; with visibility off, collate's mask[-0:] = 0 blanks the
    # whole source mask, and the variables live at the last prompt token anyway
    source_suffix_visibility=True,
    base_suffix_visibility=True,
    # train.py CLI defaults (run_experiment's own defaults differ)
    num_decoders=2,
    source_selection_sparsity_loss=not args.no_sparsity,
    sparsity_loss_warm_up_ratio=0.5,
    sparsity_loss_weight_start=0.15,
    sparsity_loss_weight_end=1.5,
    causal_loss_weight=args.causal_weight,
    iso_loss_weight=0.5,
    inference_modes=["default", "bidding_argmax"],
)
