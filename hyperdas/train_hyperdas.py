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


def to_disk(jsonl: Path, limit: int | None) -> str:
    rows = [json.loads(line) for line in open(jsonl)][:limit]
    # collate adds a space before the target, so a prompt ending in a space
    # (hours) would get two and shift the label by one token
    for r in rows:
        r["input_prefix"] = r["input_prefix"].rstrip(" ")
        r["counterfactual_input_prefix"] = r["counterfactual_input_prefix"].rstrip(" ")
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
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.save_dir, "args.json"), "w"), indent=2)
run_experiment(
    log_wandb=False,
    intervention_layer=args.layer,
    model_name_or_path="meta-llama/Llama-3.1-8B",
    train_path=to_disk(Path(args.data) / "train.jsonl", args.limit),
    test_path=to_disk(Path(args.data) / "test.jsonl", args.limit and args.limit // 4),
    das_dimension=args.das_dim,
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
