"""Train HyperDAS on the arithmetic tasks with Llama-3.1-8B (wraps HyperDAS/train.py)."""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, "/workspace/HyperDAS")
os.chdir("/workspace/HyperDAS")
from train import run_experiment  # noqa: E402


def to_disk(jsonl: Path, limit: int | None) -> str:
    rows = [json.loads(line) for line in open(jsonl)][:limit]
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
    # train.py CLI defaults (run_experiment's own defaults differ)
    num_decoders=2,
    sparsity_loss_warm_up_ratio=0.5,
    sparsity_loss_weight_start=0.15,
    sparsity_loss_weight_end=1.5,
    causal_loss_weight=3.5,
    iso_loss_weight=0.5,
    inference_modes=["default", "bidding_argmax"],
)
