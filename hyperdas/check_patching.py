"""Plumbing check without HyperDAS: last-token interchange at the output of layer 18.

full : replace the whole residual vector with the source's
das  : replace only its component in the released DAS output subspace
Accuracy = first predicted token matches the first token of counterfactual_target.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/workspace")
import train_hyperdas_tok  # noqa: F401  single-BOS tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

M, LAYER = "meta-llama/Llama-3.1-8B", 18
tok = AutoTokenizer.from_pretrained(M); tok.pad_token = tok.eos_token; tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
rows = [json.loads(l) for l in open("/workspace/hyperdas_data/test.jsonl")]

def enc(prompts):
    return tok([tok.bos_token + p.rstrip(" ") for p in prompts], return_tensors="pt", padding=True).to("cuda")

@torch.no_grad()
def run(batch, W=None):
    src = model(**enc([r["counterfactual_input_prefix"] for r in batch]), output_hidden_states=True).hidden_states[LAYER + 1][:, -1]
    def hook(m, a, out):
        h = out[0]
        if W is None:
            h[:, -1] = src
        else:
            h[:, -1] = h[:, -1] + ((src - h[:, -1]).float() @ W @ W.T).to(h.dtype)
    handle = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        logits = model(**enc([r["input_prefix"] for r in batch])).logits[:, -1]
    finally:
        handle.remove()
    pred = logits.argmax(-1).tolist()
    gold = [tok(" " + r["counterfactual_target"] if r["task"] != "hours" else " " + r["counterfactual_target"])["input_ids"][0] for r in batch]
    return [p == g for p, g in zip(pred, gold)]

for task in ["months", "weekdays", "hours"]:
    W = torch.tensor(np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"], device="cuda")
    for var in ["input", "offset", "output"]:
        for kind in ["causal", "isolate"]:
            sel = [r for r in rows if r["task"] == task and r["variable"] == var and r["attribute_type"] == kind][:128]
            res = {"full": [], "das_out": []}
            for i in range(0, len(sel), 32):
                res["full"] += run(sel[i:i + 32]); res["das_out"] += run(sel[i:i + 32], W)
            print(f"{task:8s} {var:6s} {kind:7s} n={len(sel):3d} " + " ".join(f"{k}={np.mean(v):.2f}" for k, v in res.items()))
