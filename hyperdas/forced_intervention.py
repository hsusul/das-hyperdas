"""HyperDAS's own intervention path with the token choice and subspace fixed by hand.

Base last prompt token <- source last prompt token, through the released 8-d DAS
output subspace loaded into HyperDAS's plain DAS module. Plain hooks get 84% (months)
and 97% (weekdays) here, so this separates broken plumbing from bad learned selection.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/workspace")
import train_hyperdas_tok  # noqa: F401
sys.path.insert(0, "/workspace/HyperDAS")
from transformers import AutoTokenizer
from src.hyperdas.data_utils import get_ravel_collate_fn
from src.hyperdas.llama3.model import RavelInterpreterHypernetwork

M = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(M); tok.pad_token, tok.padding_side, tok.pad_token_id = tok.eos_token, "left", tok.eos_token_id
h = RavelInterpreterHypernetwork(model_name_or_path=M, num_editing_heads=32, intervention_layer=19,
                                 subspace_module="DAS", das_dimension=8, chop_editor_at_layer=2).cuda().eval()
it = h.interpreter
collate = get_ravel_collate_fn(tok, add_space_before_target=True, source_suffix_visibility=True, base_suffix_visibility=True)
rows = [json.loads(l) for l in open("/workspace/hyperdas_data/test.jsonl")]
for r in rows:
    r["input_prefix"] = r["input_prefix"].rstrip(" "); r["counterfactual_input_prefix"] = r["counterfactual_input_prefix"].rstrip(" ")

for task in ["months", "weekdays"]:
    W = np.load(f"/workspace/das_geometry/{task}_output_L18.npz", allow_pickle=True)["W"]
    with torch.no_grad():
        it.das_module.rotate_layer.weight = torch.tensor(W, device="cuda", dtype=torch.float32)
    got = it.das_module.rotate_layer.weight.detach().float().cpu().numpy()
    print(task, "loaded W error", np.abs(got - W).max())
    sel = [r for r in rows if r["task"] == task and r["variable"] == "output" and r["attribute_type"] == "causal"][:128]
    hits = {"forced": [], "no_intervention": []}
    for i in range(0, len(sel), 16):
        b = {k: v.cuda() for k, v in collate(sel[i:i + 16]).items()}
        lab = b["labels"]; n, bl = lab.shape; sl = b["source_input_ids"].shape[1]
        first = (lab != -100).float().argmax(-1)          # first label position
        for mode in hits:
            Wt = torch.zeros(n, sl + 1, bl, device="cuda", dtype=torch.bfloat16); Wt[:, -1, :] = 1
            if mode == "forced":
                for j in range(n):
                    Wt[j, -1, first[j] - 1] = 0; Wt[j, sl - 1, first[j] - 1] = 1
            with torch.no_grad():
                out = it(editor_input_ids=b["editor_input_ids"], editor_attention_mask=b["editor_input_ids"] != tok.eos_token_id,
                         base_input_ids=b["base_input_ids"], base_attention_mask=b["base_attention_mask"],
                         base_intervention_mask=b["base_intervention_mask"], source_input_ids=b["source_input_ids"],
                         source_attention_mask=b["source_attention_mask"], source_intervention_mask=b["source_intervention_mask"],
                         intervention_weight=Wt)
            pred = out.logits[:, :-1].argmax(-1); l = lab[:, 1:]; m = l != -100
            hits[mode] += ((pred == l) | ~m).all(-1).tolist()
    print(task, {k: round(float(np.mean(v)), 3) for k, v in hits.items()}, "(no_intervention should match the source answer rarely)")
