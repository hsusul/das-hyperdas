"""No-intervention accuracy on hours with the trailing space kept as the last prompt token."""
import json, sys, torch
sys.path.insert(0, "/workspace")
import train_hyperdas_tok  # noqa: F401
sys.path.insert(0, "/workspace/HyperDAS")
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hyperdas.data_utils import get_ravel_collate_fn
from hyperdas_geometry import strip

M = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(M); tok.pad_token, tok.padding_side, tok.pad_token_id = tok.eos_token, "left", tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
rows = [strip(json.loads(l)) for l in open("/workspace/hyperdas_data/test.jsonl")]
rows = [r for r in rows if r["task"] == "hours" and r["attribute_type"] == "isolate"][:128]
b = get_ravel_collate_fn(tok, add_space_before_target=True, source_suffix_visibility=True, base_suffix_visibility=True)(rows[:4])
lab = b["labels"][0]; print("label:", repr(tok.decode(lab[lab != -100])), "| prompt tail:", [tok.decode(t) for t in b["base_input_ids"][0][lab == -100][-3:]])
hits = []
for i in range(0, len(rows), 16):
    b = get_ravel_collate_fn(tok, add_space_before_target=True, source_suffix_visibility=True, base_suffix_visibility=True)(rows[i:i + 16])
    with torch.no_grad():
        lg = model(input_ids=b["base_input_ids"].cuda(), attention_mask=b["base_attention_mask"].cuda()).logits
    pred, l = lg[:, :-1].argmax(-1).cpu(), b["labels"][:, 1:]
    hits += ((pred == l) | (l == -100)).all(-1).tolist()
print(f"hours no-intervention accuracy: {sum(hits)}/{len(hits)}")
