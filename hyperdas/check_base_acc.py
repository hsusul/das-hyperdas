"""No-intervention accuracy of Llama-3.1-8B on HyperDAS-collated base inputs."""
import json, sys, torch
sys.path.insert(0, "/workspace/HyperDAS")
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hyperdas.data_utils import get_ravel_collate_fn

M = "meta-llama/Llama-3.1-8B"
tok = AutoTokenizer.from_pretrained(M); tok.pad_token = tok.eos_token; tok.padding_side = "left"; tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
rows = [json.loads(l) for l in open("/workspace/hyperdas_data/test.jsonl")]
iso = [r for r in rows if r["attribute_type"] == "isolate"][:256]
collate = get_ravel_collate_fn(tok, add_space_before_target=True)
by_task = {}
for i in range(0, len(iso), 16):
    batch = iso[i:i + 16]
    b = collate(batch)
    with torch.no_grad():
        logits = model(input_ids=b["base_input_ids"].cuda(), attention_mask=b["base_attention_mask"].cuda()).logits
    pred = logits[:, :-1].argmax(-1).cpu(); lab = b["labels"][:, 1:]
    for j, r in enumerate(batch):
        m = lab[j] != -100
        ok = bool((pred[j][m] == lab[j][m]).all())
        by_task.setdefault(r["task"], []).append(ok)
        if not ok and len(by_task[r["task"]]) - sum(by_task[r["task"]]) <= 2:
            print(r["task"], "label", repr(tok.decode(lab[j][m])), "pred", repr(tok.decode(pred[j][m])))
for t, v in by_task.items():
    print(t, f"{sum(v)}/{len(v)} = {sum(v)/len(v):.2f}")
