from tokenizers import processors
from transformers import AutoTokenizer
_fp = AutoTokenizer.from_pretrained
def _single_bos(*a, **k):
    t = _fp(*a, **k); t._tokenizer.post_processor = processors.TemplateProcessing(single="$A", pair="$A $B", special_tokens=[]); return t
AutoTokenizer.from_pretrained = _single_bos
