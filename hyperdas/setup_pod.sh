#!/bin/bash
# Fresh A100 pod: HyperDAS + patches, a py3.10 env pinned to transformers 4.44.2, and the geometry deps.
source /etc/rp_environment
cd /workspace
git clone -q https://github.com/jiudingsun01/HyperDAS.git
(cd HyperDAS && git apply /workspace/hyperdas_llama31.patch)
git clone -q https://github.com/goodfire-ai/arithmetic-wild.git
pip install -q uv
uv venv -q -p 3.10 hdenv
VIRTUAL_ENV=/workspace/hdenv uv pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=/workspace/hdenv uv pip install -q transformers==4.44.2 datasets==2.19.1 accelerate==0.34.0 wandb tqdm pandas numpy==1.26.4 scikit-learn h5py tiktoken blobfile matplotlib seaborn hf_transfer ripser GraphRicciCurvature networkx
hdenv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B', allow_patterns=['*.json','*.safetensors'])"
echo SETUP_DONE
