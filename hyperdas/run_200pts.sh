#!/bin/bash
# 200-point clouds on Llama-3.1-8B: data, prompts, HyperDAS (3 seeds per task, warm-up recipe), geometry.
source /etc/rp_environment
cd /workspace
log() { echo "$(date -u +%H:%M) $*" >> results/progress.txt; }
mkdir -p results
./setup_pod.sh > setup.log 2>&1 && log setup
(cd arithmetic-wild && UV_CACHE_DIR=/workspace/uvcache uv sync -q) && log arithmetic-wild env
arithmetic-wild/.venv/bin/python build_hyperdas_data.py --per-base 10 > results/build_data.log 2>&1 && log hyperdas data
hdenv/bin/python make_prompts.py > results/make_prompts.log 2>&1 && log prompts
train() {  # task seed dim warmup_epochs epochs
    common="--variables output --subspace-module DAS --das-dim $3 --eval-steps 500 --tasks $1 --seed $2"
    hdenv/bin/python train_hyperdas.py $common --epochs $4 --force-last-token \
        --save-dir /workspace/results/hd_$1_seed$2_stage1 > results/hd_$1_seed$2_stage1.log 2>&1
    hdenv/bin/python train_hyperdas.py $common --epochs $5 --load-from /workspace/results/hd_$1_seed$2_stage1/final_model \
        --save-dir /workspace/results/hd_$1_seed$2_stage2 > results/hd_$1_seed$2_stage2.log 2>&1
    rm -f /workspace/results/hd_$1_seed$2_stage*/final_model/hypernetwork.pt
    log "trained $1 seed $2"
}
for seed in 0 1 2; do
    train months $seed 8 1 2
    train weekdays $seed 8 8 6
    train hours $seed 16 1 2
done
hdenv/bin/python point_clouds.py > results/point_clouds.log 2>&1 && log point clouds
