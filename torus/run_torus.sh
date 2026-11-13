#!/bin/bash
# Torus (weekday x hour) on Llama-3.1-8B: prompts, HyperDAS for day and hour (2 seeds, warm-up recipe), geometry.
source /etc/rp_environment
cd /workspace
log() { echo "$(date -u +%H:%M) $*" >> results/progress.txt; }
mkdir -p results
./setup_pod.sh > setup.log 2>&1 && VIRTUAL_ENV=/workspace/hdenv uv pip install -q persim >> setup.log 2>&1 && log setup
hdenv/bin/python torus_prompts.py > results/torus_prompts.log 2>&1 && log prompts
python3 torus_data.py > results/torus_data.log 2>&1 && log data
for seed in 0 1; do
    for var in day hour; do
        common="--data /workspace/torus_data --tasks torus --variables $var --subspace-module DAS --das-dim 8 --eval-steps 200 --seed $seed"
        hdenv/bin/python train_hyperdas.py $common --epochs 3 --force-last-token \
            --save-dir /workspace/results/hd_torus_${var}_seed${seed}_stage1 > results/hd_torus_${var}_seed${seed}_stage1.log 2>&1
        hdenv/bin/python train_hyperdas.py $common --epochs 3 --load-from /workspace/results/hd_torus_${var}_seed${seed}_stage1/final_model \
            --save-dir /workspace/results/hd_torus_${var}_seed${seed}_stage2 > results/hd_torus_${var}_seed${seed}_stage2.log 2>&1
        rm -f /workspace/results/hd_torus_${var}_seed${seed}_stage*/final_model/hypernetwork.pt
        log "trained $var seed $seed"
    done
done
hdenv/bin/python torus_geometry.py > results/torus_geometry.log 2>&1 && log geometry
