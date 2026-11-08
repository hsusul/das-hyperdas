#!/bin/bash
# Subspace learning with token choice fixed to last <- last (months, output, plain DAS module, random init).
source /etc/rp_environment
cd /workspace
common="--variables output --tasks months --subspace-module DAS --das-dim 8 --force-last-token --epochs 3 --eval-steps 100"
run() { name=$1; shift
    hdenv/bin/python train_hyperdas_v2.py $common --save-dir /workspace/results/forced_$name "$@" > results/forced_$name.log 2>&1
    rm -rf /workspace/results/forced_$name/final_model /workspace/HyperDAS/results/forced_$name
    echo "$name done" >> results/forced_progress.txt; }
run default
run noclip --clip-without-rotation
run rotlr1e-4 --rotate-lr 1e-4
