#!/bin/bash
# Months, output, plain DAS module, random init. U1: from scratch with the rotation outside the grad clip.
# U2: 1 epoch with the token choice fixed to last <- last, then 2 epochs with HyperDAS choosing tokens.
source /etc/rp_environment
cd /workspace
common="--variables output --tasks months --subspace-module DAS --das-dim 8 --eval-steps 100"
hdenv/bin/python train_hyperdas_v2.py $common --epochs 3 --clip-without-rotation \
    --save-dir /workspace/results/u1_noclip > results/u1_noclip.log 2>&1
rm -rf /workspace/results/u1_noclip/final_model
hdenv/bin/python train_hyperdas_v2.py $common --epochs 1 --force-last-token \
    --save-dir /workspace/results/u2_stage1 > results/u2_stage1.log 2>&1
hdenv/bin/python train_hyperdas_v2.py $common --epochs 2 --load-from /workspace/results/u2_stage1/final_model \
    --save-dir /workspace/results/u2_stage2 > results/u2_stage2.log 2>&1
echo done > results/unforced_done
