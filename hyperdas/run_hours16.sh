#!/bin/bash
# Hours with the trailing-space token kept and a 16-d subspace (as DAS used), warm-up recipe, then geometry.
source /etc/rp_environment
cd /workspace
while pgrep -f run_weekdays_long.sh >/dev/null; do sleep 30; done
hdenv/bin/python check_hours_labels.py > results/check_hours_labels.log 2>&1
common="--variables output --subspace-module DAS --das-dim 16 --eval-steps 200 --tasks hours"
hdenv/bin/python train_hyperdas_v2.py $common --epochs 1 --force-last-token \
    --save-dir /workspace/results/hours16_stage1 > results/hours16_stage1.log 2>&1
hdenv/bin/python train_hyperdas_v2.py $common --epochs 2 --load-from /workspace/results/hours16_stage1/final_model \
    --save-dir /workspace/results/hours16_stage2 > results/hours16_stage2.log 2>&1
rm -rf /workspace/results/hours16_stage1/final_model
hdenv/bin/python hyperdas_geometry.py --model-dir /workspace/results/hours16_stage2/final_model --subspace-module DAS \
    --das-dim 16 --variables output --tasks hours --out /workspace/results/warmup_geometry_hours16 > results/warmup_geometry_hours16.log 2>&1
echo "hours16 geometry exit $?" >> results/warmup_progress.txt
