#!/bin/bash
# Weekdays has ~800 training pairs, so one warm-up epoch is only ~50 steps: use 8 + 6 epochs instead.
source /etc/rp_environment
cd /workspace
while pgrep -f run_warmup_all.sh >/dev/null; do sleep 30; done
common="--variables output --subspace-module DAS --das-dim 8 --eval-steps 100 --tasks weekdays"
hdenv/bin/python train_hyperdas_v2.py $common --epochs 8 --force-last-token \
    --save-dir /workspace/results/warmup_weekdays_long_stage1 > results/warmup_weekdays_long_stage1.log 2>&1
hdenv/bin/python train_hyperdas_v2.py $common --epochs 6 --load-from /workspace/results/warmup_weekdays_long_stage1/final_model \
    --save-dir /workspace/results/warmup_weekdays_long_stage2 > results/warmup_weekdays_long_stage2.log 2>&1
rm -rf /workspace/results/warmup_weekdays_long_stage1/final_model
hdenv/bin/python hyperdas_geometry.py --model-dir /workspace/results/warmup_weekdays_long_stage2/final_model --subspace-module DAS \
    --variables output --tasks weekdays --out /workspace/results/warmup_geometry_weekdays_long > results/warmup_geometry_weekdays_long.log 2>&1
echo "weekdays_long geometry exit $?" >> results/warmup_progress.txt
