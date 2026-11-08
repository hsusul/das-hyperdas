#!/bin/bash
# One HyperDAS (plain DAS module, random-pca init) per task on the output variable, then geometry; auto-stop.
source /etc/rp_environment
cd /workspace
for p in $(pgrep -f run_full_pca.sh) $(pgrep -f hyperdas_geometry.py); do kill $p; done
for task in months weekdays hours; do
    hdenv/bin/python train_hyperdas_v2.py --variables output --tasks $task --subspace-module DAS --das-dim 8 --init-pca \
        --epochs 3 --eval-steps 200 --save-dir results/hd_das_$task > results/hd_das_$task.log 2>&1
    hdenv/bin/python hyperdas_geometry.py --model-dir results/hd_das_$task/final_model --subspace-module DAS \
        --variables output --tasks $task --out results/hd_das_geometry_$task > results/hd_das_geometry_$task.log 2>&1
    echo "$task analysis exit $?" >> results/per_task_progress.txt
    rm -rf results/hd_das_$task/final_model
done
touch keepalive
while [ $(( $(date +%s) - $(stat -c %Y keepalive) )) -lt 2700 ]; do sleep 60; done
runpodctl pod stop "$RUNPOD_POD_ID"
