#!/bin/bash
# Full HyperDAS run (output variable, all tasks, ReflectSelect, random-pca init), then geometry, then auto-stop.
source /etc/rp_environment
cd /workspace
hdenv/bin/python train_hyperdas_v2.py --variables output --das-dim 8 --init-pca --epochs 3 --eval-steps 500 \
    --save-dir results/hyperdas_output_pca > results/hyperdas_output_pca.log 2>&1
hdenv/bin/python hyperdas_geometry.py --model-dir results/hyperdas_output_pca/final_model --variables output \
    --out results/hyperdas_geometry_pca > results/hyperdas_geometry_pca.log 2>&1
echo "analysis exit $?" >> results/hyperdas_geometry_pca.log
touch keepalive
while [ $(( $(date +%s) - $(stat -c %Y keepalive) )) -lt 2700 ]; do sleep 60; done
runpodctl pod stop "$RUNPOD_POD_ID"
