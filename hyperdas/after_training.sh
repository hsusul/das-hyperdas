#!/bin/bash
# Wait for HyperDAS training, run the geometry analysis, then stop the pod once
# nobody has touched /workspace/keepalive for 45 minutes.
source /etc/rp_environment
cd /workspace
while pgrep -f train_hyperdas.py >/dev/null; do sleep 60; done
hdenv/bin/python hyperdas_geometry.py > results/hyperdas_geometry.log 2>&1
echo "analysis exit $?" >> results/hyperdas_geometry.log
touch keepalive
while [ $(( $(date +%s) - $(stat -c %Y keepalive) )) -lt 2700 ]; do sleep 60; done
runpodctl pod stop "$RUNPOD_POD_ID"
