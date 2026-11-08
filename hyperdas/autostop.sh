#!/bin/bash
# Stop this pod once nothing is training and nobody has touched /workspace/keepalive for 30 minutes.
source /etc/rp_environment
cd /workspace
touch keepalive
while true; do
    sleep 60
    pgrep -f train_hyperdas_v2.py >/dev/null && continue
    [ $(( $(date +%s) - $(stat -c %Y keepalive) )) -lt 1800 ] && continue
    runpodctl pod stop "$RUNPOD_POD_ID"
done
