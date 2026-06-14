#!/bin/bash
set -e
redis-server --daemonize yes
sleep 2  # Give Redis a moment to start up
echo "*****Running text trainer"
source /workspace/.grpo_env/bin/activate
python -m text_trainer "$@"
# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line; does not change behavior.
