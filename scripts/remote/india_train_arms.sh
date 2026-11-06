#!/bin/bash
set -uo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-/home/work}"
EPOCHS="${EPOCHS:-25}"
COMMON="--backbone tresnet_l.miil_in1k_448 --size 368 --epochs $EPOCHS --batch 32 --balance sqrt --eval-every 5"
mkdir -p /home/jl-runs/e18_india_e05init /home/jl-runs/e18_india_imagenet
bash "$PROJECT/scripts/remote/train_india.sh" e18_india_e05init 0 $COMMON --init "$WORK/init/e05_model.pt" \
  > /home/jl-runs/e18_india_e05init/output.log 2>&1 &
A=$!
sleep 20
bash "$PROJECT/scripts/remote/train_india.sh" e18_india_imagenet 1 $COMMON \
  > /home/jl-runs/e18_india_imagenet/output.log 2>&1 &
B=$!
wait $A; echo "e05init exit $?"
wait $B; echo "imagenet exit $?"
