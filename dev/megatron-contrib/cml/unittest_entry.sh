#!/usr/bin/env bash
# Run MFSDP v2 unit tests inside a cml job with the uv venv (Python 3.12 +
# torch 2.14.1+cu130) that lives on /planning-users, instead of the image's
# system Python 3.10 / torch 2.6.
#
# Env (all optional): RUN_NAME (result dir name), GPUS_PER_NODE (default
# RESOURCE_GPU or 8), TEST_FILES (space-separated paths relative to the dev
# repo), PYTEST_ARGS (extra pytest args), TEST_TIMEOUT (seconds, default 3600).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
source "$CONTRIB/env/mfsdp-dev.sh"   # sets PATH/PYTHONPATH/DEV_REPO

GPUS_PER_NODE=${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}
TEST_FILES=${TEST_FILES:-"tests/unit_tests/distributed/mfsdp_v2/test_overlap.py tests/unit_tests/distributed/mfsdp_v2/test_cuda_graph.py"}
OUT_DIR=${OUT_DIR:-$CONTRIB/results/unittest/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$OUT_DIR"

{
    echo "host=$(hostname) gpus_per_node=$GPUS_PER_NODE"
    echo "repo=$DEV_REPO commit=$(git -C "$DEV_REPO" rev-parse HEAD)"
    git -C "$DEV_REPO" status --short
    python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version(), 'gpus', torch.cuda.device_count())"
    nvidia-smi
    nvidia-smi topo -m
    env | grep -E '^(NCCL|CUDA|RANK|WORLD_SIZE|MASTER_|RESOURCE_)' | sort
} > "$OUT_DIR/sysinfo.txt" 2>&1 || true
cat "$OUT_DIR/sysinfo.txt" | head -5

if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi

cd "$DEV_REPO"
# Single-node test: ignore the platform's multi-node rendezvous variables.
# --tee 3 prefixes every line with the rank and keeps per-rank logs on disk.
# shellcheck disable=SC2086
timeout "${TEST_TIMEOUT:-3600}" python -m torch.distributed.run \
    --standalone --nproc-per-node "$GPUS_PER_NODE" \
    --log-dir "$OUT_DIR/torchrun" --tee 3 \
    -m pytest -q -p no:cacheprovider $TEST_FILES ${PYTEST_ARGS:-} \
    2>&1 | tee "$OUT_DIR/run.log"
