#!/usr/bin/env bash
# Entry of the E0 all-gather microbenchmark (#6714 gather side).
#
# Multi-node rendezvous comes from the platform's MLP_* variables (same as the
# earlier dist_joint_train.sh). Defaults allow a local single-node run.
#
# Env (all optional): MESH_OUTER (default 2), GPUS_PER_NODE (default 8),
# E0_ARGS (extra e0_comm.py arguments), OUT_DIR, NCCL_DEBUG, NCCL_DEBUG_SUBSYS.
# RUN_NAME names the result directory; set it in the job yaml so every node
# writes to the same place. EXPECT_NODES makes the script abort when the
# platform did not provide the multi-node rendezvous variables, instead of
# silently running each node as an independent single-node job.
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
# The platform (PyTorchJob) provides WORLD_SIZE = number of nodes, RANK = node rank,
# MASTER_ADDR/MASTER_PORT and RESOURCE_GPU = GPUs per node; this is what the
# earlier job's dist_train_cml_avp.sh reads. MLP_* are kept as a fallback.
NNODES=${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}
NODE_RANK=${RANK:-${MLP_ROLE_INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}
MESH_OUTER=${MESH_OUTER:-2}
OUT_DIR=${OUT_DIR:-$CONTRIB/results/e0/${RUN_NAME:-${MLP_TASK_ID:-local-$(date +%Y%m%d-%H%M%S)}}}
mkdir -p "$OUT_DIR"

# Always keep the complete environment: the first multi-node run showed that the
# MLP_* variables are not what this image provides, so record what is.
env | sort > "$OUT_DIR/env.$(hostname).txt"
if [ -n "${EXPECT_NODES:-}" ] && [ "$NNODES" != "$EXPECT_NODES" ]; then
    echo "[e0] ABORT: expected $EXPECT_NODES nodes but rendezvous variables give $NNODES." >&2
    echo "[e0] See $OUT_DIR/env.$(hostname).txt for the variables the platform provides." >&2
    exit 3
fi

# A single hardware queue per device would serialize the two streams' kernels.
unset CUDA_DEVICE_MAX_CONNECTIONS
# INIT/NET logs show which transport each communicator uses (NET/IB vs P2P/NVLS).
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
# Force, do not default: the platform presets this variable without NET, and the
# per-channel "via NET/IB" lines needed to verify the DP-outer transport need it.
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE=$OUT_DIR/nccl.node${NODE_RANK}.%p.log

echo "[e0] host=$(hostname) node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT out=$OUT_DIR"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
{
    nvidia-smi
    nvidia-smi topo -m
    ibv_devinfo 2>/dev/null || echo "ibv_devinfo not available"
    env | grep -E '^(NCCL|MLP_|CUDA|UCX|RANK|WORLD_SIZE|MASTER_|RESOURCE_)' | sort
} > "$OUT_DIR/node${NODE_RANK}.sysinfo.txt" 2>&1 || true

python3 -m torch.distributed.run \
    --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    "$CONTRIB/bench/e0_comm.py" \
    --mesh-outer "$MESH_OUTER" --out-dir "$OUT_DIR" ${E0_ARGS:-} \
    2>&1 | tee "$OUT_DIR/node${NODE_RANK}.log"
