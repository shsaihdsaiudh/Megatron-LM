#!/usr/bin/env bash
# Entry of E1 (#6714 gather side): exposed DP-outer all-gather in the MFSDP v2
# HFSDP baseline. Uses the uv venv (py3.12, torch 2.14.1+cu130) on /planning-users.
#
# Phases (each a separate torchrun launch, all nodes run the same sequence):
#   1. sweep: for each "hidden:layers" in E1_MODELS, all tokens x GA + nostep control.
#   2. nsys:  for each "tokens,ga" in E1_NSYS (first model only), a short run with
#             nsys on local rank 0 of every node. Kept separate so nsys overhead
#             does not bias the sweep's timings.
#
# Env: RUN_NAME, EXPECT_NODES, MESH_OUTER (2), GPUS_PER_NODE, E1_MODELS
# ("4096:32 8192:16"), E1_ARGS (extra args for every launch), E1_NSYS
# ("256,4 1024,4"; empty disables).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
source "$CONTRIB/env/mfsdp-dev.sh"
# Optional: import megatron_fsdp from a frozen source snapshot instead of the dev repo.
if [ -n "${MFSDP_SRC:-}" ]; then
    export PYTHONPATH="$MFSDP_SRC:$PYTHONPATH"
fi

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}
MESH_OUTER=${MESH_OUTER:-2}
E1_MODELS=${E1_MODELS:-"4096:32 8192:16"}
E1_NSYS=${E1_NSYS-"256,4 1024,4"}
OUT_DIR=${OUT_DIR:-$CONTRIB/results/e1/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$OUT_DIR"
# torchrun reads WORLD_SIZE/RANK itself; it must see per-process values, not the
# platform's node-level ones.
unset WORLD_SIZE RANK

env | sort > "$OUT_DIR/env.$(hostname).txt"
if [ -n "${EXPECT_NODES:-}" ] && [ "$NNODES" != "$EXPECT_NODES" ]; then
    echo "[e1] ABORT: expected $EXPECT_NODES nodes but rendezvous variables give $NNODES." >&2
    exit 3
fi

# E1_CDMC overrides CUDA_DEVICE_MAX_CONNECTIONS (diagnostics); default: unset.
if [ -n "${E1_CDMC:-}" ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=$E1_CDMC
else
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_DEBUG_FILE=$OUT_DIR/nccl.node${NODE_RANK}.%p.log

{
    echo "host=$(hostname) node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
    echo "repo=$DEV_REPO commit=$(git -C "$DEV_REPO" rev-parse HEAD)"
    git -C "$DEV_REPO" status --short
    python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
    python -c "import megatron_fsdp; print('megatron_fsdp', megatron_fsdp.__file__)"
    nvidia-smi
    nvidia-smi topo -m
} > "$OUT_DIR/node${NODE_RANK}.sysinfo.txt" 2>&1 || true
head -3 "$OUT_DIR/node${NODE_RANK}.sysinfo.txt"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi

launch=0
run() {  # run <timeout s> <log name> <bench args...>
    local limit=$1 name=$2; shift 2
    # A fresh port per launch avoids reusing a rendezvous port still in TIME_WAIT.
    timeout "$limit" python -m torch.distributed.run \
        --nnodes="$NNODES" --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port=$((MASTER_PORT + launch)) \
        --nproc_per_node="$GPUS_PER_NODE" --no-python \
        bash "$CONTRIB/cml/e1_rank.sh" -W ignore "$CONTRIB/bench/${E1_BENCH:-e1_exposure.py}" \
        --mesh-outer "$MESH_OUTER" --out-dir "$OUT_DIR" ${E1_ARGS:-} "$@" \
        2>&1 | tee "$OUT_DIR/node${NODE_RANK}.$name.log"
    launch=$((launch + 1))
}

for model in $E1_MODELS; do
    d=${model%%:*}; l=${model##*:}
    run 3600 "sweep_d${d}_L${l}" --hidden "$d" --layers "$l"
done

first=${E1_MODELS%% *}; d=${first%%:*}; l=${first##*:}
for cfg in $E1_NSYS; do
    t=${cfg%%,*}; g=${cfg##*,}
    mkdir -p "$OUT_DIR/nsys"
    NSYS_OUT="$OUT_DIR/nsys/d${d}_L${l}_T${t}_ga${g}" \
        run 900 "nsys_d${d}_T${t}_ga${g}" --hidden "$d" --layers "$l" --tokens "$t" --ga "$g" \
        --control-ga 0 --warmup 3 --steps 3 --profile "$t,$g" --profile-steps 2 \
        --out-name "nsys_d${d}_L${l}_T${t}_ga${g}.json" \
        || echo "[e1] nsys run T=$t GA=$g failed; sweep results are unaffected"
done
echo "[e1] node $NODE_RANK done"
