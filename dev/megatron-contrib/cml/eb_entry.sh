#!/usr/bin/env bash
# Entry of experiment B (#6714 gather side): GEMM vs NCCL all-gather SM contention,
# single node, all GPUs in one NVLink group. Uses the uv venv on /planning-users.
#
# Launches (separate torchrun runs, since NCCL_MAX_CTAS is read at communicator init):
#   sweep_default, sweep_ctas4, sweep_ctas16: tokens x carveout x priority x mode
#   nsys: short run with nsys on local rank 0 (T=1024, carveout 0/16), for kernel-level
#         start positions; kept separate so nsys does not bias the sweep timings.
# Env: RUN_NAME, GPUS_PER_NODE, EB_ARGS (extra args for sweeps), EB_CTAS ("4 16"),
#      EB_NSYS (1 = run the nsys launch, default 1).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
source "$CONTRIB/env/mfsdp-dev.sh"
GPUS_PER_NODE=${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}
EB_CTAS=${EB_CTAS-"4 16"}
OUT_DIR=${OUT_DIR:-$CONTRIB/results/eb/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$OUT_DIR"
unset WORLD_SIZE RANK CUDA_DEVICE_MAX_CONNECTIONS NCCL_MAX_CTAS NCCL_MIN_CTAS

{
    echo "host=$(hostname) gpus=$GPUS_PER_NODE"
    echo "repo=$DEV_REPO commit=$(git -C "$DEV_REPO" rev-parse HEAD)"
    python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
    nvidia-smi
    nvidia-smi topo -m
    env | grep -E '^(NCCL|CUDA)' | sort
} > "$OUT_DIR/sysinfo.txt" 2>&1 || true
head -3 "$OUT_DIR/sysinfo.txt"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi

port=29600
run() {  # run <timeout s> <log name> <bench args...>
    local limit=$1 name=$2; shift 2
    timeout "$limit" python -m torch.distributed.run --standalone --nproc-per-node "$GPUS_PER_NODE" \
        --master-port "$port" --no-python \
        bash "$CONTRIB/cml/e1_rank.sh" -W ignore "$CONTRIB/bench/eb_sm_contention.py" \
        --out-dir "$OUT_DIR" "$@" 2>&1 | tee "$OUT_DIR/$name.log"
    port=$((port + 1))
}

run 1800 sweep_default --tag default ${EB_ARGS:-}
for c in $EB_CTAS; do
    NCCL_MAX_CTAS=$c run 1800 "sweep_ctas$c" --tag "ctas$c" ${EB_ARGS:-}
done
if [ "${EB_NSYS:-1}" = 1 ]; then
    mkdir -p "$OUT_DIR/nsys"
    NSYS_OUT="$OUT_DIR/nsys/eb_T1024" run 900 nsys --tag nsys --tokens 1024 --carveout 0,16 \
        --warmup 2 --reps 3 --profile \
        || echo "[eb] nsys run failed; sweep results are unaffected"
fi
echo "[eb] done"
