#!/usr/bin/env bash
# Single-node A/B for Megatron-LM #6714 / #9539: three arms, interleaved ABBA so that
# drift during the run affects every arm equally.
#   off      #9539 branch, overlap_outer_communication off
#   main-on  main, flag on: DP-outer reduction on its own stream (#6973), DP-outer
#            all-gather still on the all-gather stream
#   pr-on    #9539 branch, flag on: DP-outer reduction and DP-outer all-gather each on
#            their own stream
# Order: off, main-on, pr-on, pr-on, main-on, off. Then prints a summary.
#
# Usage: PR_REPO=/path/to/megatron-lm-9539 MAIN_REPO=/path/to/megatron-lm-main bash run.sh
# Env:
#   NPROC        GPUs to use (default 8)
#   MESH_OUTER   DP-outer size (default 2, so 8 GPUs give mesh (2, 4))
#   OUT_DIR      results directory (default ./results-<timestamp>)
#   ARMS         override the arm sequence, e.g. "off pr-on"
#   NSYS=1       add one short run per PR arm with nsys on rank 0 (T=4096, GA=4)
#   BENCH_ARGS   extra arguments for bench_hfsdp_outer_ag.py
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${PR_REPO:?set PR_REPO to a Megatron-LM checkout of the #9539 branch}"
: "${MAIN_REPO:?set MAIN_REPO to a Megatron-LM checkout of main}"
NPROC=${NPROC:-8}
MESH_OUTER=${MESH_OUTER:-2}
OUT_DIR=${OUT_DIR:-$PWD/results-$(date +%Y%m%d-%H%M%S)}
ARMS=${ARMS:-"off main-on pr-on pr-on main-on off"}
mkdir -p "$OUT_DIR"

{
    for repo in "$PR_REPO" "$MAIN_REPO"; do
        echo "$repo $(git -C "$repo" rev-parse HEAD)"
    done
    python -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c
    echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-unset}"
} | tee "$OUT_DIR/env.txt"

port=29500
launch() {  # launch <name> <repo> <flag> [bench args...]
    local name=$1 repo=$2 flag=$3; shift 3
    echo "[run] $name repo=$repo flag=${flag:-none}"
    PYTHONPATH="$repo:$repo/megatron/core/distributed/fsdp/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        --master_port=$port -- "$HERE/bench_hfsdp_outer_ag.py" \
        --mesh-outer "$MESH_OUTER" $flag ${BENCH_ARGS:-} "$@" 2>&1 | tee "$OUT_DIR/$name.log"
    port=$((port + 1))
}

i=0
for arm in $ARMS; do
    i=$((i + 1))
    case $arm in
        off) repo=$PR_REPO; flag="" ;;
        main-on) repo=$MAIN_REPO; flag="--overlap-outer-communication" ;;
        pr-on) repo=$PR_REPO; flag="--overlap-outer-communication" ;;
        *) echo "unknown arm $arm" >&2; exit 2 ;;
    esac
    launch "$i-$arm" "$repo" "$flag" --out "$OUT_DIR/$i-$arm.json"
done

if [ "${NSYS:-0}" = 1 ]; then
    for arm in off pr-on; do
        flag=""; [ "$arm" = pr-on ] && flag="--overlap-outer-communication"
        echo "[run] nsys-$arm"
        PYTHONPATH="$PR_REPO:$PR_REPO/megatron/core/distributed/fsdp/src${PYTHONPATH:+:$PYTHONPATH}" \
            NSYS_OUT="$OUT_DIR/nsys-$arm" \
            python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
            --master_port=$port --no-python -- bash -c '
                if [ "$RANK" = 0 ]; then
                    exec nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
                        --capture-range-end=stop -f true -o "$NSYS_OUT.rank0" python "$@"
                fi
                exec python "$@"' _ "$HERE/bench_hfsdp_outer_ag.py" \
            --mesh-outer "$MESH_OUTER" $flag ${BENCH_ARGS:-} --tokens 4096 --ga 4 --control-ga 0 \
            --warmup 3 --steps 3 --profile 4096,4 --profile-steps 2 \
            --out "$OUT_DIR/nsys-$arm.json" 2>&1 | tee "$OUT_DIR/nsys-$arm.log"
        port=$((port + 1))
    done
fi

python "$HERE/summarize.py" "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
