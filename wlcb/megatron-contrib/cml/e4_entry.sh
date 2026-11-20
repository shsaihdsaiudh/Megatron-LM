#!/usr/bin/env bash
# Entry of E4 (#6714 / #9539): real-model A/B with Megatron's pretrain_gpt.py.
# Llama3-8B-shaped GPT (TE layers, mock data), MFSDP v2 HFSDP: DP-outer "optim" across
# nodes (RDMA), DP-inner "optim_grads_params" within a node (NVLink); mesh (2, 8) on 2x8 H20.
#
# Arms (frozen source snapshots under $CONTRIB/snapshots):
#   off      PR snapshot (29824b2df), flag off
#   main-on  main snapshot (d2b0e35cc), --overlap-outer-communication (reduce-side split only)
#   pr-on    PR snapshot (29824b2df), --overlap-outer-communication (+ outer all-gather stream)
#
# Each launch is a separate torchrun; every node runs the same sequence.
# Env:
#   RUN_NAME, EXPECT_NODES, GPUS_PER_NODE (8)
#   E4_PLAN   "ga:iters:arm ..." timed runs, in order
#   E4_NSYS   arms for a short nsys run (GA 4, local rank 0 of every node); empty disables
#   E4_MODEL  llama8b (default) | tiny (smoke test)
#   E4_HFSDP  1 (default) | 0 (plain FSDP; for a 1-GPU smoke test)
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
source "$CONTRIB/env/mfsdp-dev.sh"
SNAP=$CONTRIB/snapshots
PR_SNAP=$SNAP/e4-pr-29824b2df
MAIN_SNAP=$SNAP/e4-main-d2b0e35cc

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-${RESOURCE_GPU:-8}}
E4_PLAN=${E4_PLAN:-"4:20:off 4:20:main-on 4:20:pr-on 4:20:pr-on 4:20:main-on 4:20:off 8:15:off 8:15:main-on 8:15:pr-on"}
E4_NSYS=${E4_NSYS-"off pr-on"}
E4_MODEL=${E4_MODEL:-llama8b}
E4_HFSDP=${E4_HFSDP:-1}
OUT_DIR=${OUT_DIR:-$CONTRIB/results/e4/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$OUT_DIR"
# torchrun reads WORLD_SIZE/RANK itself; it must see per-process values.
unset WORLD_SIZE RANK
DP=$((NNODES * GPUS_PER_NODE))

env | sort > "$OUT_DIR/env.$(hostname).txt"
if [ -n "${EXPECT_NODES:-}" ] && [ "$NNODES" != "$EXPECT_NODES" ]; then
    echo "[e4] ABORT: expected $EXPECT_NODES nodes but rendezvous variables give $NNODES." >&2
    exit 3
fi

# Megatron-FSDP requires CUDA_DEVICE_MAX_CONNECTIONS unset or > 1 (arguments.py).
unset CUDA_DEVICE_MAX_CONNECTIONS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=${NCCL_DEBUG:-VERSION}

{
    echo "host=$(hostname) node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT"
    echo "pr_snap=$PR_SNAP main_snap=$MAIN_SNAP"
    python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
    python -c "import transformer_engine; print('te', transformer_engine.__version__)"
    nvidia-smi
    nvidia-smi topo -m
} > "$OUT_DIR/node${NODE_RANK}.sysinfo.txt" 2>&1 || true
head -3 "$OUT_DIR/node${NODE_RANK}.sysinfo.txt"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi

case $E4_MODEL in
    llama8b)
        MODEL_ARGS=(--num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336
            --num-attention-heads 32 --group-query-attention --num-query-groups 8
            --kv-channels 128 --seq-length 4096 --max-position-embeddings 8192
            --rotary-base 500000 --vocab-size 128256) ;;
    tiny)
        MODEL_ARGS=(--num-layers 2 --hidden-size 512 --ffn-hidden-size 1792
            --num-attention-heads 8 --group-query-attention --num-query-groups 2
            --kv-channels 64 --seq-length 1024 --max-position-embeddings 1024
            --rotary-base 10000 --vocab-size 32000) ;;
    *) echo "[e4] unknown E4_MODEL=$E4_MODEL" >&2; exit 2 ;;
esac
MODEL_ARGS+=(--use-mcore-models --transformer-impl transformer_engine
    --position-embedding-type rope --normalization RMSNorm --norm-epsilon 1e-5 --swiglu
    --untie-embeddings-and-output-weights --disable-bias-linear
    --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02)

FSDP_ARGS=(--use-megatron-fsdp --megatron-fsdp-version 2
    --data-parallel-sharding-strategy optim_grads_params
    --overlap-grad-reduce --overlap-param-gather
    --ckpt-format fsdp_dtensor --no-gradient-accumulation-fusion --grad-reduce-in-bf16)
if [ "$E4_HFSDP" = 1 ]; then
    FSDP_ARGS+=(--num-distributed-optimizer-instances "$NNODES" --outer-dp-sharding-strategy optim)
fi

common_args() {  # common_args <ga> <iters>
    local ga=$1 iters=$2
    echo --bf16 --micro-batch-size 1 --global-batch-size $((DP * ga)) --train-iters "$iters" \
        --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine --lr-warmup-iters 2 \
        --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --seed 1234 \
        --mock-data --tokenizer-type NullTokenizer --split 100,0,0 \
        --eval-iters 0 --eval-interval 1000000 --log-interval 1 --log-throughput \
        --manual-gc --manual-gc-interval 100 --distributed-timeout-minutes 30
}

launch=0
run() {  # run <name> <arm> <ga> <iters> [extra args...]
    local name=$1 arm=$2 ga=$3 iters=$4; shift 4
    local snap flag=()
    case $arm in
        off) snap=$PR_SNAP ;;
        main-on) snap=$MAIN_SNAP; flag=(--overlap-outer-communication) ;;
        pr-on) snap=$PR_SNAP; flag=(--overlap-outer-communication) ;;
        *) echo "[e4] unknown arm $arm" >&2; return 2 ;;
    esac
    echo "[e4] run $name arm=$arm ga=$ga iters=$iters snap=$snap flag=${flag[*]:-none}"
    # Only the snapshot on the path; megatron/ is a namespace package, so mixing trees
    # would silently combine modules from both.
    (cd "$snap" && PYTHONPATH="$snap:$snap/megatron/core/distributed/fsdp/src" \
        timeout 3600 python -m torch.distributed.run \
        --nnodes="$NNODES" --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port=$((MASTER_PORT + launch)) \
        --nproc_per_node="$GPUS_PER_NODE" --no-python \
        bash "$CONTRIB/cml/e1_rank.sh" -W ignore "$snap/pretrain_gpt.py" \
        "${MODEL_ARGS[@]}" "${FSDP_ARGS[@]}" $(common_args "$ga" "$iters") "${flag[@]}" "$@") \
        > "$OUT_DIR/node${NODE_RANK}.$name.log" 2>&1 \
        || echo "[e4] run $name exited with $?"
    tail -2 "$OUT_DIR/node${NODE_RANK}.$name.log"
    launch=$((launch + 1))
}

i=0
for item in $E4_PLAN; do
    i=$((i + 1))
    IFS=: read -r ga iters arm <<< "$item"
    run "$(printf %02d $i)-${arm}-ga${ga}" "$arm" "$ga" "$iters"
done

if [ -n "$E4_NSYS" ]; then
    mkdir -p "$OUT_DIR/nsys"
    for arm in $E4_NSYS; do
        i=$((i + 1))
        name="$(printf %02d $i)-nsys-${arm}-ga4"
        # e1_rank.sh wraps local rank 0 of every node in nsys when NSYS_OUT is set.
        export NSYS_OUT="$OUT_DIR/nsys/${arm}-ga4"
        run "$name" "$arm" 4 13 \
            --profile --profile-step-start 10 --profile-step-end 12 \
            --profile-ranks 0 "$GPUS_PER_NODE" --nvtx-ranges
        unset NSYS_OUT
    done
fi
echo "[e4] node $NODE_RANK done"
