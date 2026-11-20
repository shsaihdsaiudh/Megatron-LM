#!/usr/bin/env bash
# Entry of E2-resync (#6714 gather side): does P1's mb1..3 forward slowdown go away
# if all ranks are drained and barriered before the later microbatches? Diagnostic
# only (bench flag --resync); the barrier is not meant for the PR.
#   off         P1 snapshot, flag off
#   p1          P1 snapshot, overlap_outer_communication=True
#   p1-rs1      p1, resync once before mb1 forward
#   p1-rsall    p1, resync before every forward after mb0
#   off-rsall   off, resync before every forward after mb0 (cost of the resync itself)
# Order: the five arms, then reversed (10 runs). Results: $OUT_DIR/<i>-<arm>/.
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$SNAP/p1-20261112/p1.diff" "$CONTRIB/bench/e1_exposure.py" "$BASE/"
fi

arms="off p1 p1-rs1 p1-rsall off-rsall"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    case $arm in
        off) flag="" ;;
        p1) flag="--overlap-outer-communication" ;;
        p1-rs1) flag="--overlap-outer-communication --resync mb1" ;;
        p1-rsall) flag="--overlap-outer-communication --resync all" ;;
        off-rsall) flag="--resync all" ;;
    esac
    echo "[e2resync] run $i/10 arm=$arm flag=${flag:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) \
        MFSDP_SRC=$SNAP/p1-20261112/$SRC_TAIL E1_NSYS="" E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e2resync] run $i arm=$arm failed; continuing"
done
echo "[e2resync] node ${RANK:-0} done"
