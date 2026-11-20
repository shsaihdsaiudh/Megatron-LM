#!/usr/bin/env bash
# Entry of E2-skew (#6714 gather side): are the inner all-gathers (AG-I) in mb1..3
# forward of P1 long because the 8 ranks of an inner group are out of phase?
# nsys on every rank (16 reports per arm), T=4096, GA=4, after a short T=4096 sweep.
#   v2-off    P1 snapshot, flag off
#   v2-p1     P1 snapshot, overlap_outer_communication=True
# Results: $OUT_DIR/<i>-<arm>/ (nsys in nsys/). Env: RUN_NAME, EXPECT_NODES, E1_MODELS,
# E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$SNAP/p1-20261112/p1.diff" "$CONTRIB/bench/e1_exposure.py" "$CONTRIB/cml/e1_rank.sh" "$BASE/"
fi

port=${MASTER_PORT:-29500}
i=0
for arm in v2-off v2-p1; do
    i=$((i + 1))
    case $arm in
        v2-off) flag="" ;;
        v2-p1) flag="--overlap-outer-communication" ;;
    esac
    echo "[e2skew] run $i/2 arm=$arm flag=${flag:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) \
        MFSDP_SRC=$SNAP/p1-20261112/$SRC_TAIL NSYS_LOCAL_RANKS=all \
        E1_NSYS="4096,4" E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e2skew] run $i arm=$arm failed; continuing"
done
echo "[e2skew] node ${RANK:-0} done"
