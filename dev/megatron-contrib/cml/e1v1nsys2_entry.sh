#!/usr/bin/env bash
# Entry of E1v1-nsys2 (#6714 gather side): nsys traces of the two unpatched v1 arms, to check
# whether their mb1..3 forward shows the same AG-I race as v2-p1 (e2skew RACE2.md).
# nsys on every rank (16 reports per arm), T=4096, GA=4, after a short T=4096 sweep.
#   v1-base   v1 main: outer AG on its own stream, inner waits for it (serialized)
#   v1-5985   v1 main + hfsdp_param_gather_overlap (#5985, outer prefetched 1 unit ahead)
# Results: $OUT_DIR/<i>-<arm>/ (nsys in nsys/). Env: RUN_NAME, EXPECT_NODES, E1_MODELS,
# E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e1v1/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$CONTRIB/bench/e1v1_exposure.py" "$CONTRIB/cml/e1_rank.sh" "$BASE/"
fi

port=${MASTER_PORT:-29500}
i=0
for arm in v1-base v1-5985; do
    i=$((i + 1))
    case $arm in
        v1-base) flag="" ;;
        v1-5985) flag="--hfsdp-param-gather-overlap" ;;
    esac
    echo "[e1v1nsys2] run $i/2 arm=$arm flag=${flag:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) \
        MFSDP_SRC=$SNAP/main-d2b0e35cc/$SRC_TAIL NSYS_LOCAL_RANKS=all \
        E1_BENCH=e1v1_exposure.py E1_NSYS="4096,4" E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e1v1nsys2] run $i arm=$arm failed; continuing"
done
echo "[e1v1nsys2] node ${RANK:-0} done"
