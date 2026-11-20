#!/usr/bin/env bash
# Entry of E1v1-nsys (#6714 gather side): why is v2-off's later-microbatch forward
# (fwd_rest) ~10% faster than both v2-p1 and every v1 arm at T>=1024? One run per arm,
# each a short T=4096 sweep (timing check) plus an nsys capture at T=4096, GA=4.
#   v2-off    v2, P1 snapshot, flag off (outer AG on the all-gather stream)
#   v2-p1     v2, P1 snapshot, overlap_outer_communication=True
#   v1-same   v1 patched: outer AG on the inner param-gather stream
# Results: $OUT_DIR/<i>-<arm>/ (nsys in nsys/). Env: RUN_NAME, EXPECT_NODES, E1_MODELS,
# E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e1v1/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$SNAP/p1-20261112/p1.diff" "$BASE/p1.diff"
    diff -ru "$SNAP/main-d2b0e35cc/$SRC_TAIL" "$SNAP/v1-outer-on-ag-d2b0e35cc/$SRC_TAIL" \
        > "$BASE/v1-same.diff" || true
    cp "$CONTRIB/bench/e1v1_exposure.py" "$CONTRIB/bench/e1_exposure.py" "$BASE/"
fi

port=${MASTER_PORT:-29500}
i=0
for arm in v2-off v2-p1 v1-same; do
    i=$((i + 1))
    case $arm in
        v2-off) src=$SNAP/p1-20261112; bench=e1_exposure.py; flag="" ;;
        v2-p1) src=$SNAP/p1-20261112; bench=e1_exposure.py; flag="--overlap-outer-communication" ;;
        v1-same) src=$SNAP/v1-outer-on-ag-d2b0e35cc; bench=e1v1_exposure.py; flag="" ;;
    esac
    echo "[e1v1nsys] run $i/3 arm=$arm bench=$bench src=$src flag=${flag:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src/$SRC_TAIL \
        E1_BENCH=$bench E1_NSYS="4096,4" E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e1v1nsys] run $i arm=$arm failed; continuing"
done
echo "[e1v1nsys] node ${RANK:-0} done"
