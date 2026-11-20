#!/usr/bin/env bash
# Entry of E1v1 (#6714 gather side): MFSDP v1 vs v2 on the same model, mesh, data.
# Does v1 also pay the later-microbatch forward penalty when the DP-outer all-gather
# runs on a stream other than the DP-inner all-gather stream?
#   v2-off    v2, P1 snapshot, flag off (outer AG on the all-gather stream)
#   v2-p1     v2, P1 snapshot, overlap_outer_communication=True
#   v1-base   v1 main: outer AG on its own stream, inner waits for it (serialized)
#   v1-5985   v1 main + hfsdp_param_gather_overlap (#5985, outer prefetched 1 unit ahead)
#   v1-same   v1 main patched: outer AG on the inner param-gather stream
# Order: the five arms, then reversed (10 runs). Results: $OUT_DIR/<i>-<arm>/.
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
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

arms="v2-off v2-p1 v1-base v1-5985 v1-same"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    case $arm in
        v2-off) src=$SNAP/p1-20261112; bench=e1_exposure.py; flag="" ;;
        v2-p1) src=$SNAP/p1-20261112; bench=e1_exposure.py; flag="--overlap-outer-communication" ;;
        v1-base) src=$SNAP/main-d2b0e35cc; bench=e1v1_exposure.py; flag="" ;;
        v1-5985) src=$SNAP/main-d2b0e35cc; bench=e1v1_exposure.py; flag="--hfsdp-param-gather-overlap" ;;
        v1-same) src=$SNAP/v1-outer-on-ag-d2b0e35cc; bench=e1v1_exposure.py; flag="" ;;
    esac
    echo "[e1v1] run $i/10 arm=$arm bench=$bench src=$src flag=${flag:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src/$SRC_TAIL \
        E1_BENCH=$bench E1_NSYS="" E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e1v1] run $i arm=$arm failed; continuing"
done
echo "[e1v1] node ${RANK:-0} done"
