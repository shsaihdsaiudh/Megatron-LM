#!/usr/bin/env bash
# Entry of E2-ABBA (#6714 gather side): separate the effect of P1 (outer all-gather
# stream) from the rest of overlap_outer_communication (reduce side, #6973) and from
# run-order drift. Three arms, interleaved so that each arm's mean has the same
# position in time:
#   off      P1 snapshot, flag off (same code path as main with the flag off)
#   main-on  main d2b0e35cc snapshot, flag on (reduce-side split only)
#   p1-on    P1 snapshot, flag on (reduce-side split + outer all-gather stream)
# Order: off, main-on, p1-on, p1-on, main-on, off.
# Results: $OUT_DIR/<i>-<arm>/ in the E1 layout (no nsys).
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
MAIN_SRC=$CONTRIB/snapshots/main-d2b0e35cc/megatron/core/distributed/fsdp/src
P1_SRC=$CONTRIB/snapshots/p1-20261112/megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$CONTRIB/snapshots/p1-20261112/p1.diff" "$BASE/p1.diff"
fi

port=${MASTER_PORT:-29500}
i=0
for arm in off main-on p1-on p1-on main-on off; do
    i=$((i + 1))
    case $arm in
        off) src=$P1_SRC; flag="" ;;
        main-on) src=$MAIN_SRC; flag="--overlap-outer-communication" ;;
        p1-on) src=$P1_SRC; flag="--overlap-outer-communication" ;;
    esac
    echo "[e2abba] run $i/6 arm=$arm src=$src flag=${flag:-none}"
    OUT_DIR=$BASE/$i-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src E1_NSYS="" \
        E1_ARGS="${E1_ARGS:-} $flag" bash "$CONTRIB/cml/e1_entry.sh"
done
echo "[e2abba] node ${RANK:-0} done"
