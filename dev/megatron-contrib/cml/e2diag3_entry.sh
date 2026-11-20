#!/usr/bin/env bash
# Entry of E2-diag3 (#6714 gather side): does P1's forward slowdown need the DP-outer
# and DP-inner all-gathers to actually run concurrently? All arms have
# overlap_outer_communication=True.
#   main-on  main d2b0e35cc snapshot
#   p1-on    P1 snapshot
#   f5       P1, but each DP-outer all-gather on the outer stream first waits for the
#            all-gather stream (and the all-gather stream waits for it after), so O
#            runs on its own stream yet never overlaps I
# Order: main-on, p1-on, f5, f5, p1-on, main-on. The first f5 run also records nsys
# (T=1024, GA=4) after its sweep. Results: $OUT_DIR/<i>-<arm>/.
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    diff -ru "$SNAP/p1-20261112/$SRC_TAIL" "$SNAP/f5-p1-serialized-20261113/$SRC_TAIL" \
        > "$BASE/f5-vs-p1.diff" || true
fi

port=${MASTER_PORT:-29500}
i=0
nsys_done=0
for arm in main-on p1-on f5 f5 p1-on main-on; do
    i=$((i + 1))
    nsys=""
    case $arm in
        main-on) src=$SNAP/main-d2b0e35cc ;;
        p1-on) src=$SNAP/p1-20261112 ;;
        f5) src=$SNAP/f5-p1-serialized-20261113
            if [ $nsys_done = 0 ]; then nsys="1024,4"; nsys_done=1; fi ;;
    esac
    echo "[e2diag3] run $i/6 arm=$arm src=$src nsys=${nsys:-none}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src/$SRC_TAIL \
        E1_NSYS="$nsys" E1_ARGS="${E1_ARGS:-} --overlap-outer-communication" \
        bash "$CONTRIB/cml/e1_entry.sh"
done
echo "[e2diag3] node ${RANK:-0} done"
