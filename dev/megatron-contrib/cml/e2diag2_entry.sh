#!/usr/bin/env bash
# Entry of E2-diag2 (#6714 gather side): candidate fix and mechanism test for P1's
# large-T forward slowdown. All arms have overlap_outer_communication=True.
#   main-on  main d2b0e35cc snapshot
#   p1-on    P1 snapshot (new outer all-gather stream)
#   f3       P1 with the outer all-gather on the existing outer reduction stream
#   f4       main + an extra stream that, in each stale (first-microbatch) unshard,
#            forks from the all-gather stream, runs a 1-element add, and is joined
#            back; the DP-outer all-gather stays where main puts it
# Order: the four arms, then reversed (8 runs). Results: $OUT_DIR/<i>-<arm>/.
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$SNAP/p1-20261112/p1.diff" "$BASE/p1.diff"
    diff -ru "$SNAP/p1-20261112/$SRC_TAIL" "$SNAP/f3-reuse-outer-rs-20261112/$SRC_TAIL" \
        > "$BASE/f3-vs-p1.diff" || true
    diff -ru "$SNAP/main-d2b0e35cc/$SRC_TAIL" "$SNAP/f4-main-touchstream-d2b0e35cc/$SRC_TAIL" \
        > "$BASE/f4-vs-main.diff" || true
fi

arms="main-on p1-on f3 f4"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    case $arm in
        main-on) src=$SNAP/main-d2b0e35cc ;;
        p1-on) src=$SNAP/p1-20261112 ;;
        f3) src=$SNAP/f3-reuse-outer-rs-20261112 ;;
        f4) src=$SNAP/f4-main-touchstream-d2b0e35cc ;;
    esac
    echo "[e2diag2] run $i/8 arm=$arm src=$src"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src/$SRC_TAIL \
        E1_NSYS="" E1_ARGS="${E1_ARGS:-} --overlap-outer-communication" \
        bash "$CONTRIB/cml/e1_entry.sh"
done
echo "[e2diag2] node ${RANK:-0} done"
