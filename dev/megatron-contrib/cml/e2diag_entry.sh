#!/usr/bin/env bash
# Entry of E2-diag (#6714 gather side): why is fwd slower with P1 at large T, even
# in the nostep control? All arms have overlap_outer_communication=True.
#   main-on      main d2b0e35cc snapshot
#   main-dummy   main + one extra never-used stream where P1 creates outer_allgather_stream
#   p1-on        P1 snapshot
#   main-on-c32  main-on with CUDA_DEVICE_MAX_CONNECTIONS=32
#   p1-on-c32    p1-on with CUDA_DEVICE_MAX_CONNECTIONS=32
# Order: the five arms, then the same five reversed (10 runs, each arm's two repeats
# are symmetric in time). Results: $OUT_DIR/<i>-<arm>/ in the E1 layout (no nsys).
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    cp "$SNAP/p1-20261112/p1.diff" "$BASE/p1.diff"
    diff -ru "$SNAP/main-d2b0e35cc/$SRC_TAIL" "$SNAP/main-dummystream-d2b0e35cc/$SRC_TAIL" \
        > "$BASE/main-dummy.diff" || true
fi

arms="main-on main-dummy p1-on main-on-c32 p1-on-c32"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    cdmc=""
    case $arm in
        main-on) src=$SNAP/main-d2b0e35cc ;;
        main-dummy) src=$SNAP/main-dummystream-d2b0e35cc ;;
        p1-on) src=$SNAP/p1-20261112 ;;
        main-on-c32) src=$SNAP/main-d2b0e35cc; cdmc=32 ;;
        p1-on-c32) src=$SNAP/p1-20261112; cdmc=32 ;;
    esac
    echo "[e2diag] run $i/10 arm=$arm src=$src cdmc=${cdmc:-unset}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) MFSDP_SRC=$src/$SRC_TAIL \
        E1_NSYS="" E1_CDMC=$cdmc E1_ARGS="${E1_ARGS:-} --overlap-outer-communication" \
        bash "$CONTRIB/cml/e1_entry.sh"
done
echo "[e2diag] node ${RANK:-0} done"
