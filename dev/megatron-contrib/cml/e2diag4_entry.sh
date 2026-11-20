#!/usr/bin/env bash
# Entry of E2-diag4 (#6714 gather side): do NCCL launch-ordering knobs remove P1's
# large-T forward slowdown? All arms have overlap_outer_communication=True.
#   main-on / p1-on            no extra env (reference)
#   main-lo / p1-lo            NCCL_LAUNCH_ORDER_IMPLICIT=1
#   main-gm0 / p1-gm0          NCCL_GRAPH_MIXING_SUPPORT=0
# Order: the six arms, then reversed (12 runs). Results: $OUT_DIR/<i>-<arm>/; each
# run's env.<host>.txt records the NCCL variables actually set.
# Env: RUN_NAME, EXPECT_NODES, E1_MODELS, E1_ARGS (as e1_entry.sh).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
SNAP=$CONTRIB/snapshots
SRC_TAIL=megatron/core/distributed/fsdp/src
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"

arms="main-on p1-on main-lo p1-lo main-gm0 p1-gm0"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    case $arm in
        main-*) src=$SNAP/main-d2b0e35cc ;;
        p1-*) src=$SNAP/p1-20261112 ;;
    esac
    case $arm in
        *-lo) extra="NCCL_LAUNCH_ORDER_IMPLICIT=1" ;;
        *-gm0) extra="NCCL_GRAPH_MIXING_SUPPORT=0" ;;
        *) extra="" ;;
    esac
    echo "[e2diag4] run $i/12 arm=$arm src=$src env=${extra:-none}"
    env $extra OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) \
        MFSDP_SRC=$src/$SRC_TAIL E1_NSYS="" \
        E1_ARGS="${E1_ARGS:-} --overlap-outer-communication" \
        bash "$CONTRIB/cml/e1_entry.sh"
done
echo "[e2diag4] node ${RANK:-0} done"
