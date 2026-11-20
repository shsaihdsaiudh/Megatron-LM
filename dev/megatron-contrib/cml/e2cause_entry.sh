#!/usr/bin/env bash
# Entry of E2-cause (#6714 gather side): what makes the compute stream occasionally
# ~9 us slower to dispatch after a cross-stream wait, so the next inner all-gather wins
# the SMs (RACE.md / RACE2.md)? All arms use the P1 snapshot; "off" = flag off.
#   off        flag off (fast reference)
#   p1         overlap_outer_communication=True (slow reference)
#   off-kern   off + before each mb0 forward, a tiny elementwise kernel on a side stream
#   off-nccl   off + before each mb0 forward, a tiny DP-outer all-reduce on a side stream
#   off-c1     off, CUDA_DEVICE_MAX_CONNECTIONS=1
#   p1-c1      p1,  CUDA_DEVICE_MAX_CONNECTIONS=1
#   p1-first   p1 for the very first step only, then outer AG back on the all-gather stream
#   p1-prio    p1 with the all-gather stream created at high priority (-1)  [fix candidate]
#   off-prio   off with the all-gather stream created at high priority (-1)
# Order: the nine arms, then reversed (18 runs). Results: $OUT_DIR/<i>-<arm>/.
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

P1="--overlap-outer-communication"
arms="off p1 off-kern off-nccl off-c1 p1-c1 p1-first p1-prio off-prio"
order="$arms $(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ')"
port=${MASTER_PORT:-29500}
i=0
for arm in $order; do
    i=$((i + 1))
    cdmc=""
    case $arm in
        off) flag="" ;;
        p1) flag="$P1" ;;
        off-kern) flag="--side-work kernel" ;;
        off-nccl) flag="--side-work nccl" ;;
        off-c1) flag=""; cdmc=1 ;;
        p1-c1) flag="$P1"; cdmc=1 ;;
        p1-first) flag="$P1 --outer-ag-first-step-only" ;;
        p1-prio) flag="$P1 --ag-stream-priority -1" ;;
        off-prio) flag="--ag-stream-priority -1" ;;
    esac
    echo "[e2cause] run $i/18 arm=$arm flag=${flag:-none} cdmc=${cdmc:-unset}"
    OUT_DIR=$BASE/$(printf %02d $i)-$arm MASTER_PORT=$((port + 50 * i)) \
        MFSDP_SRC=$SNAP/p1-20261112/$SRC_TAIL E1_CDMC="$cdmc" E1_NSYS="" \
        E1_ARGS="${E1_ARGS:-} $flag" \
        bash "$CONTRIB/cml/e1_entry.sh" \
        || echo "[e2cause] run $i arm=$arm failed; continuing"
done
echo "[e2cause] node ${RANK:-0} done"
