#!/usr/bin/env bash
# Entry of E2 P1 (#6714 gather side): the E1 benchmark on the dev branch with
# overlap_outer_communication off and on. "off" runs the same code path as main
# (the outer all-gather stream aliases the all-gather stream). Results:
#   $OUT_DIR/off/..., $OUT_DIR/on/...  (same layout as E1)
#   $OUT_DIR/p1.diff                    (uncommitted dev-repo changes under test)
# Env: same as e1_entry.sh (RUN_NAME, EXPECT_NODES, E1_MODELS, E1_NSYS, E1_ARGS).
set -euo pipefail

CONTRIB=/planning-users/yangyeyuan/megatron-contrib
DEV_REPO=/planning-users/yangyeyuan/Megatron-LM
BASE=${OUT_DIR:-$CONTRIB/results/e2/${RUN_NAME:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$BASE"
if [ "${RANK:-0}" = 0 ]; then
    {
        git -C "$DEV_REPO" rev-parse --abbrev-ref HEAD
        git -C "$DEV_REPO" rev-parse HEAD
        git -C "$DEV_REPO" diff
    } > "$BASE/p1.diff"
fi

port=${MASTER_PORT:-29500}
OUT_DIR=$BASE/off MASTER_PORT=$port E1_ARGS="${E1_ARGS:-}" \
    bash "$CONTRIB/cml/e1_entry.sh"
OUT_DIR=$BASE/on MASTER_PORT=$((port + 50)) E1_ARGS="${E1_ARGS:-} --overlap-outer-communication" \
    bash "$CONTRIB/cml/e1_entry.sh"
echo "[e2] node ${RANK:-0} done"
