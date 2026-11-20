#!/usr/bin/env bash
# Per-rank launcher used by e1_entry.sh (torchrun --no-python). Wraps local rank 0
# of every node (global ranks 0 and 8 on 2x8) in nsys when NSYS_OUT is set.
# NSYS_LOCAL_RANKS=all wraps every local rank instead (one report per global rank).
set -euo pipefail
if [ -n "${NSYS_OUT:-}" ] && { [ "${NSYS_LOCAL_RANKS:-0}" = all ] || [ "${LOCAL_RANK:-0}" = 0 ]; }; then
    exec nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
        -f true -o "${NSYS_OUT}.rank${RANK}" python "$@"
fi
exec python "$@"
