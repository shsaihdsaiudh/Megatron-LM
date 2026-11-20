#!/usr/bin/env python3
"""Compare CUDA runtime/driver API calls and stream syncs between two nsys sqlite exports.

Question: in mb1..3 forward, does p1 issue any CUDA call that off does not (from any thread,
including PyTorch ProcessGroupNCCL and NCCL internal threads)? Also lists whole-trace totals per
thread and API so that background threads (watchdog, NCCL proxy) are covered.

Sections:
  A. Per thread (ordered by first call) and API name: calls whose CPU start lies in a mb1..3 fwd
     NVTX range (any thread, time only), summed over all instances. off vs p1.
  B. Stream syncs and event records launched in those windows: (kind, stream) counts.
     cudaStreamWaitEvent rows are from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, records from
     CUPTI_ACTIVITY_KIND_CUDA_EVENT. The waited event's recording stream is resolved by eventId.
  C. Whole-trace totals per thread and API (only rows that differ).

Usage: python3 bench/api_diff.py OFF.sqlite P1.sqlite
"""

import sqlite3
import sys
from collections import Counter, defaultdict


def load(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    wins = list(
        c.execute(
            "select start, end from NVTX_EVENTS where text in ('mb1_fwd','mb2_fwd','mb3_fwd') order by start"
        )
    )
    calls = list(c.execute("select start, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    first = {}
    for s, t, _, _ in sorted(calls):
        first.setdefault(t, s)
    tid_idx = {t: i for i, t in enumerate(sorted(first, key=first.get))}

    def inwin(s):
        return any(a <= s <= b for a, b in wins)

    secA = Counter()
    corr_in = set()
    total = Counter()
    for s, t, corr, nm in calls:
        key = (f"T{tid_idx[t]}", names.get(nm, str(nm)))
        total[key] += 1
        if inwin(s):
            secA[key] += 1
            corr_in.add(corr)

    # Resolve each wait to the latest record of the same eventId that precedes it in API order
    # (correlation id), because event handles are created and destroyed repeatedly.
    import bisect

    by_ev = defaultdict(list)
    recs = Counter()
    for st, corr, ev in c.execute(
        "select streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT order by correlationId"
    ):
        by_ev[ev].append((corr, st))
        if corr in corr_in:
            recs[("record", st)] += 1
    waits = Counter()
    for st, corr, ev, ty in c.execute(
        "select streamId, correlationId, eventId, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    ):
        if corr not in corr_in:
            continue
        lst = by_ev.get(ev, [])
        i = bisect.bisect_left(lst, (corr, -1)) - 1
        src = lst[i][1] if i >= 0 else "?"
        waits[(f"sync type {ty} on stream {st}", f"event from stream {src}")] += 1
    return len(wins), secA, recs, waits, total


def table(title, a, b, only_diff=False):
    print(f"\n## {title}\n")
    print("| key | off | p1 |")
    print("|---|---|---|")
    n = 0
    for k in sorted(set(a) | set(b), key=str):
        if only_diff and a.get(k, 0) == b.get(k, 0):
            continue
        mark = "" if a.get(k, 0) == b.get(k, 0) else " **diff**"
        print(f"| {k} | {a.get(k, 0)} | {b.get(k, 0)}{mark} |")
        n += 1
    if n == 0:
        print("| (no differences) | | |")


def main():
    off, p1 = load(sys.argv[1]), load(sys.argv[2])
    print(f"mb1..3 fwd NVTX instances: off {off[0]}, p1 {p1[0]}")
    table("A. API calls in mb1..3 fwd windows (thread, API)", off[1], p1[1])
    table("B1. Event records launched in mb1..3 fwd windows (kind, stream)", off[2], p1[2])
    table("B2. Stream syncs launched in mb1..3 fwd windows", off[3], p1[3])
    table("C. Whole-trace API totals that differ (thread, API)", off[4], p1[4], only_diff=True)


if __name__ == "__main__":
    main()
