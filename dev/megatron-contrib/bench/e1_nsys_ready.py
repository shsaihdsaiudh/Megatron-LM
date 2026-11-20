#!/usr/bin/env python3
"""Why do NCCL all-gather (AG) kernels start late? Readiness analysis of an nsys sqlite export.

Input: an Nsight Systems report exported with `nsys export --type sqlite`.
Usage: python3 e1_nsys_ready.py FILE.sqlite [--range mb0_fwd] [--rows N]

Stream roles are detected per file. The AG stream is the stream with the most
kernels whose name contains "AllGather". The compute stream is the stream with
the most kernels whose name contains "nvjet" (GEMMs). fc1 GEMMs contain "coopB"
in their name, fc2 GEMMs do not.

Method, for the first occurrence of the NVTX range (default mb0_fwd):

1. AG kernels in the range are the AG-stream kernels whose launching runtime
   call (CUPTI_ACTIVITY_KIND_RUNTIME, joined on correlationId) starts inside the
   range. Times are reported relative to the first GPU op launched in the range.

2. For each AG kernel K, let P be the previous GPU op on the AG stream in GPU
   start order (kernel, memset, or memcpy; P may lie before the range).
   prev_end = P.end. The stream waits W on the AG stream
   (CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, syncType 2 = stream wait event) with
   P.correlationId < W.correlationId < K.correlationId gate K.

3. Event completion. Each wait is matched to its record (CUDA_EVENT row) by
   eventSyncId. CUDA_EVENT.timestamp is 0 in these exports, so completion is
   computed from the trace. Streams execute in order, so the completion time of
   a record is the latest completion among everything enqueued earlier on the
   recording stream: GPU ops (in correlationId order) and stream waits on that
   stream, whose value is the completion of the event they wait for. One pass over
   all ops, records, and waits in correlationId order computes this. Some AG
   waits target events recorded on streams 151 and 153, which have no kernels
   but do wait on other streams, so the transitive resolution is required.
   The origin of each completion (the GPU op that finishes last) is kept for
   binding reports. A correlationId-heuristic match (largest record
   correlationId below the wait, same eventId) is cross-checked against
   eventSyncId and its agreement count is printed.

4. ready = max(prev_end, all wait completions). binding is "prev_op" if
   prev_end is the max, otherwise "event". delay = K.start - ready.

5. At ready and at K.start, report the compute-stream op running (fc1/fc2 for
   GEMMs, otherwise the kernel name) and its progress (t - op.start) /
   (op.end - op.start), or "idle". Also report the time inside [ready, K.start]
   during which the compute stream and the other non-AG streams had a kernel
   running (union of intervals). Delay time with no kernel running is "idle".

6. Classification: delay < 0.05 ms is "ready" (the AG starts as soon as its
   dependencies finish; supports H2). Otherwise it is "delayed" (the AG is ready
   but waits; supports H1 when GEMMs are running during [ready, start] and the
   start falls in the last 40 percent of a GEMM). Also reported:
   - cpu_late: the AG's launch call began after ready (CPU launch was the limit).
   - binding "prev_op": the previous AG on the same stream set ready.

7. fc1 gaps: for each fc1 GEMM in the range, gap_op = fc1.start minus the end of
   the immediately previous compute-stream op (the spec's definition), and
   gap_gemm = fc1.start minus the end of the previous GEMM on the compute stream.
   gap_gemm captures stalls that sit before FSDP's memset/elementwise ops. The
   compute-stream wait with the latest completion inside (previous GEMM, fc1)
   is the gate. If the gate's origin is an AG kernel, that AG's classification
   is printed.
"""

import argparse
import bisect
import sqlite3
import statistics
from collections import Counter, defaultdict

AG_MARK = "AllGather"
GEMM_MARK = "nvjet"
FC1_MARK = "coopB"
DELAY_EPS_MS = 0.05
LAST_FRAC = 0.6  # GEMM progress >= 0.6 is the last 40 percent


class Op:
    __slots__ = ("stream", "start", "end", "corr", "name", "launch")

    def __init__(self, stream, start, end, corr, name, launch):
        self.stream, self.start, self.end = stream, start, end
        self.corr, self.name, self.launch = corr, name, launch


class Trace:
    pass


def label(name):
    if GEMM_MARK in name:
        return "fc1-GEMM" if FC1_MARK in name else "fc2-GEMM"
    return name[:18]


def origin_str(op):
    if op is None:
        return "start"
    return f"{label(op.name)} s{op.stream}"


def load(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    launch = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))

    ops = []
    for sid, s, e, corr, nm in c.execute(
        "select streamId, start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where end is not null"
    ):
        ops.append(Op(sid, s, e, corr, names.get(nm, str(nm)), launch.get(corr)))
    for table, lab in (("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):
        for sid, s, e, corr in c.execute(f"select streamId, start, end, correlationId from {table} where end is not null"):
            ops.append(Op(sid, s, e, corr, lab, launch.get(corr)))

    T = Trace()
    T.by_stream = defaultdict(list)
    for op in ops:
        T.by_stream[op.stream].append(op)
    for lst in T.by_stream.values():
        lst.sort(key=lambda o: o.start)

    gemm_n, ag_n = Counter(), Counter()
    for op in ops:
        if GEMM_MARK in op.name:
            gemm_n[op.stream] += 1
        if AG_MARK in op.name:
            ag_n[op.stream] += 1
    T.compute = gemm_n.most_common(1)[0][0]
    T.ag = ag_n.most_common(1)[0][0]
    T.compute_starts = [o.start for o in T.by_stream[T.compute]]

    # stream waits (syncType 2 = stream wait event), keyed by waiting stream
    T.waits = defaultdict(list)
    wait_rows = list(
        c.execute(
            "select correlationId, streamId, eventSyncId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
            "where syncType = 2 order by correlationId"
        )
    )
    for corr, sid, esid, eid in wait_rows:
        T.waits[sid].append((corr, esid, eid))
    T.wcorrs = {sid: [w[0] for w in lst] for sid, lst in T.waits.items()}

    # event records, matched by eventSyncId; per-eventId lists for the heuristic cross-check
    rec_rows = list(
        c.execute(
            "select timestamp, streamId, correlationId, eventId, eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT "
            "order by correlationId"
        )
    )
    T.ev_by_sync = {}
    T.ev_by_id = defaultdict(list)
    T.ev_nonzero_ts = 0
    T.ev_total = len(rec_rows)
    for ts, sid, corr, eid, esid in rec_rows:
        T.ev_by_sync[esid] = (corr, eid, sid)
        T.ev_by_id[eid].append((corr, sid))
        T.ev_nonzero_ts += ts != 0
    T.ev_id_corrs = {eid: [x[0] for x in lst] for eid, lst in T.ev_by_id.items()}

    # Single pass in correlationId (API enqueue) order. Each stream carries the
    # time at which everything enqueued so far on it is complete.
    items = [(o.corr, 0, o) for o in ops]
    items += [(corr, 1, (sid, esid)) for _, sid, corr, _, esid in rec_rows]
    items += [(corr, 2, (sid, esid)) for corr, sid, esid, _ in wait_rows]
    items.sort(key=lambda x: (x[0], x[1]))
    done = {}  # stream -> (time, origin Op or None)
    rec_val = {}  # eventSyncId -> (time, origin)
    T.wait_val = {}  # wait correlationId -> (time, origin), or None if its record is missing
    for corr, kind, payload in items:
        if kind == 0:
            op = payload
            cur = done.get(op.stream, (0, None))
            if op.end > cur[0]:
                done[op.stream] = (op.end, op)
        elif kind == 1:
            sid, esid = payload
            rec_val[esid] = done.get(sid, (0, None))
        else:
            sid, esid = payload
            v = rec_val.get(esid)
            T.wait_val[corr] = v
            if v is not None and v[0] > done.get(sid, (0, None))[0]:
                done[sid] = v

    T.agree = Counter()
    T.names = names
    return T


def op_at(T, t):
    lst = T.by_stream[T.compute]
    i = bisect.bisect_right(T.compute_starts, t) - 1
    if i >= 0 and lst[i].end > t:
        return lst[i]
    return None


def wait_value(T, wcorr, esid, eid):
    """(completion time, origin Op or None) for one stream wait; None if unresolved."""
    rec = T.ev_by_sync.get(esid)
    if rec is None:
        return None
    rcorr, reid, rsid = rec
    idx = bisect.bisect_left(T.ev_id_corrs.get(eid, []), wcorr)
    heur = T.ev_by_id[eid][idx - 1] if idx > 0 else None
    T.agree["match"] += heur == (rcorr, rsid)
    T.agree["total"] += 1
    return T.wait_val.get(wcorr)


def waits_in_window(T, sid, lo_corr, hi_corr):
    corrs = T.wcorrs.get(sid, [])
    a = bisect.bisect_right(corrs, lo_corr)
    b = bisect.bisect_left(corrs, hi_corr)
    return T.waits.get(sid, [])[a:b]


def progress_str(T, t):
    op = op_at(T, t)
    if op is None:
        return "idle"
    p = (t - op.start) / (op.end - op.start)
    return f"{label(op.name)}@{p:.2f}"


def union_len(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def busy_ms(T, a, b):
    """(compute-busy ms, any-non-AG-busy ms, GEMM-busy ms) inside [a, b]."""
    if b <= a:
        return 0.0, 0.0, 0.0
    comp, other, gemm = [], [], []
    for sid, lst in T.by_stream.items():
        if sid == T.ag:
            continue
        for o in lst:
            if o.end <= a:
                continue
            if o.start >= b:
                break
            iv = (max(o.start, a), min(o.end, b))
            if sid == T.compute:
                comp.append(iv)
                if GEMM_MARK in o.name:
                    gemm.append(iv)
            else:
                other.append(iv)
    return union_len(comp) / 1e6, union_len(comp + other) / 1e6, union_len(gemm) / 1e6


def analyze_ag(T, lo, hi):
    lst = T.by_stream[T.ag]
    rows = []
    for idx, K in enumerate(lst):
        if AG_MARK not in K.name or K.launch is None or not (lo <= K.launch <= hi):
            continue
        prev = lst[idx - 1] if idx > 0 else None
        prev_end = prev.end if prev else None
        lo_corr = prev.corr if prev else -1
        ready, binding, origin, nunk = prev_end, "prev_op", (prev.name if prev else None), 0
        best_ev = None
        for wcorr, esid, eid in waits_in_window(T, T.ag, lo_corr, K.corr):
            v = wait_value(T, wcorr, esid, eid)
            if v is None:
                nunk += 1
                continue
            if best_ev is None or v[0] > best_ev[0]:
                best_ev = v
        if best_ev is not None and best_ev[0] > 0 and (prev_end is None or best_ev[0] > prev_end):
            ready, binding, origin = best_ev[0], "event", best_ev[1]
        row = {
            "corr": K.corr,
            "dur": (K.end - K.start) / 1e6,
            "start": K.start,
            "t_rel": (K.start - T.t0) / 1e6,
            "ready_rel": None if ready is None else (ready - T.t0) / 1e6,
            "delay": None if ready is None else (K.start - ready) / 1e6,
            "binding": binding,
            "bind_src": "prev AG" if binding == "prev_op" else origin_str(origin),
            "bind_stream": None if binding == "prev_op" else (origin.stream if origin else None),
            "unknown_waits": nunk,
            "cmp_ready": "n/a" if ready is None else progress_str(T, ready),
            "cmp_start": progress_str(T, K.start),
            "cpu_late": ready is not None and K.launch > ready,
            "start_last40": False,
            "start_prog": None,
            "busy_c": 0.0,
            "busy_any": 0.0,
            "busy_gemm": 0.0,
        }
        if ready is not None:
            row["busy_c"], row["busy_any"], row["busy_gemm"] = busy_ms(T, ready, K.start)
        s_op = op_at(T, K.start)
        if s_op is not None and GEMM_MARK in s_op.name:
            row["start_prog"] = (K.start - s_op.start) / (s_op.end - s_op.start)
            row["start_last40"] = row["start_prog"] >= LAST_FRAC
        row["cls"] = "no_ready" if row["delay"] is None else ("ready" if row["delay"] < DELAY_EPS_MS else "delayed")
        row["idx"] = len(rows)
        rows.append(row)
    return rows


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else float("nan")


def summarize(rows, tag):
    valid = [r for r in rows if r["delay"] is not None]
    n = len(rows)
    print(f"[{tag}] AG kernels in range: {n}   without ready time: {n - len(valid)}")
    if not valid:
        return
    delays = [r["delay"] for r in valid]
    ready_rows = [r for r in valid if r["cls"] == "ready"]
    delayed = [r for r in valid if r["cls"] == "delayed"]
    nev = sum(r["binding"] == "event" for r in valid)
    print(
        f"    delay ms: median {statistics.median(delays):.3f}  p90 {pct(delays, 0.9):.3f}  "
        f"max {max(delays):.3f}  sum {sum(delays):.2f}"
    )
    print(
        f"    ready (delay<{DELAY_EPS_MS}): {len(ready_rows)}/{len(valid)} = {len(ready_rows) / len(valid):.3f}   "
        f"binding=event: {nev}/{len(valid)} = {nev / len(valid):.3f}"
    )
    print(
        f"    ready rows by binding: event {sum(r['binding'] == 'event' for r in ready_rows)}, "
        f"prev_op {sum(r['binding'] == 'prev_op' for r in ready_rows)}"
    )
    print(f"    binding event origin stream: {dict(Counter(r['bind_stream'] for r in valid if r['binding'] == 'event'))}")
    if delayed:
        d_sum = sum(r["delay"] for r in delayed)
        c_busy = sum(r["busy_c"] for r in delayed)
        a_busy = sum(r["busy_any"] for r in delayed)
        g_busy = sum(r["busy_gemm"] for r in delayed)
        progs = sorted(r["start_prog"] for r in delayed if r["start_prog"] is not None)
        print(
            f"    delayed: {len(delayed)}  delay sum {d_sum:.2f} ms  "
            f"starts in last 40% of a GEMM: {sum(r['start_last40'] for r in delayed)}/{len(delayed)}  "
            f"binding event/prev_op: {sum(r['binding'] == 'event' for r in delayed)}/"
            f"{sum(r['binding'] == 'prev_op' for r in delayed)}"
        )
        if progs:
            print(
                f"    GEMM progress at AG start: min {progs[0]:.2f} median {progs[len(progs) // 2]:.2f} "
                f"max {progs[-1]:.2f} (n={len(progs)})"
            )
        print(
            f"    delayed window ms: GEMM busy {g_busy:.2f}, compute-stream busy {c_busy:.2f}, "
            f"any non-AG stream busy {a_busy:.2f}, no kernel running {d_sum - a_busy:.2f}"
        )
    else:
        print("    delayed: 0")
    print(f"    cpu_late (launch call began after ready): {sum(r['cpu_late'] for r in valid)}")
    print(f"    waits with unresolved record: {sum(r['unknown_waits'] for r in rows)}")


def print_rows(rows, n):
    print(
        f"{'idx':>4} {'t_start':>9} {'dur':>7} {'ready':>9} {'delay':>8} {'bind':>8} {'class':>8} "
        f"{'compute@ready':>22} {'compute@start':>22} {'bind_src':>18}"
    )
    for r in rows[:n]:
        ready = "n/a" if r["ready_rel"] is None else f"{r['ready_rel']:9.3f}"
        delay = "n/a" if r["delay"] is None else f"{r['delay']:8.3f}"
        print(
            f"{r['idx']:>4} {r['t_rel']:9.3f} {r['dur']:7.3f} {ready:>9} {delay:>8} {r['binding']:>8} {r['cls']:>8} "
            f"{r['cmp_ready']:>22} {r['cmp_start']:>22} {r['bind_src']:>18}"
        )


def fc1_gaps(T, lo, hi, ag_rows_by_corr, rows_n):
    lst = T.by_stream[T.compute]
    out = []
    for idx, F in enumerate(lst):
        if GEMM_MARK not in F.name or FC1_MARK not in F.name or F.launch is None or not (lo <= F.launch <= hi):
            continue
        prev = lst[idx - 1] if idx > 0 else None
        gap_op = (F.start - prev.end) / 1e6 if prev else float("nan")
        pg = idx - 1
        while pg >= 0 and GEMM_MARK not in lst[pg].name:
            pg -= 1
        prev_gemm = lst[pg] if pg >= 0 else None
        gap_gemm = (F.start - prev_gemm.end) / 1e6 if prev_gemm else float("nan")
        gate, gate_val = "none", None
        if prev_gemm is not None:
            best = None
            for wcorr, esid, eid in waits_in_window(T, T.compute, prev_gemm.corr, F.corr):
                v = wait_value(T, wcorr, esid, eid)
                if v is not None and v[1] is not None and (best is None or v[0] > best[0]):
                    best = v
            if best is not None:
                gate_val = (best[0] - T.t0) / 1e6
                origin = best[1]
                if origin.stream == T.ag and origin.corr in ag_rows_by_corr:
                    gate = ag_rows_by_corr[origin.corr]
                elif origin.stream == T.ag:
                    gate = "AG outside range"
                else:
                    gate = f"{label(origin.name)} s{origin.stream}"
        comp_busy_gap = busy_ms(T, prev_gemm.end, F.start)[0] if prev_gemm is not None else 0.0
        out.append((F, gap_op, gap_gemm, gate, gate_val, comp_busy_gap))
    tot_op = sum(x[1] for x in out if x[1] == x[1])
    tot_gemm = sum(x[2] for x in out if x[2] == x[2])
    tot_busy = sum(x[5] for x in out)
    big = [x for x in out if x[2] == x[2] and x[2] >= DELAY_EPS_MS]
    print(
        f"[fc1 gaps] fc1 GEMMs in range: {len(out)}  sum gap_op {tot_op:.3f} ms  "
        f"sum gap_gemm {tot_gemm:.3f} ms  compute busy inside gap_gemm {tot_busy:.3f} ms  "
        f"gap_gemm >= {DELAY_EPS_MS} ms: {len(big)}"
    )
    gated = Counter(x[3]["cls"] if isinstance(x[3], dict) else x[3] for x in big)
    print(f"    gap_gemm >= {DELAY_EPS_MS} ms, gate: {dict(gated)}")
    print(f"{'fc1#':>5} {'t_start':>9} {'gap_op':>8} {'gap_gemm':>9} {'gate':>24} {'AG_class':>9} {'AG_delay':>9}")
    for i, (F, go, gg, gate, gv, _) in enumerate(out[:rows_n]):
        t = (F.start - T.t0) / 1e6
        gos = "     n/a" if go != go else f"{go:8.3f}"
        ggs = "     n/a" if gg != gg else f"{gg:9.3f}"
        if isinstance(gate, dict):
            print(f"{i:>5} {t:9.3f} {gos} {ggs} {'AG#' + str(gate['idx']):>24} {gate['cls']:>9} {gate['delay'] or 0:9.3f}")
        else:
            print(f"{i:>5} {t:9.3f} {gos} {ggs} {gate:>24} {'-':>9} {'-':>9}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="mb0_fwd")
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()

    T = load(args.sqlite)
    c = sqlite3.connect(args.sqlite)
    spans = []
    for s, e, text, tid in c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null order by start"):
        lab = text if text is not None else T.names.get(tid)
        if lab == args.range:
            spans.append((s, e))
    if not spans:
        raise SystemExit(f"range {args.range} not found")
    lo, hi = spans[0]
    print(f"file: {args.sqlite}")
    print(f"streams: compute={T.compute} ag={T.ag}   range {args.range} (first occurrence) = [{lo}, {hi}] ns")
    print(
        f"CUDA_EVENT rows: {T.ev_total}, nonzero timestamp: {T.ev_nonzero_ts} "
        f"(event completion computed from the trace, not timestamp)"
    )

    in_range = [o for lst in T.by_stream.values() for o in lst if o.launch is not None and lo <= o.launch <= hi]
    if not in_range:
        raise SystemExit("no GPU ops launched in range")
    T.t0 = min(o.start for o in in_range)

    rows = analyze_ag(T, lo, hi)
    print(
        f"event-link check (eventSyncId vs largest-correlation heuristic): "
        f"{T.agree['match']}/{T.agree['total']} wait links agree"
    )
    print()
    print_rows(rows, args.rows)
    print()
    summarize(rows, args.range)
    print()
    fc1_gaps(T, lo, hi, {r["corr"]: r for r in rows}, args.rows)


if __name__ == "__main__":
    main()
