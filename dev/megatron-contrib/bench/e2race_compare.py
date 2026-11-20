#!/usr/bin/env python3
"""Race at the end of each inner all-gather (AG-I) between the next compute kernel and the next AG-I.

For each AG-I k in mb1..3 forward whose successor k+1 (next AG-I on the inner stream) belongs to the same
phase instance, E is the end of k on that rank. The script records:
- the compute-stream state at E: busy (an op spans E) or idle (time since the last op ended);
- C: start of the first compute-stream op starting at or after E, and dC = C - E;
- A: start of AG-I k+1, and dA = A - E;
- readiness of C and of A from their waited events (cudaStreamWaitEvent), and the op that sets it.
  An event completes when its recording stream reaches the record. On a stream with kernels, that is the
  end of the last op before the record. On a stream with no kernels (a relay: it only waits and records
  events), the record completes after the waits before it on that stream, so the chain is followed back to
  the op that sets the time. The gate of a consumer is the waited event that completes last.
  "AG-I k" means that gate is AG-I k itself.
winner = "compute" if C < A, else "AG".
For winner AG: the first GEMM on the compute stream that ends after A, its overlap with AG-I k+1, and its duration.

The loader, AG-I detection, phase membership and the raw wait/record tables come from
bench/e2skew_compare.py (which builds on bench/e1v1_nsys_compare.py). All measurements are within one rank,
so no clock correction is applied. The sqlite exports are read as they are; nothing is profiled or exported.

Output: RACE.md (sections 1-7), also printed to stdout.
Usage: python3 bench/e2race_compare.py [--out PATH]
"""

import argparse
import bisect
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v1_nsys_compare as E  # noqa: E402
import e2skew_compare as S  # noqa: E402

ROOT = S.ROOT
DEFAULT_OUT = f"{ROOT}/RACE.md"
ARMS = S.ARMS
RANKS = S.RANKS
PHASES_FWD = S.PHASES_FWD
US = 1e3                  # ns per microsecond
GEMM_SEARCH = 200         # compute ops scanned for the GEMM after A
NAN = float("nan")
NEG = float("-inf")
CASE_RACE = "idle, gated by AG-I k"
CASES = ["busy", CASE_RACE, "idle, gate other or none", "idle, gate unresolved"]

FINDINGS = """Compute stream at E: idle at 96 of 2976 AG-I ends in off (3.2%) and 101 in p1 (3.4%). Winner when idle: compute 95, AG 1 (off); compute 58, AG 43 (p1). Busy at E: compute wins 2880 of 2880 (off) and 2875 of 2875 (p1).

True races (96 per arm, 6 per rank): C and AG-I k+1 are both ready at E in all 96 cases in both arms. dC median 1.9 us (p10-p90 1.6-2.2) in off and 3.2 us (2.2-6.9) in p1. dA median 3810.8 us in off and 3802.3 us in p1. In the 43 p1 AG wins, dA median is 3.2 us. In the 95 off compute wins, C is a memset of 1.3 us median, the GEMM after it starts 5.4 us after E, and AG-I k+1 starts 146.5 us before that GEMM ends. The same figures in p1 compute wins (53): C start 2.7 us, GEMM start 6.4 us, GEMM end minus A 146.8 us. In the 43 p1 AG wins, AG-I k+1 runs 4567 us median, C starts 5.9 us after E, the GEMM starts 10.2 us after E and runs 5285 us (p1 compute-win median GEMM duration: 3956 us).

Busy at E: AG-I k+1 starts with the compute stream idle in 0 of 2880 cases in off and in 1455 of 2875 cases in p1. In those p1 cases, A minus readiness is 9.2 us median. In the 2880 off cases, A minus readiness is 3812.7 us median, with a GEMM running at A (median 145.9 us before its end). In p1 the 1420 busy cases with a GEMM at A show A minus readiness 3816.0 us."""


def dist(xs):
    if not xs:
        return dict(n=0, med=NAN, p10=NAN, p90=NAN, mn=NAN, mx=NAN)
    return dict(n=len(xs), med=statistics.median(xs), p10=S.pctl(xs, .1), p90=S.pctl(xs, .9),
                mn=min(xs), mx=max(xs))


def f1(x):
    return "n/a" if x != x else f"{x:.1f}"


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def isnan(x):
    return x != x


def op_kind(D, o):
    if o.kind == "MEMCPY":
        return "memcpy"
    if o.kind == "MEMSET":
        return "memset"
    if o.is_gemm:
        return "GEMM"
    if o.is_nccl:
        return "AG-I" if (o.stream, o.grid, o.short) == D["inner_key"] else f"other NCCL {E.disp(o)}"
    return f"kernel {E.disp(o)}"


def build_queues(D):
    """Per stream, the launch-ordered items: ops (kind 0), waits (kind 1), records (kind 2)."""
    Q = defaultdict(list)
    for o in D["ops"]:
        Q[o.stream].append((o.corr, 0, o))
    for s, lst in D["waits"].items():
        for corr, _ws, eid, esid in lst:
            Q[s].append((corr, 1, (eid, esid)))
    for _key, (s, corr) in D["recs"].items():
        Q[s].append((corr, 2, None))
    out = {}
    for s, lst in Q.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        out[s] = ([t[0] for t in lst], lst)
    return out


def comp_at(ctx, s, bound):
    """(completion ns, root op, unresolved) of the last item on stream s launched before correlation id `bound`.
    op: completion = op end. wait: max(previous item, waited event). record: same as the previous item.
    root op: the op that sets the completion time (None when there is no earlier work)."""
    key = (s, bound)
    memo = ctx["memo"]
    if key in memo:
        return memo[key]
    corrs, items = ctx["Q"].get(s, ([], []))
    i = bisect.bisect_left(corrs, bound) - 1
    while i >= 0 and items[i][1] == 2:
        i -= 1
    if i < 0:
        res = (NEG, None, False)
    else:
        corr, kind, payload = items[i]
        if kind == 0:
            res = (payload.end, payload, False)
        else:
            prev = comp_at(ctx, s, corr)
            ev = event_ready(ctx, payload)
            best = prev if prev[0] >= ev[0] else ev
            res = (best[0], best[1], prev[2] or ev[2])
    memo[key] = res
    return res


def event_ready(ctx, key):
    rec = ctx["D"]["recs"].get(key)
    if rec is None:
        return (NEG, None, True)
    s, rc = rec
    return comp_at(ctx, s, rc)


def consumer_ready(ctx, o, prev):
    """Waited events of op o: cudaStreamWaitEvent calls on o's stream between prev and o. Each has a completion
    time, the root op that sets it, and whether its recording stream is a relay (no kernels on it)."""
    D = ctx["D"]
    lo = prev.corr if prev is not None else -1
    out = []
    for wcorr, _ws, eid, esid in D["waits"].get(o.stream, []):
        if not lo < wcorr < o.corr:
            continue
        rec = D["recs"].get((eid, esid))
        if rec is None:
            out.append(dict(status="record not in trace", rec_stream=None, relay=None, t=NEG, root=None))
            continue
        t, root, unres = event_ready(ctx, (eid, esid))
        out.append(dict(status="unresolved chain" if unres else "ok", rec_stream=rec[0],
                        relay=rec[0] not in ctx["has_ops"], t=t, root=root))
    return out


def gate_of(deps, k_op):
    """label: 'none' (no waited event), 'AG-I k' (the gate is AG-I k), 'other', or 'unresolved' (waits exist,
    none resolves). t: completion time of the gate (ns, -inf if no earlier work). has_unres: some waited event
    could not be resolved."""
    if not deps:
        return dict(label="none", t=NEG, root=None, has_unres=False)
    ok = [d for d in deps if d["status"] == "ok"]
    has_unres = len(ok) < len(deps)
    if not ok:
        return dict(label="unresolved", t=NEG, root=None, has_unres=True)
    g = max(ok, key=lambda d: d["t"])
    root = g["root"]
    is_k = root is not None and root.corr == k_op.corr and root.stream == k_op.stream
    return dict(label="AG-I k" if is_k else "other", t=g["t"], root=root, has_unres=has_unres)


def gate_desc(D, g):
    if g["label"] in ("AG-I k", "none", "unresolved"):
        return g["label"]
    if g["root"] is None:
        return "no prior work on the wait chain"
    p = g["root"]
    return f"{op_kind(D, p)} on stream {p.stream}"


def dep_items(D, deps, E_ns):
    out = []
    for d in deps:
        if d["status"] == "record not in trace":
            out.append(dict(status=d["status"], rec_stream=None, relay=None, root="n/a", complete=None, rel_us=NAN))
            continue
        if d["status"] != "ok":
            out.append(dict(status=d["status"], rec_stream=d["rec_stream"], relay=d["relay"], root="n/a",
                            complete=None, rel_us=NAN))
            continue
        root = "no prior work" if d["root"] is None else gate_desc(D, dict(label="other", root=d["root"]))
        done = d["t"] <= E_ns
        out.append(dict(status="ok", rec_stream=d["rec_stream"], relay=d["relay"], root=root, complete=done,
                        rel_us=NAN if done else (d["t"] - E_ns) / US))
    return out


def xcheck(D, o, prev, gate, an, tag):
    """Compare the gate time with e2skew_compare.wait_info, which resolves each event to the op recorded
    directly before it (no relay chain). Used for C, whose events have no relay in this trace."""
    if prev is None:
        return
    w = S.wait_info(D, o, prev)
    theirs = w["gate_rel"]
    mine = None if gate["t"] == NEG else (gate["t"] - o.start) / E.NS
    if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > 1e-9):
        an[f"{tag}: gate time differs from e2skew wait_info"] += 1
    else:
        an[f"{tag}: gate time agrees with e2skew wait_info"] += 1


def analyze_rank(D, arm, rank):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50000))
    ctx = dict(D=D, Q=build_queues(D), memo={}, has_ops=set(o.stream for o in D["ops"]))
    inner = D["inner_key"]
    cops, cst = D["cops"], D["cstarts"]
    s_in = [o for o in D["ops"] if o.stream == inner[0]]
    ag_pos = [p for p, o in enumerate(s_in) if E.is_inner(D, o)]
    an = Counter()
    recs = []
    for j, p in enumerate(ag_pos):
        k_op = s_in[p]
        if k_op.cpu is None or k_op.cpu[0] not in PHASES_FWD:
            continue
        an["AG-I in mb1..3 fwd"] += 1
        if j + 1 >= len(ag_pos):
            an["AG-I k is last AG-I in file"] += 1
            continue
        q = ag_pos[j + 1]
        nxt = s_in[q]
        if q - p - 1 > 0:
            an["ops on inner stream between AG-I k and k+1 (any kind)"] += q - p - 1
        if nxt.cpu != k_op.cpu:
            an["AG-I k is last of its phase instance (excluded)"] += 1
            if nxt.cpu is None:
                an["AG-I k+1 outside any phase (excluded)"] += 1
            continue
        an["pairs (k, k+1) in the same phase instance"] += 1

        E_ns = k_op.end
        i = bisect.bisect_right(cst, E_ns) - 1
        if i >= 0 and cops[i].end > E_ns:
            state = "busy"
            run = cops[i]
            busy_left = (run.end - E_ns) / US
            idle_us = NAN
            busy_name = E.kname(run)
            busy_gemm = run.is_gemm
        else:
            state = "idle"
            idle_us = (E_ns - cops[i].end) / US if i >= 0 else NAN
            if i < 0:
                an["idle at E with no earlier compute op"] += 1
            busy_left, busy_name, busy_gemm = NAN, "(idle)", False

        ci = bisect.bisect_left(cst, E_ns)
        if ci >= len(cops):
            an["no compute op at or after E (excluded)"] += 1
            continue
        C = cops[ci]
        prevC = cops[ci - 1] if ci > 0 else None
        if prevC is None:
            an["C has no previous compute op"] += 1
        deps_C = consumer_ready(ctx, C, prevC)
        gC = gate_of(deps_C, k_op)
        xcheck(D, C, prevC, gC, an, "C")

        A = nxt.start
        prevA = s_in[q - 1]
        deps_A = consumer_ready(ctx, nxt, prevA)
        gA = gate_of(deps_A, k_op)
        jA = bisect.bisect_right(cst, A) - 1
        runA = cops[jA] if jA >= 0 and cops[jA].end > A else None
        A_state = "idle" if runA is None else ("GEMM" if runA.is_gemm else "other op")
        A_gemm_end_us = (runA.end - A) / US if runA is not None and runA.is_gemm else NAN
        if any(d["relay"] for d in deps_A if d["status"] == "ok"):
            an["AG-I k+1 waits on an event recorded through a relay stream"] += 1
        if gA["label"] == "AG-I k":
            an["AG-I k+1 gated by AG-I k itself"] += 1
        if gA["has_unres"]:
            an["AG-I k+1 has a waited event that could not be resolved"] += 1
        if gC["has_unres"]:
            an["C has a waited event that could not be resolved"] += 1

        dC = (C.start - E_ns) / US
        dA = (A - E_ns) / US
        winner = "compute" if C.start < A else "AG"
        if C.start == A:
            an["C.start == A (tie, counted as AG)"] += 1
        readyA = gA["t"] <= E_ns
        readyC = gC["t"] <= E_ns
        if winner == "AG" and not readyA:
            an["AG-I k+1 starts before its event readiness time (check)"] += 1

        nx = cops[ci + 1] if ci + 1 < len(cops) else None
        gem_ov, gem_dur, gem_run = NAN, NAN, None
        if winner == "AG":
            gi = max(0, bisect.bisect_right(cst, A) - 1)
            gemm = None
            for jj in range(gi, min(len(cops), gi + GEMM_SEARCH)):
                o = cops[jj]
                if o.end > A and o.is_gemm:
                    gemm = o
                    break
            if gemm is None:
                an["AG wins, no GEMM within 200 compute ops after A"] += 1
            else:
                gem_ov = max(0, min(nxt.end, gemm.end) - max(nxt.start, gemm.start)) / US
                gem_dur = gemm.dur
                gem_run = gemm.start <= A

        if state == "busy":
            case = "busy"
        elif gC["label"] == "AG-I k":
            case = CASE_RACE
        elif gC["label"] == "unresolved":
            case = "idle, gate unresolved"
        else:
            case = "idle, gate other or none"

        recs.append(dict(
            arm=arm, rank=rank, node=rank // 8, step=S.step_of(D, k_op.launch),
            phase=k_op.cpu[0], inst=k_op.cpu[1], state=state, case=case,
            idle_us=idle_us, busy_left_us=busy_left, busy_name=busy_name, busy_gemm=busy_gemm,
            C_name=E.kname(C), C_gemm=C.is_gemm, dC_us=dC,
            readyC=readyC, readyC_us=NAN if gC["t"] == NEG else (gC["t"] - E_ns) / US,
            gateC=gC["label"], gateC_desc=gate_desc(D, gC), C_deps=dep_items(D, deps_C, E_ns),
            dA_us=dA, readyA=readyA, readyA_us=NAN if gA["t"] == NEG else (gA["t"] - E_ns) / US,
            gateA=gA["label"], gateA_desc=gate_desc(D, gA), A_deps=dep_items(D, deps_A, E_ns),
            A_relay=any(d["relay"] for d in deps_A if d["status"] == "ok"),
            winner=winner, tie=(C.start == A), gem_ov_us=gem_ov, gem_dur_ms=gem_dur, gem_run=gem_run,
            C_start_us=(C.start - E_ns) / US, C_dur_us=(C.end - C.start) / US,
            nx_start_us=NAN if nx is None else (nx.start - E_ns) / US,
            nx_dur_us=NAN if nx is None else (nx.end - nx.start) / US,
            nx_end_minus_A_us=NAN if nx is None else (nx.end - A) / US,
            A_dur_us=(nxt.end - nxt.start) / US,
            A_state=A_state, A_gemm_end_us=A_gemm_end_us,
        ))
    info = dict(inner=D["inner_key"], compute=D["compute"], overlapping=D["overlapping_compute"],
                nccl_on_compute=D["nccl_on_compute"], n_rec=len(recs))
    return recs, an, info


def run_arm(arm, adir):
    recs, an_all, meta = [], Counter(), {}
    for r in RANKS:
        D = S.load_rank(adir, r)
        if D["inner_key"] is None or D["compute"] is None:
            raise SystemExit(f"{arm} rank {r}: missing inner or compute stream")
        rr, an, info = analyze_rank(D, arm, r)
        recs += rr
        an_all.update(an)
        meta[r] = info
        print(f"[race] {arm} rank {r}: inner {info['inner']}, compute stream {info['compute']}, "
              f"k {info['n_rec']}, winners compute {sum(1 for x in rr if x['winner'] == 'compute')}, "
              f"AG {sum(1 for x in rr if x['winner'] == 'AG')}", flush=True)
        del D
    return recs, an_all, meta


def arm_recs(recs, arm, case=None):
    return [x for x in recs if x["arm"] == arm and (case is None or x["case"] == case)]


def sec1(recs):
    L = ["## 1. Compute-stream state at E, and winner counts", ""]
    rows = []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm)
        for st in ["idle", "busy"]:
            ys = [x for x in xs if x["state"] == st]
            tl = [x["idle_us"] if st == "idle" else x["busy_left_us"] for x in ys]
            rows.append((arm, st, len(ys), f"{len(ys) / len(xs):.4f}",
                         sum(1 for x in ys if x["winner"] == "compute"),
                         sum(1 for x in ys if x["winner"] == "AG"),
                         f1(dist(tl)["med"]),
                         sum(1 for x in ys if st == "busy" and x["busy_gemm"])))
        rows.append((arm, "all", len(xs), "1.0000", sum(1 for x in xs if x["winner"] == "compute"),
                     sum(1 for x in xs if x["winner"] == "AG"), "", ""))
    L += E.md_table(["arm", "compute stream at E", "AG-I k ends", "share", "winner compute", "winner AG",
                     "median time left (busy) or since last op ended (idle), us", "busy op is GEMM"], rows)
    L.append("Winner `compute` means C starts before A. Ties (C start equal to A start) count as `AG`.\n")
    rows = []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm)
        for st in ["idle", "busy"]:
            for sa in ["GEMM", "other op", "idle"]:
                ys = [x for x in xs if x["state"] == st and x["A_state"] == sa]
                rows.append((arm, st, sa, len(ys), f"{len(ys) / max(1, sum(1 for x in xs if x['state'] == st)):.4f}",
                             f1(dist([x["A_gemm_end_us"] for x in ys])["med"]) if sa == "GEMM" else "n/a",
                             f1(dist([x["readyA_us"] for x in ys])["med"]),
                             f1(dist([x["dA_us"] for x in ys])["med"]),
                             f1(dist([x["dA_us"] - x["readyA_us"] for x in ys])["med"])))
    L.append("Compute stream at A (start of AG-I k+1), split by the state at E. Readiness is the completion time of "
             "AG-I k+1's last waited event, relative to E:\n")
    L += E.md_table(["arm", "compute stream at E", "compute stream at A", "AG-I k ends", "share of E state",
                     "median (GEMM end - A), us", "AG-I k+1 readiness - E median, us", "dA median, us",
                     "A - readiness median, us"], rows)
    return L


def readiness_block(recs, case):
    rows = []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm, case)
        ra = [x for x in xs if x["readyA"]]
        na = [x for x in xs if not x["readyA"]]
        rows.append((arm, len(xs), sum(1 for x in xs if x["readyC"]),
                     len(ra), sum(1 for x in ra if x["winner"] == "AG"),
                     len(na), sum(1 for x in na if x["winner"] == "AG"),
                     f1(dist([x["readyA_us"] for x in na])["med"]),
                     f1(dist([x["readyA_us"] for x in na])["p90"])))
    return E.md_table(["arm", "cases", "C ready at E", "AG-I k+1 ready at E", "AG wins when AG ready",
                       "AG-I k+1 not ready at E", "AG wins when AG not ready",
                       "AG-I k+1 readiness after E, median (us)", "p90 (us)"], rows)


def case_block(recs, case, title):
    L = [f"### {title}", ""]
    rows_d, rows_w, rows_o = [], [], []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm, case)
        for key, name in [("dC_us", "dC = C - E"), ("dA_us", "dA = A - E")]:
            d = dist([x[key] for x in xs])
            rows_d.append((arm, name, d["n"], f1(d["med"]), f1(d["p10"]), f1(d["p90"]), f1(d["mn"]), f1(d["mx"])))
        rows_w.append((arm, len(xs), sum(1 for x in xs if x["winner"] == "compute"),
                       sum(1 for x in xs if x["winner"] == "AG"), sum(1 for x in xs if x["tie"])))
        aw = [x for x in xs if x["winner"] == "AG"]
        ov = [x["gem_ov_us"] for x in aw if not isnan(x["gem_ov_us"])]
        gd = [x["gem_dur_ms"] for x in aw if not isnan(x["gem_dur_ms"])]
        od, gdd = dist(ov), dist(gd)
        rows_o.append((arm, len(aw), len(ov), sum(1 for x in aw if x["gem_run"]),
                       sum(1 for x in ov if x > 0), f1(od["med"]), f1(od["p10"]), f1(od["p90"]),
                       f3(gdd["med"]), f3(gdd["mn"]), f3(gdd["mx"])))
    L += E.md_table(["arm", "metric", "n", "median (us)", "p10 (us)", "p90 (us)", "min (us)", "max (us)"], rows_d)
    L += E.md_table(["arm", "cases", "winner compute", "winner AG", "ties (C start = A start)"], rows_w)
    L.append("Readiness: `ready` means every waited event of the op had completed by E. An op with no waited "
             "events counts as ready.\n")
    L += readiness_block(recs, case)
    L.append("")
    L.append("AG wins: overlap of AG-I k+1 with the first GEMM that ends after A, and that GEMM's duration.\n")
    L += E.md_table(["arm", "AG wins", "GEMM found", "GEMM running at A", "overlap > 0", "overlap median (us)",
                     "overlap p10 (us)", "overlap p90 (us)", "GEMM dur median (ms)", "GEMM dur min (ms)",
                     "GEMM dur max (ms)"], rows_o)
    return L


def sec2(recs):
    L = ["## 2. True race: compute idle at E and next compute kernel gated by AG-I k", ""]
    L.append("Gate: the waited event whose recording chain ends at AG-I k, and which completes last among the waited "
             "events of C. dC and dA are measured from E.\n")
    L += case_block(recs, CASE_RACE, "True-race cases")
    L.append("### Timeline of the true-race cases, by winner (medians, us; GEMM = next compute op after C)\n")
    rows = []
    for arm, _ in ARMS:
        for w in ["compute", "AG"]:
            xs = [x for x in arm_recs(recs, arm, CASE_RACE) if x["winner"] == w]
            if not xs:
                rows.append((arm, w, 0, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"))
                continue
            rows.append((arm, w, len(xs),
                         f1(dist([x["C_start_us"] for x in xs])["med"]),
                         f1(dist([x["C_dur_us"] for x in xs])["med"]),
                         f1(dist([x["nx_start_us"] for x in xs])["med"]),
                         f1(dist([x["nx_dur_us"] for x in xs])["med"]),
                         f1(dist([x["nx_end_minus_A_us"] for x in xs])["med"]),
                         f1(dist([x["A_dur_us"] for x in xs])["med"]),
                         f1(dist([x["dA_us"] for x in xs])["med"])))
    L += E.md_table(["arm", "winner", "n", "C start - E", "C dur", "GEMM start - E", "GEMM dur",
                     "GEMM end - A", "AG-I k+1 dur", "AG-I k+1 start - E"], rows)
    return L


def sec3(recs):
    L = ["## 3. Other cases, for comparison", ""]
    for case in ["busy", "idle, gate other or none", "idle, gate unresolved"]:
        if not any(x["case"] == case for x in recs):
            L.append(f"No cases in `{case}`.\n")
            continue
        L += case_block(recs, case, f"Case: {case}")
    return L


def gate_table(recs, who, by_case):
    """Gate of AG-I k+1 (who='A') or of C (who='C'): gate description, readiness at E, relay use."""
    rows = []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm)
        groups = CASES if by_case else [None]
        for case in groups:
            ys = xs if case is None else [x for x in xs if x["case"] == case]
            agg = defaultdict(list)
            for x in ys:
                agg[x["gate" + who + "_desc"]].append(x)
            for desc, lst in sorted(agg.items(), key=lambda kv: -len(kv[1])):
                ready = [x for x in lst if x["ready" + who]]
                notr = [x for x in lst if not x["ready" + who]]
                d = dist([x["ready" + who + "_us"] for x in notr])
                relay = sum(1 for x in lst if (x["A_relay"] if who == "A" else False))
                rows.append((arm, "all" if case is None else case, desc, len(lst), len(ready), len(notr),
                             f1(d["med"]), f1(d["p90"]), f1(d["mx"]), relay if who == "A" else "-"))
    return rows


def sec4(recs):
    L = ["## 4. Gating dependencies", ""]
    L.append("Gate: the waited event that completes last. Its time is the completion of the chain that sets it: "
             "an op end, or for a relay stream the waits and records before it (see the definitions at the top). "
             "`ready at E`: that completion time is at or before E. Readiness after E is in microseconds.\n")
    L.append("### 4a. AG-I k+1 (gate by case)\n")
    summ = []
    for arm, _ in ARMS:
        xs = arm_recs(recs, arm)
        summ.append((arm, len(xs), sum(1 for x in xs if x["readyA"]), sum(1 for x in xs if not x["readyA"]),
                     sum(1 for x in xs if x["A_relay"]), sum(1 for x in xs if x["gateA"] == "AG-I k")))
    L += E.md_table(["arm", "AG-I k+1", "ready at E", "not ready at E", "waits through a relay stream",
                     "gate = AG-I k"], summ)
    L += E.md_table(["arm", "case", "gate of AG-I k+1", "AG-I k+1", "ready at E", "not ready at E",
                     "readiness after E median (us)", "p90 (us)", "max (us)", "relay-stream waits"],
                    gate_table(recs, "A", True))
    L.append("### 4b. Compute kernel C (gate by case)\n")
    L += E.md_table(["arm", "case", "gate of C", "C", "ready at E", "not ready at E",
                     "readiness after E median (us)", "p90 (us)", "max (us)", "relay-stream waits"],
                    gate_table(recs, "C", True))
    return L


def sec5(recs):
    L = ["## 5. Per rank", ""]
    L.append("True races: the race case of section 2. Medians of dC and dA are over the true races of that rank. "
             "AG wins are over all AG-I k of the rank (186 pairs per rank in mb1..3 forward).\n")
    for arm, _ in ARMS:
        rows = []
        for r in RANKS:
            xs = [x for x in recs if x["arm"] == arm and x["rank"] == r]
            races = [x for x in xs if x["case"] == CASE_RACE]
            rows.append((r, f"node {r // 8}", len(xs), len(races),
                         sum(1 for x in xs if x["winner"] == "AG"),
                         f1(dist([x["dC_us"] for x in races])["med"]),
                         f1(dist([x["dA_us"] for x in races])["med"])))
        L.append(f"### {arm}\n")
        L += E.md_table(["rank", "node", "AG-I k", "true races", "AG wins", "median dC (us)", "median dA (us)"],
                        rows)
    return L


def sec6():
    return ["## 6. Findings", "", FINDINGS.strip(), ""]


def sec7(an_by_arm, meta_by_arm):
    L = ["## 7. Checks, anomalies and unclassified cases", ""]
    rows = []
    for arm, _ in ARMS:
        inners = sorted({str(meta_by_arm[arm][r]["inner"]) for r in RANKS})
        comps = sorted({str(meta_by_arm[arm][r]["compute"]) for r in RANKS})
        ov = [meta_by_arm[arm][r]["overlapping"] for r in RANKS]
        rows.append((arm, "; ".join(inners), "; ".join(comps), min(ov), max(ov)))
    L += E.md_table(["arm", "inner AG-I (stream, gridX, name)", "compute stream", "overlapping compute ops min",
                     "max"], rows)
    keys = sorted({k for arm, _ in ARMS for k in an_by_arm[arm]})
    rows = [(k,) + tuple(an_by_arm[arm].get(k, 0) for arm, _ in ARMS) for k in keys]
    L += E.md_table(["check (summed over 16 ranks)"] + [arm for arm, _ in ARMS], rows)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    recs, an_by_arm, meta_by_arm = [], {}, {}
    for arm, adir in ARMS:
        rr, an, meta = run_arm(arm, adir)
        recs += rr
        an_by_arm[arm] = an
        meta_by_arm[arm] = meta
        print(f"[arm] {arm}: {len(rr)} AG-I k with a successor in the same phase", flush=True)

    L = ["# E2 race: next compute kernel versus next inner all-gather at the end of each AG-I", ""]
    L.append("Script: `bench/e2race_compare.py`, which reuses the loader, AG-I detection, phase membership and "
             "raw wait/record tables of `bench/e2skew_compare.py`. Inputs: the nsys sqlite exports of "
             "`d4096_L32_T4096_ga4.rank{0..15}` for `01-v2-off` (fast) and `02-v2-p1` (slow), mb1..3 forward only. "
             "All measurements are within one rank, so no clock offset is used.\n")
    L.append("Definitions (per AG-I k on a rank, k in mb1..3 forward, k+1 the next AG-I on the inner stream, in the "
             "same phase instance):\n")
    L.append("- E: end of AG-I k. Compute state at E: busy if a compute-stream op spans E, else idle.")
    L.append("- C: start of the first compute-stream op starting at or after E. dC = C - E. A: start of AG-I k+1. dA = A - E.")
    L.append("- Event completion: a waited event (cudaStreamWaitEvent) completes when its recording stream reaches the "
             "record. On a stream with kernels that is the end of the last op before the record. A stream with no "
             "kernels (relay) completes the record after the waits before it on that stream, so the chain is followed.")
    L.append("- Gate: the waited event of the op that completes last. `AG-I k` means its chain is set by AG-I k.")
    L.append("- True race: compute idle at E and the gate of C is AG-I k. Winner: `compute` if C < A, else `AG`.\n")
    L += sec1(recs)
    L += sec2(recs)
    L += sec3(recs)
    L += sec4(recs)
    L += sec5(recs)
    L += sec6()
    L += sec7(an_by_arm, meta_by_arm)
    text = "\n".join(L) + "\n"
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
