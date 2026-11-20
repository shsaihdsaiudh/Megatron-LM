#!/usr/bin/env python3
"""Second race at the end of each inner all-gather (AG-I): compute stream after the gating kernel X.

Scope: AG-I k+1 in mb1..3 forward (same phase instance as AG-I k) whose gate is a kernel X on the compute
stream. The gate is the waited event that completes last, resolved through relay streams with the same
functions as bench/e2race_compare.py (loader from bench/e2skew_compare.py, which builds on
bench/e1v1_nsys_compare.py). Per rank, with R = end of X and A = start of AG-I k+1:
- N: next op (kernel, memset or memcpy) on the compute stream after X. G: first GEMM (nvjet) after X.
  gapN = N.start - R, gapG = G.start - R, dA = A - R.
- Winner: AG if A < G.start, else GEMM.
- Waits on the compute stream between X and N (launch order by correlation id), and between N and G.
  Each wait resolves to the op that sets its record time. Completion is compared with R.
- Other GPU ops (any stream) that start in [R, min(A, G.start)], excluding X, N, G and AG-I k+1.
- CPU launch (CUPTI runtime start) of N and G relative to R. Negative means queued before R.

All times are within one rank, so no clock offset is used. Times are in microseconds.
Each record is checked against a second read of the sqlite export (raw UNION of kernel, memset and
memcpy rows) for the next op, the first GEMM and the window op count.

Output: RACE2.md (sections 1-7), also printed to stdout.
Usage: python3 bench/e2race2_compare.py [--out PATH]
"""

import argparse
import bisect
import math
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v1_nsys_compare as E  # noqa: E402
import e2skew_compare as S  # noqa: E402
import e2race_compare as RC  # noqa: E402

ROOT = S.ROOT
DEFAULT_OUT = f"{ROOT}/RACE2.md"
ARMS = S.ARMS
RANKS = S.RANKS
PHASES_FWD = S.PHASES_FWD
US = 1e3                       # ns per microsecond
NAN = float("nan")
COMPUTE_STREAM = 7             # expected compute stream (checked per rank)
SPAN_LOOK_NS = 50e6            # look-back for ops that started before R and are still running at R (longest AG/GEMM ~8 ms)
WINNERS = ["AG", "GEMM"]
WINDOWS = [("w1", "X..N"), ("w2", "N..G")]

# RACE.md section 1 and section 2 reference values for the same gate (X = elementwise kernel on stream 7).
# off: busy-at-E cases with GEMM at A and no idle A. p1: 1455 of 2875 busy-at-E cases start AG-I k+1 with
# stream 7 idle.
RACE_REF = {"off": 2880, "p1": 2875}
RACE_REF_IDLE_A = {"off": 0, "p1": 1455}

FINDINGS = """X-gated cases: 2880 per arm (AG-I k+1 gated by a compute-stream kernel X), out of 2976 pairs in mb1..3 forward; the other 96 are gated by AG-I k. In all 2880 cases of both arms, N (next compute-stream op after X) is a memset and G (first GEMM after X) is a GEMM. off: AG wins 0 of 2880; gapN median 3.0 us (p10 2.8, p90 3.5), gapG median 6.3 us (5.8, 7.2), dA median 3812.7 us. p1: AG wins 1457 of 2880 (50.6%), GEMM wins 1423. p1 AG wins: gapN median 12.1 us (10.7, 13.6), gapG median 16.5 us (15.0, 18.1), dA median 9.2 us (7.9, 10.4). p1 GEMM wins: gapN median 4.0 us (3.4, 4.7), gapG median 7.7 us (6.4, 9.1), dA median 3816.0 us. The gapG median of p1 AG wins is 8.8 us above p1 GEMM wins and 10.2 us above off. p1 AG wins per rank: 81 to 103 of 180 cases. Stream 7 has exactly one wait between X and N per case in both arms (1 of 1 in 2880 cases each), recorded by stream 39 (AG-I k) and complete before R in 2880 of 2880 cases in both arms. Between N and G there are no waits in any case. N and G were launched before R in all cases: N launch minus R median -1391505.7 us (off), -1431387.6 us (p1 AG wins), -1431347.0 us (p1 GEMM wins). No other op on any stream starts in [R, min(A, G.start)] in any case, and no op on another stream runs across R. Nothing in sections 2 and 3 appears only in p1 AG wins. In the 1457 p1 AG wins, AG-I k+1 starts before N.start in 1457 of 1457 cases, and stream 7 has no op spanning A in 1457 of 1457 cases."""

UNION_BODY = ("select start, end, correlationId, 'K' as kind, shortName as nm, streamId as sid "
              "from CUPTI_ACTIVITY_KIND_KERNEL "
              "union all select start, end, correlationId, 'M', -1, streamId from CUPTI_ACTIVITY_KIND_MEMSET "
              "union all select start, end, correlationId, 'C', -1, streamId from CUPTI_ACTIVITY_KIND_MEMCPY")


def isnan(x):
    return x != x


def dist(xs):
    xs = [x for x in xs if not isnan(x)]
    if not xs:
        return dict(n=0, med=NAN, p10=NAN, p90=NAN, mx=NAN)
    return dict(n=len(xs), med=statistics.median(xs), p10=S.pctl(xs, .1), p90=S.pctl(xs, .9), mx=max(xs))


def f1(x):
    return "n/a" if isnan(x) else f"{x:.1f}"


def f4(x):
    return "n/a" if isnan(x) else f"{x:.4f}"


def sel(recs, arm, w=None):
    return [x for x in recs if x["arm"] == arm and (w in (None, "all") or x["winner"] == w)]


def waits_between(ctx, s, lo, hi, R):
    """cudaStreamWaitEvent calls on stream s with lo < correlation id < hi. Each waited event is resolved
    through its record chain (RC.event_ready): a relay stream is followed back to the op that sets the time."""
    D = ctx["D"]
    out = []
    for wcorr, _ws, eid, esid in D["waits"].get(s, []):
        if not lo < wcorr < hi:
            continue
        rec = D["recs"].get((eid, esid))
        if rec is None:
            out.append(dict(wcorr=wcorr, status="record not in trace", rec_desc="n/a", root_desc="n/a",
                            done=None, rel=NAN))
            continue
        rs = rec[0]
        relay = rs not in ctx["has_ops"]
        t, root, unres = RC.event_ready(ctx, (eid, esid))
        rec_desc = f"stream {rs} ({'relay' if relay else 'with ops'})"
        if unres:
            out.append(dict(wcorr=wcorr, status="unresolved chain", rec_desc=rec_desc, root_desc="n/a",
                            done=None, rel=NAN))
            continue
        done = t <= R
        root_desc = "no prior work" if root is None else f"{root.kind.lower()} on stream {root.stream}"
        out.append(dict(wcorr=wcorr, status="ok", rec_desc=rec_desc, root_desc=root_desc, done=done,
                        rel=NAN if done else (t - R) / US))
    return out


def analyze_rank(D, arm, rank):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50000))
    cs = D["compute"]
    cops = D["cops"]
    ops = D["ops"]
    ost = [o.start for o in ops]
    ctx = dict(D=D, Q=RC.build_queues(D), memo={}, has_ops=set(o.stream for o in ops))
    cidx = {id(o): i for i, o in enumerate(cops)}
    inner = D["inner_key"]
    s_in = [o for o in ops if o.stream == inner[0]]
    ag_pos = [p for p, o in enumerate(s_in) if E.is_inner(D, o)]
    an = Counter()
    recs = []
    for j, p in enumerate(ag_pos):
        k_op = s_in[p]
        if k_op.cpu is None or k_op.cpu[0] not in PHASES_FWD:
            continue
        if j + 1 >= len(ag_pos):
            an["AG-I k is the last AG-I of the file (excluded)"] += 1
            continue
        q = ag_pos[j + 1]
        A_op = s_in[q]
        if A_op.cpu != k_op.cpu:
            an["k+1 in another phase instance (excluded)"] += 1
            continue
        an["pairs (k, k+1) in mb1..3 fwd, same instance"] += 1
        deps = RC.consumer_ready(ctx, A_op, s_in[q - 1])
        g = RC.gate_of(deps, k_op)
        if g["label"] != "other":
            an[f"gate of k+1: {g['label']}"] += 1
            continue
        X = g["root"]
        if X is None:
            an["gate of k+1: other, no prior work"] += 1
            continue
        an[f"gate of k+1: root {X.kind.lower()} on stream {X.stream}"] += 1
        if X.stream != cs or X.kind != "KERNEL":
            an["gate root not an X kernel on the compute stream (excluded)"] += 1
            continue
        if g["has_unres"]:
            an["X-gated with an unresolved waited event of k+1 (excluded)"] += 1
            continue
        an["X-gated cases (kernel on compute stream)"] += 1

        R = X.end
        i = cidx[id(X)]
        if i + 1 >= len(cops):
            an["X is the last compute-stream op (excluded)"] += 1
            continue
        N = cops[i + 1]
        G, gi = None, None
        for jj in range(i + 1, len(cops)):
            if cops[jj].is_gemm:
                G, gi = cops[jj], jj
                break
        if G is None:
            an["X-gated, no GEMM after X on the compute stream"] += 1
        A = A_op.start
        if N.start < R:
            an["N starts before R (stream overlap)"] += 1
        if A < R:
            an["A starts before R"] += 1
        Gs = G.start if G is not None else math.inf
        winner = "AG" if A < Gs else "GEMM"
        hi = min(A, Gs)

        w1 = waits_between(ctx, cs, X.corr, N.corr, R)
        w2 = waits_between(ctx, cs, N.corr, G.corr, R) if G is not None else []

        def rel_launch(o):
            return NAN if o.launch is None else (o.launch - R) / US

        lN = rel_launch(N)
        lG = rel_launch(G) if G is not None else NAN
        if N.launch is None:
            an["N has no CUPTI runtime row"] += 1
        if G is not None and G.launch is None:
            an["G has no CUPTI runtime row"] += 1

        excl = {id(X), id(N), id(A_op)}
        if G is not None:
            excl.add(id(G))
        a = bisect.bisect_left(ost, R)
        b = bisect.bisect_right(ost, hi)
        others = [(o.stream, E.kname(o), (o.start - R) / US) for o in ops[a:b] if id(o) not in excl]

        # Ops on other streams that started before R and are still running at R (they can hold SMs at R).
        spans = []
        j = a - 1
        while j >= 0 and R - ops[j].start < SPAN_LOOK_NS:
            o = ops[j]
            if o.end > R and o.stream != cs:
                spans.append((o.stream, E.kname(o), (o.start - R) / US, (o.end - R) / US))
            j -= 1

        E_busy = E.running_op(D, k_op.end) is not None
        recs.append(dict(
            arm=arm, rank=rank, winner=winner,
            gapN=(N.start - R) / US, gapG=(G.start - R) / US if G is not None else NAN, dA=(A - R) / US,
            N_kind=N.kind, N_name=E.kname(N), X_name=E.kname(X),
            N_launch=lN, G_launch=lG, N_after=lN > 0, G_after=(not isnan(lG)) and lG > 0,
            w1=w1, w2=w2, others=others, spans=spans, E_busy=E_busy, A_before_N=A < N.start,
            A_idle=E.running_op(D, A) is None,
            X_idx=i, G_idx=gi, X_corr=X.corr, N_corr=N.corr, G_corr=G.corr if G is not None else None,
            R=R, A=A, hi=hi,
            excl_starts=[s for s in [N.start, A, G.start if G is not None else None] if s is not None],
        ))
    return recs, an


def verify_rank(sq, D, recs):
    """Second read of the sqlite export: raw stream list, N, G and window op counts, all from SQL rows."""
    cs = D["compute"]
    c = sqlite3.connect(sq)
    names = dict(c.execute("select id, value from StringIds"))
    raw_s = c.execute(f"select start, end, correlationId, kind, nm from ({UNION_BODY}) where sid = ? "
                      "order by start", (cs,)).fetchall()
    raw_all = [r[0] for r in c.execute(f"select start from ({UNION_BODY}) order by start").fetchall()]
    c.close()
    out = Counter()
    for k in ["stream list length differs from loader", "stream list entries differing from loader",
              "X index mismatch", "N mismatch", "G mismatch", "G: raw has a GEMM, loader has none",
              "window op count differs from raw"]:
        out[k] += 0
    cops = D["cops"]
    if len(raw_s) != len(cops):
        out["stream list length differs from loader"] += 1
    out["stream list entries differing from loader"] += sum(
        1 for (s, e, corr, _k, _n), o in zip(raw_s, cops) if (s, e, corr) != (o.start, o.end, o.corr))
    for r in recs:
        out["records checked"] += 1
        xi = r["X_idx"]
        if raw_s[xi][2] != r["X_corr"]:
            out["X index mismatch"] += 1
        if xi + 1 >= len(raw_s) or raw_s[xi + 1][2] != r["N_corr"]:
            out["N mismatch"] += 1
        gj = None
        for j in range(xi + 1, len(raw_s)):
            if raw_s[j][3] == "K" and "nvjet" in names.get(raw_s[j][4], ""):
                gj = j
                break
        if r["G_corr"] is None:
            if gj is not None:
                out["G: raw has a GEMM, loader has none"] += 1
        elif gj is None or raw_s[gj][2] != r["G_corr"]:
            out["G mismatch"] += 1
        raw_cnt = bisect.bisect_right(raw_all, r["hi"]) - bisect.bisect_left(raw_all, r["R"])
        exp = len(r["others"]) + sum(1 for s in r["excl_starts"] if r["R"] <= s <= r["hi"])
        if raw_cnt != exp:
            out["window op count differs from raw"] += 1
    return out


def run_arm(arm, adir):
    recs, an_all, meta, ver_all = [], Counter(), {}, Counter()
    for r in RANKS:
        sq = S.sqlite_path(adir, r)
        D = S.load_rank(adir, r)
        if D["inner_key"] is None or D["compute"] is None:
            raise SystemExit(f"{arm} rank {r}: missing inner or compute stream")
        rr, an = analyze_rank(D, arm, r)
        ver = verify_rank(sq, D, rr)
        meta[r] = dict(compute=D["compute"], inner=D["inner_key"])
        recs += rr
        an_all.update(an)
        ver_all.update(ver)
        print(f"[race2] {arm} rank {r}: compute {D['compute']}, X-gated {len(rr)}, "
              f"AG wins {sum(1 for x in rr if x['winner'] == 'AG')}, verify {dict(ver)}", flush=True)
        del D
    return recs, an_all, meta, ver_all


def sec1(recs):
    L = ["## 1. Gaps and dA by arm and winner (us)", ""]
    L.append("R = end of X. gapN = N.start - R, gapG = G.start - R, dA = A - R. Winner AG if A < G.start, else GEMM. "
             "Quantiles are nearest-rank over the cases listed.\n")
    rows = []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            for key, label in [("gapN", "gapN"), ("gapG", "gapG"), ("dA", "dA")]:
                d = dist([x[key] for x in xs])
                rows.append((arm, w, label, d["n"], f1(d["med"]), f1(d["p10"]), f1(d["p90"]), f1(d["mx"])))
    L += E.md_table(["arm", "winner", "metric", "n", "median", "p10", "p90", "max"], rows)
    L.append("Top 3 kinds and names of N (next compute-stream op after X):\n")
    rows = []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            c = Counter(f"{x['N_kind']}: {x['N_name']}" for x in xs)
            for i, (nm, cn) in enumerate(c.most_common(3), 1):
                rows.append((arm, w, i, nm, cn, f"{cn / len(xs):.4f}" if xs else "n/a"))
            if not c:
                rows.append((arm, w, "-", "none", 0, "n/a"))
    L += E.md_table(["arm", "winner", "rank", "N kind: name", "count", "share"], rows)
    return L


def sec2(recs):
    L = ["## 2. Waits on the compute stream between X and N, and between N and G", ""]
    L.append("Window `X..N`: waits launched after X and before N (correlation id order). Window `N..G`: after N and "
             "before G. A wait is `complete` when its chain completes at or before R. For the others, the "
             "completion minus R is given. The relay stream (no kernels) is followed to the op that sets the time. "
             "`root` is the kind and stream of that op.\n")
    rows = []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            for win, wl in WINDOWS:
                items = [wt for x in xs for wt in x[win]]
                ok = [wt for wt in items if wt["status"] == "ok"]
                inc = [wt for wt in ok if not wt["done"]]
                d = dist([wt["rel"] for wt in inc])
                rows.append((arm, w, wl, len(xs), len(items), sum(1 for x in xs if x[win]),
                             sum(1 for wt in ok if wt["done"]), f4(sum(1 for wt in ok if wt["done"]) / len(ok))
                             if ok else "n/a", len(inc), f1(d["med"]), f1(d["p90"]), f1(d["mx"]),
                             sum(1 for wt in items if wt["status"] == "unresolved chain"),
                             sum(1 for wt in items if wt["status"] == "record not in trace")))
    L += E.md_table(["arm", "winner", "window", "cases", "waits", "cases with >=1 wait", "complete before R",
                     "fraction complete", "incomplete", "incomplete: completion - R median (us)", "p90", "max",
                     "unresolved", "record not in trace"], rows)
    L.append("Waits by window, recording stream and root (all cases of the arm and winner):\n")
    grp = defaultdict(list)
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            for x in sel(recs, arm, w):
                for win, wl in WINDOWS:
                    for wt in x[win]:
                        grp[(arm, w, wl, wt["rec_desc"], wt["root_desc"], wt["status"])].append(wt)
    rows = []
    for k, lst in sorted(grp.items()):
        ok = [wt for wt in lst if wt["status"] == "ok"]
        done = sum(1 for wt in ok if wt["done"])
        inc = [wt["rel"] for wt in ok if not wt["done"]]
        d = dist(inc)
        rows.append((k[0], k[1], k[2], k[3], k[4], k[5], len(lst), done,
                     f4(done / len(ok)) if ok else "n/a", f1(d["med"]), f1(d["p90"]), f1(d["mx"])))
    L += E.md_table(["arm", "winner", "window", "record stream", "root", "status", "waits", "complete before R",
                     "fraction complete", "incomplete: median (us)", "p90", "max"], rows)
    L.append("Launch of N and G relative to R (CUPTI runtime start minus R, us). Negative means queued before R.\n")
    rows = []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            Nl = [x["N_launch"] for x in xs]
            Gl = [x["G_launch"] for x in xs if x["G_corr"] is not None]
            dn, dg = dist(Nl), dist(Gl)
            rows.append((arm, w, len(xs), sum(1 for x in xs if x["N_after"]),
                         sum(1 for x in xs if x["G_after"]), f1(dn["med"]), f1(dn["p90"]), f1(dn["mx"]),
                         f1(dg["med"]), f1(dg["p90"]), f1(dg["mx"])))
    L += E.md_table(["arm", "winner", "cases", "N launched after R", "G launched after R",
                     "N launch - R median", "N p90", "N max", "G launch - R median", "G p90", "G max"], rows)
    return L


def sec3(recs):
    L = ["## 3. Other GPU ops (any stream) starting in [R, min(A, G.start)]", ""]
    L.append("Excluded from the window: X, N, G and AG-I k+1. Counts are over all ops in the window of each case. "
             "`cases with it` counts the cases where the (stream, name) pair occurs at least once.\n")
    rows, srows = [], []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            cnt, cases = Counter(), Counter()
            starts = defaultdict(list)
            for x in xs:
                seen = set()
                for st, nm, rel in x["others"]:
                    cnt[(st, nm)] += 1
                    starts[(st, nm)].append(rel)
                    seen.add((st, nm))
                for kk in seen:
                    cases[kk] += 1
            tot = sum(cnt.values())
            srows.append((arm, w, len(xs), tot, f"{tot / len(xs):.2f}" if xs else "n/a",
                          sum(1 for x in xs if x["others"]), len(cnt)))
            for kk, c in cnt.most_common():
                rows.append((arm, w, kk[0], kk[1], c, cases[kk], f1(dist(starts[kk])["med"]),
                             f1(dist(starts[kk])["mx"])))
    L += E.md_table(["arm", "winner", "cases", "other ops in window", "per case", "cases with >=1 op",
                     "distinct (stream, name)"], srows)
    if rows:
        L += E.md_table(["arm", "winner", "stream", "name", "count", "cases with it", "start - R median (us)",
                         "start - R max (us)"], rows)
    else:
        L.append("No other op starts in the window in any case.\n")
    L.append("Ops on other streams that started before R and are still running at R (look-back "
             f"{SPAN_LOOK_NS / 1e6:.0f} ms). Times relative to R, median over the occurrences.\n")
    srows, rows = [], []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            cnt, cases = Counter(), Counter()
            st_, en_ = defaultdict(list), defaultdict(list)
            for x in xs:
                seen = set()
                for st, nm, s0, e0 in x["spans"]:
                    cnt[(st, nm)] += 1
                    st_[(st, nm)].append(s0)
                    en_[(st, nm)].append(e0)
                    seen.add((st, nm))
                for kk in seen:
                    cases[kk] += 1
            srows.append((arm, w, len(xs), sum(cnt.values()), sum(1 for x in xs if x["spans"])))
            for kk, c in cnt.most_common():
                rows.append((arm, w, kk[0], kk[1], c, cases[kk], f1(dist(st_[kk])["med"]),
                             f1(dist(en_[kk])["med"])))
    L += E.md_table(["arm", "winner", "cases", "spanning ops", "cases with >=1"], srows)
    if rows:
        L += E.md_table(["arm", "winner", "stream", "name", "count", "cases with it", "start - R median (us)",
                         "end - R median (us)"], rows)
    else:
        L.append("No op on another stream is running at R.\n")
    return L


def case_keys(x):
    keys = set()
    for win, wl in WINDOWS:
        for wt in x[win]:
            keys.add(("wait", wl, wt["rec_desc"], wt["root_desc"], wt["status"],
                      "complete" if wt["done"] else "incomplete"))
    if x["N_after"]:
        keys.add(("launch", "N launched after R"))
    if x["G_after"]:
        keys.add(("launch", "G launched after R"))
    for st, nm, _rel in x["others"]:
        keys.add(("other", st, nm))
    return keys


def sec4(recs):
    L = ["## 4. AG wins in p1 against GEMM wins in p1 and all of off", ""]
    off_all = sel(recs, "off")
    p1_ag = sel(recs, "p1", "AG")
    p1_gemm = sel(recs, "p1", "GEMM")
    groups = [("off, all winners", off_all), ("p1, AG wins", p1_ag), ("p1, GEMM wins", p1_gemm)]
    rows, meds = [], {}
    for label, xs in groups:
        dg = dist([x["gapG"] for x in xs])
        da = dist([x["dA"] for x in xs])
        dd = dist([x["gapG"] - x["dA"] for x in xs])
        meds[label] = dg["med"]
        rows.append((label, dg["n"], f1(dg["med"]), f1(dg["p10"]), f1(dg["p90"]), f1(dg["mx"]),
                     f1(da["med"]), f1(dd["med"])))
    L += E.md_table(["group", "n", "gapG median", "gapG p10", "gapG p90", "gapG max", "dA median",
                     "gapG minus dA median"], rows)
    L.append(f"gapG median of p1 AG wins minus p1 GEMM wins: {f1(meds['p1, AG wins'] - meds['p1, GEMM wins'])} us. "
             f"Minus off: {f1(meds['p1, AG wins'] - meds['off, all winners'])} us.\n")
    pag = Counter()
    for x in p1_ag:
        pag.update(case_keys(x))
    pother = Counter()
    for x in p1_gemm + off_all:
        pother.update(case_keys(x))
    only = sorted(k for k in pag if k not in pother)
    L.append(f"Keys (sections 2 and 3) present in p1 AG-win cases and in no p1 GEMM-win or off case: {len(only)}.\n")
    if only:
        L += E.md_table(["key", "p1 AG-win cases with it (of {})".format(len(p1_ag))],
                        [(" | ".join(str(v) for v in k), pag[k]) for k in only])
    else:
        L.append("None.\n")
    gone = sorted(k for k in pother if k not in pag)
    L.append(f"Keys present in p1 GEMM-win or off cases and absent from p1 AG-win cases: {len(gone)}.\n")
    if gone:
        L += E.md_table(["key", "cases with it in p1 GEMM wins and off"],
                        [(" | ".join(str(v) for v in k), pother[k]) for k in gone[:50]])
    L.append("Keys present in both groups, with the share of cases having the key:\n")
    both = sorted(k for k in pag if k in pother)
    rows = []
    for k in both:
        rows.append((" | ".join(str(v) for v in k), f"{pag[k] / len(p1_ag):.3f}",
                     f"{pother[k] / (len(p1_gemm) + len(off_all)):.3f}"))
    L += E.md_table(["key", "p1 AG wins share", "p1 GEMM wins + off share"], rows)
    return L


def sec5(recs):
    L = ["## 5. Per rank", ""]
    L.append("n = X-gated cases of the rank. AG wins = winner AG among them. Medians over the cases of the rank.\n")
    for arm, _ in ARMS:
        rows = []
        for r in RANKS:
            xs = [x for x in recs if x["arm"] == arm and x["rank"] == r]
            rows.append((r, f"node {r // 8}", len(xs), sum(1 for x in xs if x["winner"] == "AG"),
                         f1(dist([x["gapN"] for x in xs])["med"]), f1(dist([x["gapG"] for x in xs])["med"])))
        L.append(f"### {arm}\n")
        L += E.md_table(["rank", "node", "n", "AG wins", "median gapN (us)", "median gapG (us)"], rows)
    return L


def sec6():
    return ["## 6. Findings", "", FINDINGS.strip(), ""]


def sec7(an_by_arm, ver_by_arm, meta_by_arm, recs):
    L = ["## 7. Checks, anomalies and cross-checks", ""]
    keys = sorted({k for arm, _ in ARMS for k in an_by_arm[arm]})
    rows = [(k,) + tuple(an_by_arm[arm].get(k, 0) for arm, _ in ARMS) for k in keys]
    L += E.md_table(["loader and case counts (summed over 16 ranks)"] + [arm for arm, _ in ARMS], rows)
    keys = sorted({k for arm, _ in ARMS for k in ver_by_arm[arm]})
    rows = [(k,) + tuple(ver_by_arm[arm].get(k, 0) for arm, _ in ARMS) for k in keys]
    L += E.md_table(["second read of sqlite (summed over 16 ranks; mismatches should be 0)"]
                    + [arm for arm, _ in ARMS], rows)
    rows = []
    for arm, _ in ARMS:
        cs = sorted({meta_by_arm[arm][r]["compute"] for r in RANKS})
        inner = sorted({str(meta_by_arm[arm][r]["inner"]) for r in RANKS})
        rows.append((arm, ", ".join(str(c) for c in cs), "; ".join(inner)))
    L += E.md_table(["arm", "compute stream id(s)", "inner AG-I (stream, gridX, name)"], rows)
    rows = []
    for arm, _ in ARMS:
        xs = sel(recs, arm)
        idle = sum(1 for x in xs if x["A_idle"])
        rows.append((arm, len(xs), RACE_REF[arm], sum(1 for x in xs if x["E_busy"]), idle,
                     sum(1 for x in xs if x["E_busy"] and x["A_idle"]), RACE_REF_IDLE_A[arm],
                     f1(dist([x["dA"] for x in xs])["med"])))
    L.append("Cross-check against RACE.md. `E busy` is whether stream 7 is busy at E (end of AG-I k). `A idle` "
             "is whether stream 7 has no op spanning the start of AG-I k+1. RACE.md counts the busy-at-E cases, "
             "so the reference is the same set except for the 5 idle-at-E cases here.\n")
    L += E.md_table(["arm", "X-gated cases here", "RACE.md busy-at-E cases", "E busy here", "A idle here",
                     "E busy and A idle here", "RACE.md busy-at-E, A idle", "dA median (us)"], rows)
    rows = []
    for arm, _ in ARMS:
        for w in ["all"] + WINNERS:
            xs = sel(recs, arm, w)
            rows.append((arm, w, len(xs), sum(1 for x in xs if x["A_before_N"]),
                         sum(1 for x in xs if x["A_idle"]), sum(1 for x in xs if x["spans"])))
    L.append("Winner checks: A before N.start (AG-I k+1 starts while N is still queued on the GPU), A idle, and "
             "cases with a spanning op on another stream at R.\n")
    L += E.md_table(["arm", "winner", "cases", "A before N.start", "A idle", "cases with spanning op"], rows)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    recs, an_by_arm, ver_by_arm, meta_by_arm = [], {}, {}, {}
    for arm, adir in ARMS:
        rr, an, meta, ver = run_arm(arm, adir)
        recs += rr
        an_by_arm[arm] = an
        ver_by_arm[arm] = ver
        meta_by_arm[arm] = meta
        print(f"[arm] {arm}: {len(rr)} X-gated AG-I k+1", flush=True)

    L = ["# E2 race 2: compute stream after the gating kernel, at the start of the next inner all-gather", ""]
    L.append("Script: `bench/e2race2_compare.py`. It reuses the loader, AG-I detection, phase membership and the "
             "relay-chain gate resolution of `bench/e2race_compare.py` and `bench/e2skew_compare.py`. Inputs: the "
             "nsys sqlite exports of `d4096_L32_T4096_ga4.rank{0..15}` for `01-v2-off` (fast) and `02-v2-p1` (slow). "
             "Cases: AG-I k+1 in mb1..3 forward (same phase instance as AG-I k) whose gate is a kernel X on the "
             "compute stream. Measurements are within one rank.\n")
    L.append("Definitions: R = end of X. N = next compute-stream op after X. G = first GEMM on the compute stream "
             "after X. A = start of AG-I k+1. Winner AG if A < G.start, else GEMM. Times are microseconds from R.\n")
    L += sec1(recs)
    L += sec2(recs)
    L += sec3(recs)
    L += sec4(recs)
    L += sec5(recs)
    L += sec6()
    L += sec7(an_by_arm, ver_by_arm, meta_by_arm, recs)
    text = "\n".join(L) + "\n"
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
