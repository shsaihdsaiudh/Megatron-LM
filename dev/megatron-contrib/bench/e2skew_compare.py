#!/usr/bin/env python3
"""Per-collective skew of the inner FSDP all-gather (AG-I) in the e2 two-node nsys runs.

Question: in mb1..3 forward, are the long AG-I kernels long because the ranks of one node reach the
collective at different times (the kernel spins until the last rank arrives), or are they slow even
when all 8 ranks start together? And what delays the last rank?

Inputs: nsys sqlite exports of d4096_L32_T4096_ga4.rank{0..15}.nsys-rep for each arm. A missing
.sqlite is exported next to its .nsys-rep. Membership, detection and kernel naming come from
bench/e1v1_nsys_compare.py. Only the 8 ranks of one node are compared (node 0 = ranks 0-7,
node 1 = ranks 8-15). Node 0 and node 1 are never compared.

Clock: the ranks of a node do not share a timestamp origin in these exports (raw end skews are a
constant 1.3-2.2 s). Each rank gets a constant offset, the median difference of its AG-I end times to the
first rank of its node. Every cross-rank quantity uses the corrected times. Same-rank quantities (launch
lead, previous-op gap, waited events) use raw times.

Output: SKEW.md (sections 1-7 plus anomalies), also printed to stdout.

Usage: python3 bench/e2skew_compare.py [--out PATH]
"""

import argparse
import bisect
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1v1_nsys_compare as E  # noqa: E402

ROOT = os.environ.get("E2SKEW_ROOT", "/planning-users/yangyeyuan/megatron-contrib/results/e2/e2skew-2node-20261116")
DEFAULT_OUT = f"{ROOT}/SKEW.md"
# E2SKEW_ARM_DIRS="DIR_OFF DIR_P1" overrides the run dirs (relative to ROOT; use symlinks). Labels stay
# "off" (reference) and "p1" (arm under test) because the reports compare these two keys.
_dirs = os.environ.get("E2SKEW_ARM_DIRS", "01-v2-off 02-v2-p1").split()
ARMS = [("off", _dirs[0]), ("p1", _dirs[1])]
NSYS = "/usr/local/bin/nsys"
NODES = [0, 1]
RANKS = list(range(16))
NS = 1e6                      # ns per ms
SLOW_MS = 4.5                 # GEMM above this is "slow" (task definition)
SKEW_MIN_MS = 0.1             # start skew below this: the 8 ranks started together
TIGHT_RATIO = 0.2             # end skew <= ratio * start skew: ends aligned, the kernel waited for the last rank
LAG_MS = 0.1                  # a rank that started this much before the last rank of its node is "early"
BUSY_MS = 0.02                # previous stream op ended within this of the AG start: "just before"
ENQ_MS = 0.05                 # AG enqueued within this of its GPU start: launched just in time
GATE_MS = 0.05                # a waited event completes within this of the AG start: it gates the AG
PHASES_FWD = ["mb1_fwd", "mb2_fwd", "mb3_fwd"]
GROUPS = [("mb0 fwd", ["mb0_fwd"]),
          ("mb1..3 fwd", PHASES_FWD),
          ("bwd, mb0..3", ["mb0_bwd", "mb1_bwd", "mb2_bwd", "mb3_bwd"])]
PHASES_ALL = ["mb0_fwd", "mb1_fwd", "mb2_fwd", "mb3_fwd", "mb0_bwd", "mb1_bwd", "mb2_bwd", "mb3_bwd"]
CLASS_NAMES = {
    "a": "(a) stream-serialized: enqueued behind the previous op on the inner stream",
    "b": "(b) CPU-bound: enqueued just before start, inner stream idle before",
    "c": "(c) launched early, GPU start delayed while the inner stream was idle",
    "u": "unclassifiable (no previous op on the stream)",
}

# Section 7 text, written after the numbers were checked.
FINDINGS = """In mb1..3 forward, the long p1 AG-I kernels are waits for the last-arriving rank. They are not slow kernels. Median corrected start skew within a node is 5.35 ms (node 0) and 5.25 ms (node 1) in p1, against 0.40 ms and 0.37 ms in off. Median end skew is 0.16 ms and 0.17 ms in p1, and 0.13 ms and 0.11 ms in off. For non-last ranks, duration fits the wait for the last rank with slope 0.996 and intercept -0.08 ms (2688 ranks; off slope 0.943). The last starter's AG-I median is 0.79 ms in both arms, while the other ranks' median is 3.63 ms (node 0) and 4.05 ms (node 1) in p1. When all 8 ranks start together (6 collectives in p1, 19 in off), the AG-I median is 0.76 ms in off and 0.77 ms in p1.

The latest rank's delay is on the GPU, not the CPU and not an event. Its AG-I was enqueued a median 1.43 s before it started (p1; 1.40 s off). None of the 384 p1 last starters had a previous inner-stream op end just before its start (class a) or a late enqueue (class b). Of the waited events, 358 had completed earlier, 2 gated the start, and 24 had no gating op in the trace. Its AG-I started a median 0.146 ms before the end of the GEMM running on its compute stream (p10-p90 0.143-0.149 ms), and 370 of 384 had a GEMM running at start. That GEMM had a median duration of 3.96 ms, and none was slow. Early ranks in p1 had a GEMM running at AG start in 1106 of 2688 cases, against 2603 of 2688 in off, so 59% of early-rank AG starts in p1 found no GEMM on the compute stream. Slow GEMMs (>4.5 ms) are 23.8 per microbatch across 16 ranks in p1. Of those GEMMs, 58.9% overlapped an early AG-I, and none overlapped a last-or-near-last AG-I. Per-rank Pearson r between early-AG-I count and slow-GEMM count is 0.55 (96 points, p1) and -0.01 (off). Whether the 0.15 ms tail is SM availability held by the GEMM is inferred from this fixed lead and was not measured."""


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pctl(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def fmt(x, nd=3):
    return "n/a" if x != x else f"{x:.{nd}f}"


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def md(headers, rows):
    return E.md_table(headers, rows)


def sqlite_path(arm_dir, rank):
    base = f"{ROOT}/{arm_dir}/nsys/d4096_L32_T4096_ga4.rank{rank}"
    if not os.path.exists(base + ".sqlite"):
        subprocess.run([NSYS, "export", "--type", "sqlite", "-o", base + ".sqlite", base + ".nsys-rep"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return base + ".sqlite"


def pred_op(D, stream, corr):
    """Last op on `stream` launched before correlation id `corr`. Stream order equals launch order."""
    keys = D["bsc_keys"].get(stream)
    if not keys:
        return None
    i = bisect.bisect_left(keys, corr) - 1
    return D["bsc"][stream][i] if i >= 0 else None


def step_of(D, t):
    if t is None:
        return -1
    for i, (s, e, _lab) in enumerate(D["steps"]):
        if s <= t <= e:
            return i
    return -1


def wait_info(D, o, prev):
    """cudaStreamWaitEvent calls on o's stream between prev and o, and the op the latest waited-for event follows."""
    ws = [w for w in D["waits"].get(o.stream, []) if prev.corr < w[0] < o.corr]
    info = {"n": len(ws), "cat": "none in window", "gate_name": None, "gate_rel": None, "gate_stream": None}
    best = None
    unresolved_rec = 0
    unresolved_pred = 0
    for wcorr, _ws, eid, esid in ws:
        rec = D["recs"].get((eid, esid))
        if rec is None:
            unresolved_rec += 1
            continue
        rstream, rcorr = rec
        pred = pred_op(D, rstream, rcorr)
        if pred is None:
            unresolved_pred += 1
            continue
        if best is None or pred.end > best[1].end:
            best = (wcorr, pred, rstream)
    if best is None:
        if ws:
            info["cat"] = "record not in trace" if unresolved_rec else "recording stream had no earlier op"
        return info
    _wc, pred, rstream = best
    rel = (pred.end - o.start) / NS
    info["gate_name"] = E.kname(pred)
    info["gate_rel"] = rel
    info["gate_stream"] = rstream
    if abs(rel) <= GATE_MS:
        info["cat"] = "event completes at AG start (gating)"
    elif rel < 0:
        info["cat"] = "event completed before AG start"
    else:
        info["cat"] = "event completes after AG start"
    return info


def classify(D, o, prev, enq):
    if prev is None:
        return "u", None
    if prev.end >= o.start - BUSY_MS * NS:
        if enq <= prev.end:
            return "a", None
        return "b", None
    if o.start - enq < ENQ_MS * NS:
        return "b", None
    return "c", wait_info(D, o, prev)


def load_rank(arm_dir, rank):
    sq = sqlite_path(arm_dir, rank)
    D = E.load(sq)
    c = sqlite3.connect(sq)
    S = dict(c.execute("select id, value from StringIds"))
    D["rt"] = {}
    D["rt_dup"] = 0
    for corr, s, e, nid in c.execute("select correlationId, start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if corr in D["rt"]:
            D["rt_dup"] += 1
        D["rt"][corr] = (s, e, S.get(nid, ""))
    D["steps"] = []
    for text, tid, s, e in c.execute(
            "select text, textId, start, end from NVTX_EVENTS where end is not null order by start"):
        lab = text if text is not None else S.get(tid)
        if lab and lab.startswith("train_T4096"):
            D["steps"].append((s, e, lab))
    D["recs"] = {}
    D["rec_dup"] = 0
    D["rec_other"] = 0
    for corr, sid, eid, esid in c.execute(
            "select correlationId, streamId, eventId, eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT"):
        if "EventRecord" not in D["rt"].get(corr, (0, 0, ""))[2]:
            D["rec_other"] += 1
            continue
        if (eid, esid) in D["recs"]:
            D["rec_dup"] += 1
        D["recs"][(eid, esid)] = (sid, corr)
    D["waits"] = defaultdict(list)
    D["wait_bad"] = 0
    for s, _e, sid, corr, eid, esid in c.execute(
            "select start, end, streamId, correlationId, eventId, eventSyncId "
            "from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType = 2"):
        if "StreamWaitEvent" not in D["rt"].get(corr, (0, 0, ""))[2]:
            D["wait_bad"] += 1
        D["waits"][sid].append((corr, s, eid, esid))
    for v in D["waits"].values():
        v.sort()
    bsc = defaultdict(list)
    for o in D["ops"]:
        bsc[o.stream].append((o.corr, o))
    D["bsc_keys"], D["bsc"] = {}, {}
    for s, v in bsc.items():
        v.sort(key=lambda t: t[0])
        D["bsc_keys"][s] = [t[0] for t in v]
        D["bsc"][s] = [t[1] for t in v]
    c.close()
    return D


def ag_records(D):
    """AG-I records of one rank in stream order. Record k is the k-th collective on this rank."""
    inner = D["inner_key"]
    s_in = [o for o in D["ops"] if o.stream == inner[0]]
    out = []
    cnt = Counter()
    for pos, o in enumerate(s_in):
        if not E.is_inner(D, o):
            continue
        prev = s_in[pos - 1] if pos > 0 else None
        rt = D["rt"].get(o.corr)
        launch = rt[0] if rt else o.start
        enq = rt[1] if rt else o.start
        step = step_of(D, launch)
        phase = o.cpu[0] if o.cpu else "outside"
        idx = cnt[(step, phase)]
        cnt[(step, phase)] += 1
        r = E.running_op(D, o.start)
        cls, info = classify(D, o, prev, enq)
        out.append(dict(
            k=len(out), step=step, phase=phase, idx=idx,
            start=o.start, end=o.end, dur=o.dur, launch=launch, enq=enq,
            lead=(o.start - enq) / NS,
            prev_name=E.kname(prev) if prev else "(none)",
            pgap=(o.start - prev.end) / NS if prev else float("nan"),
            run=E.kname(r) if r is not None else "(compute stream idle)",
            run_gemm=bool(r is not None and r.is_gemm),
            run_left=(r.end - o.start) / NS if r is not None else float("nan"),
            run_dur=(r.end - r.start) / NS if r is not None else float("nan"),
            cls=cls, info=info,
        ))
    return out


def apply_offsets(recs_by_rank):
    """Constant clock offset per rank: median of (own AG-I end - first rank's AG-I end) over the node's collectives.

    Adds cstart, cend and claunch (ns) to every record of the node. Returns {rank: offsets in ms}.
    """
    out = {}
    for n in NODES:
        rks = [8 * n + i for i in range(8)]
        ref = rks[0]
        n_k = min(len(recs_by_rank[r]) for r in rks)
        for r in rks:
            diffs = [recs_by_rank[r][k]["end"] - recs_by_rank[ref][k]["end"] for k in range(n_k)]
            off = med(diffs)
            per_step = {}
            for s in (0, 1):
                ds = [recs_by_rank[r][k]["end"] - recs_by_rank[ref][k]["end"]
                      for k in range(n_k) if recs_by_rank[ref][k]["step"] == s]
                per_step[s] = med(ds)
            for x in recs_by_rank[r][:n_k]:
                x["cstart"] = x["start"] - off
                x["cend"] = x["end"] - off
                x["claunch"] = x["launch"] - off
            out[r] = dict(off=off / NS, s0=per_step[0] / NS, s1=per_step[1] / NS)
    return out


def best_overlap(recs, starts, a, b):
    """Index (= k) of the AG-I on the same rank with the largest overlap with [a, b), or None."""
    j = max(0, bisect.bisect_right(starts, a) - 1)
    best_x, best_j = 0, None
    while j < len(recs) and recs[j]["start"] < b:
        x = min(b, recs[j]["end"]) - max(a, recs[j]["start"])
        if x > best_x:
            best_x, best_j = x, j
        j += 1
    return best_j


def gemm_records(D, recs):
    starts = [r["start"] for r in recs]
    out = []
    for key, lst in D["members"].items():
        if key[0] not in PHASES_FWD:
            continue
        for g in lst:
            if g.stream != D["compute"] or not g.is_gemm:
                continue
            out.append(dict(dur=g.dur, step=step_of(D, g.launch), phase=key[0],
                            ag_k=best_overlap(recs, starts, g.start, g.end)))
    return out


def load_arm(arm_dir):
    data, meta = {}, {}
    for r in RANKS:
        D = load_rank(arm_dir, r)
        if D["inner_key"] is None:
            raise SystemExit(f"{arm_dir} rank {r}: no AG-I found")
        recs = ag_records(D)
        data[r] = dict(recs=recs, gemm=gemm_records(D, recs))
        meta[r] = dict(
            file=os.path.basename(D["name"]),
            compute=D["compute"],
            inner=(D["inner_key"][0], D["inner_key"][1]),
            outer=(D["outer_key"][0], D["outer_key"][1]) if D["outer_key"] else None,
            unmatched=D["unmatched"], multi=D["multi"], rt_dup=D["rt_dup"],
            rec_dup=D["rec_dup"], rec_other=D["rec_other"], wait_bad=D["wait_bad"],
            n_steps=len(D["steps"]), step_labels=[lab for _s, _e, lab in D["steps"]],
        )
        print(f"[load] {arm_dir} rank {r}: AG-I {len(recs)}, GEMM mb1..3 fwd {len(data[r]['gemm'])}, "
              f"inner {meta[r]['inner']}, unmatched {D['unmatched']}", flush=True)
        del D
    return data, meta


def build_instances(data):
    """Pair the k-th AG-I of the 8 ranks of each node. Returns collectives, max corrected start per (node, k),
    counts per rank, and the number of collectives whose ranks disagree on (step, phase)."""
    insts, maxstart, counts, tag_mis = [], {}, {}, 0
    for n in NODES:
        rks = [8 * n + i for i in range(8)]
        counts[n] = {r: len(data[r]["recs"]) for r in rks}
        n_k = min(counts[n].values())
        for k in range(n_k):
            per = {r: data[r]["recs"][k] for r in rks}
            tags = {(x["step"], x["phase"]) for x in per.values()}
            if len(tags) > 1:
                tag_mis += 1
            st = {r: per[r]["cstart"] for r in rks}
            en = {r: per[r]["cend"] for r in rks}
            st_raw = {r: per[r]["start"] for r in rks}
            en_raw = {r: per[r]["end"] for r in rks}
            ss = (max(st.values()) - min(st.values())) / NS
            es = (max(en.values()) - min(en.values())) / NS
            ss_raw = (max(st_raw.values()) - min(st_raw.values())) / NS
            es_raw = (max(en_raw.values()) - min(en_raw.values())) / NS
            last = max(rks, key=lambda r: st[r])
            others = [per[r]["dur"] for r in rks if r != last]
            if ss < SKEW_MIN_MS:
                kind = "together"
            elif es <= TIGHT_RATIO * ss:
                kind = "tight"
            else:
                kind = "loose"
            insts.append(dict(
                node=n, k=k, step=per[rks[0]]["step"], phase=per[rks[0]]["phase"],
                ss=ss, es=es, ss_raw=ss_raw, es_raw=es_raw, last=last, last_rec=per[last],
                maxend=max(en.values()), pr={r: per[r] for r in rks},
                last_dur=per[last]["dur"], oth_med=med(others),
                durs=[per[r]["dur"] for r in rks], kind=kind,
            ))
            maxstart[(n, k)] = max(st.values())
    return insts, maxstart, counts, tag_mis


def sel(insts, phases, node=None):
    return [x for x in insts if x["phase"] in phases and (node is None or x["node"] == node)]


def sec1(data, meta, offs, insts):
    L = ["## 1. Sanity", ""]
    L.append("### fwd_rest in train mode at T=4096 (ms, max over ranks)\n")
    rows = []
    for mode in ["train", "nostep"]:
        for label, fname in [("sweep, no nsys (e1_d4096_L32.json)", "e1_d4096_L32.json"),
                             ("nsys run (nsys_d4096_L32_T4096_ga4.json)", "nsys_d4096_L32_T4096_ga4.json")]:
            vals = {}
            for arm, adir in ARMS:
                with open(f"{ROOT}/{adir}/{fname}") as f:
                    J = json.load(f)
                ent = next((e for e in J["results"] if e.get("mode") == mode and e.get("tokens") == 4096), None)
                vals[arm] = ent["max"]["fwd_rest"] if ent else None
            if vals["off"] is None or vals["p1"] is None:
                continue
            rows.append((f"{mode}: {label}", f"{vals['off']:.2f}", f"{vals['p1']:.2f}",
                         f"{vals['p1'] - vals['off']:+.2f}"))
    L += md(["source", "off", "p1", "p1 - off"], rows)
    nlog = {}
    for arm, adir in ARMS:
        n = 0
        for node in ("node0", "node1"):
            with open(f"{ROOT}/{adir}/{node}.nsys_d4096_T4096_ga4.log", errors="replace") as f:
                n += sum(1 for line in f if "fwd_rest" in line)
        nlog[arm] = n
    L.append(f"The nsys run logs (`node{{0,1}}.nsys_d4096_T4096_ga4.log`) contain {nlog['off']} (off) and {nlog['p1']} "
             "(p1) `fwd_rest` lines, so the nsys row uses `nsys_d4096_L32_T4096_ga4.json`.\n")
    L.append("### AG-I per rank per phase (count, min-max over 16 ranks)\n")
    rows = []
    for ph in PHASES_ALL + ["outside"]:
        cells = []
        for arm, _ in ARMS:
            per = [sum(1 for x in data[arm][r]["recs"] if x["phase"] == ph) for r in RANKS]
            cells.append(f"{min(per)}-{max(per)}")
        rows.append((ph, cells[0], cells[1]))
    L += md(["phase (CPU-launch membership)", "off", "p1"], rows)
    L.append("Expected: 64 per phase per rank (2 steps x 32 AG-I per microbatch). AG-I per rank in total: "
             + ", ".join(f"{arm} {len(data[arm][0]['recs'])}" for arm, _ in ARMS) + ".\n")
    L.append("### Clock offsets within each node\n")
    L.append("Offsets are relative to the first rank of the node. Raw end skew is the max-min of the 8 raw end "
             "times of one collective. Corrected skew uses the shifted times.\n")
    rows = []
    for arm, _ in ARMS:
        for r in RANKS:
            o = offs[arm][r]
            rows.append((arm, r, f"node {r // 8}", f"{o['off']:+.3f}", f"{o['s0']:+.3f}", f"{o['s1']:+.3f}",
                         f"{o['s1'] - o['s0']:+.4f}"))
    L += md(["arm", "rank", "node", "offset (ms)", "offset, step 0 (ms)", "offset, step 1 (ms)",
             "step 1 minus step 0 (ms)"], rows)
    rows = []
    for arm, _ in ARMS:
        for n in NODES:
            xs = [x for x in insts[arm] if x["node"] == n]
            rows.append((arm, f"node {n}", fmt(med([x["es_raw"] for x in xs])), fmt(med([x["ss_raw"] for x in xs])),
                         fmt(med([x["es"] for x in xs])), fmt(pctl([x["es"] for x in xs], .9)),
                         fmt(max(x["es"] for x in xs)), fmt(med([x["ss"] for x in xs]))))
    L += md(["arm", "node", "raw end skew median (ms)", "raw start skew median (ms)",
             "corrected end skew median (ms)", "corrected end skew p90 (ms)", "corrected end skew max (ms)",
             "corrected start skew median (ms)"], rows)
    return L


def sec2(insts):
    L = ["## 2. Skew and duration per node", ""]
    L.append("Start skew is max(start) - min(start) over the 8 ranks of one collective, on corrected times. End skew is "
             "max(end) - min(end). `together`: start skew < 0.1 ms. `tight`: start skew >= 0.1 ms and end skew <= 0.2 x "
             "start skew, so the ends line up and the kernel waited for the last rank. `loose`: other.\n")
    rows_a, rows_b = [], []
    for arm, _ in ARMS:
        for n in NODES:
            for gname, phases in GROUPS:
                xs = sel(insts[arm], phases, n)
                ss = [x["ss"] for x in xs]
                es = [x["es"] for x in xs]
                durs = [d for x in xs for d in x["durs"]]
                kc = Counter(x["kind"] for x in xs)
                rows_a.append((arm, f"node {n}", gname, len(xs),
                               fmt(med(ss)), fmt(pctl(ss, .9)), fmt(max(ss)),
                               fmt(med(es)), fmt(pctl(es, .9)), fmt(max(es)),
                               kc["together"], kc["tight"], kc["loose"]))
                rows_b.append((arm, f"node {n}", gname,
                               fmt(med(durs)), fmt(pctl(durs, .9)), fmt(max(durs)),
                               fmt(med([x["last_dur"] for x in xs])), fmt(med([x["oth_med"] for x in xs])),
                               fmt(med([x["oth_med"] - x["last_dur"] for x in xs]))))
    L.append("### Start and end skew (ms, corrected)\n")
    L += md(["arm", "node", "phase group", "collectives", "start skew median", "start skew p90", "start skew max",
             "end skew median", "end skew p90", "end skew max", "together", "tight", "loose"], rows_a)
    L.append("### AG-I duration (ms)\n")
    L.append("Durations are over all 8 ranks of each collective. Last-starter duration is the duration of the rank "
             "that started last. Others median is the median of the 7 other ranks per collective. The last two "
             "columns are medians over collectives.\n")
    L += md(["arm", "node", "phase group", "AG-I dur median", "AG-I dur p90", "AG-I dur max",
             "last-starter dur median", "others median dur", "others minus last (median)"], rows_b)
    L.append("### AG-I duration by collective kind (mb1..3 forward)\n")
    rows_k = []
    for arm, _ in ARMS:
        xs = sel(insts[arm], PHASES_FWD)
        for kind in ["together", "tight", "loose"]:
            ks = [x for x in xs if x["kind"] == kind]
            rows_k.append((arm, kind, len(ks), fmt(med([d for x in ks for d in x["durs"]])),
                           fmt(med([x["last_dur"] for x in ks])),
                           fmt(med([x["oth_med"] for x in ks])) if ks else "n/a"))
    L += md(["arm", "kind", "collectives", "AG-I dur median, all 8 ranks", "last-starter dur median",
             "others median dur"], rows_k)
    L.append("### Spin check: duration of non-last ranks against the wait for the last rank (mb1..3 forward)\n")
    L.append("For a rank that started early, a pure wait gives duration = (latest corrected end) - (own corrected "
             "start). The fit is duration = a + b x (latest end - own start), over all non-last ranks. A pure wait "
             "gives b near 1 and a near 0.\n")
    rows_s = []
    for arm, _ in ARMS:
        xs_, ys_ = [], []
        for x in sel(insts[arm], PHASES_FWD):
            for r, rec in x["pr"].items():
                if r == x["last"]:
                    continue
                xs_.append((x["maxend"] - rec["cstart"]) / NS)
                ys_.append(rec["dur"])
        a, b = ls_fit(xs_, ys_)
        resid = [y - xx for xx, y in zip(xs_, ys_)]
        rows_s.append((arm, len(xs_), fmt(b, 3), fmt(a, 3), fmt(med(resid)), fmt(pctl([abs(v) for v in resid], .9))))
    L += md(["arm", "non-last ranks", "slope b", "intercept a (ms)", "median duration minus wait (ms)",
             "p90 of |duration minus wait| (ms)"], rows_s)
    return L


def ls_fit(x, y):
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    b = sxy / sxx if sxx > 0 else float("nan")
    return my - b * mx, b


def sec3(insts):
    L = ["## 3. Which rank starts last (mb1..3 forward, corrected times)", ""]
    rows, hist_rows = [], []
    for arm, _ in ARMS:
        xs = sel(insts[arm], PHASES_FWD)
        hist = Counter(x["last"] for x in xs)
        hist_rows.append([arm] + [str(hist[r]) for r in RANKS])
        for n in NODES:
            xn = [x for x in xs if x["node"] == n]
            hn = Counter(x["last"] for x in xn)
            top, tc = hn.most_common(1)[0]
            by_step = {}
            for s in (0, 1):
                hs = Counter(x["last"] for x in xn if x["step"] == s)
                t, c = hs.most_common(1)[0]
                by_step[s] = (t, c, sum(hs.values()))
            rows.append((arm, f"node {n}", f"rank {top}: {tc}/{len(xn)}",
                         f"step 0: rank {by_step[0][0]} ({by_step[0][1]}/{by_step[0][2]}); "
                         f"step 1: rank {by_step[1][0]} ({by_step[1][1]}/{by_step[1][2]})",
                         "yes" if by_step[0][0] == by_step[1][0] else "no"))
    L.append("Histogram of the last-starting rank over all mb1..3 forward collectives (both steps):\n")
    L += md(["arm"] + [f"r{r}" for r in RANKS], hist_rows)
    L += md(["arm", "node", "most often last (share)", "top rank per step", "same top rank in both steps"], rows)
    return L


def sec4(insts):
    L = ["## 4. Why the last-starting rank started late (mb1..3 forward)", ""]
    L.append("The class of each collective is decided on its own rank, so it uses raw times. Lead = GPU start minus "
             "CUPTI runtime-call end (enqueue). pgap = GPU start minus end of the previous op on the inner stream. "
             "run left = end of the compute-stream op running at the start, minus the start. Rules in `classify()`: "
             "(a) the previous op ended within 0.02 ms before start and the collective was enqueued before that end; "
             "(b) enqueued within 0.05 ms before start with the stream idle before; (c) enqueued at least 0.05 ms "
             "before start with the stream idle for at least 0.02 ms before start.\n")
    rows, c_rows, gate_lines, kind_lines = [], [], [], []
    for arm, _ in ARMS:
        last = [x["last_rec"] for x in sel(insts[arm], PHASES_FWD)]
        for c in ["a", "b", "c", "u"]:
            s_ = [r for r in last if r["cls"] == c]
            pg = [r["pgap"] for r in s_ if r["pgap"] == r["pgap"]]
            rl = [r["run_left"] for r in s_ if r["run_left"] == r["run_left"]]
            rows.append((arm, CLASS_NAMES[c], f"{len(s_)} / {len(last)}",
                         fmt(med([r["lead"] for r in s_])), fmt(med(pg)),
                         fmt(med([r["dur"] for r in s_])),
                         sum(1 for r in s_ if r["run_gemm"]),
                         sum(1 for r in s_ if r["run"] == "(compute stream idle)"),
                         sum(1 for x in rl if x <= GATE_MS)))
        cs = [r for r in last if r["cls"] == "c"]
        cats = Counter(r["info"]["cat"] for r in cs)
        c_rows.append((arm, len(cs), cats["none in window"], cats["event completes at AG start (gating)"],
                       cats["event completed before AG start"], cats["event completes after AG start"],
                       cats["record not in trace"] + cats["recording stream had no earlier op"]))
        gates = Counter(r["info"]["gate_name"] for r in cs if r["info"]["cat"] == "event completes at AG start (gating)")
        if gates:
            gate_lines.append(f"- {arm}: gating events were recorded after: "
                              + "; ".join(f"{g} ({v})" for g, v in gates.most_common(5)) + ".")
        else:
            gate_lines.append(f"- {arm}: no gating event.")
        kinds = Counter((x["kind"], x["last_rec"]["cls"]) for x in sel(insts[arm], PHASES_FWD))
        kind_lines.append(f"- {arm}: " + "; ".join(f"{k} kind, class {c}: {v}" for (k, c), v in sorted(kinds.items()))
                          + ".")
    L += md(["arm", "class", "last starters in class", "median lead (ms)", "median pgap (ms)",
             "median AG-I dur of last starter (ms)", "compute stream on GEMM at start",
             "compute stream idle at start", "running op ends within 0.05 ms after start"], rows)
    L.append("Class (c) detail. The waited-for event is the record with the same event id and sync id. Its "
             "predecessor is the last op on the recording stream before the record. `gating` means that predecessor "
             "ends within 0.05 ms of the AG start.\n")
    L += md(["arm", "class (c) last starters", "no cudaStreamWaitEvent in window", "gating (event ends at AG start)",
             "event completed before AG start", "event ends after AG start", "no gating op found"], c_rows)
    L += gate_lines
    L.append("")
    L.append("Blocking GEMM: the compute-stream op running on the rank at the AG-I start (mb1..3 forward). `ends within "
             "0.05 ms` counts AG-I that start at most 0.05 ms before that op ends.\n")
    rows_g = []
    for arm, _ in ARMS:
        xs = sel(insts[arm], PHASES_FWD)
        every = [rec for x in xs for rec in x["pr"].values()]
        lastr = [x["last_rec"] for x in xs]
        nonlast = [rec for x in xs for r, rec in x["pr"].items() if r != x["last"]]
        for label, rs in [("every rank", every), ("non-last ranks", nonlast), ("last starters", lastr)]:
            rows_g.append((arm, label, len(rs), sum(1 for r in rs if r["run_gemm"]),
                           sum(1 for r in rs if r["run_left"] == r["run_left"] and r["run_left"] <= GATE_MS),
                           fmt(med([r["run_dur"] for r in rs if r["run_gemm"]])),
                           sum(1 for r in rs if r["run_gemm"] and r["run_dur"] > SLOW_MS),
                           f"{fmt(med([r['run_left'] for r in rs if r['run_gemm']]))} ({fmt(pctl([r['run_left'] for r in rs if r['run_gemm']], .1))}-{fmt(pctl([r['run_left'] for r in rs if r['run_gemm']], .9))})"))
    L += md(["arm", "set", "AG-I", "GEMM running at start", "GEMM ends within 0.05 ms after start",
             "blocking GEMM median dur (ms)", "blocking GEMM slow (>4.5 ms)", "median GEMM time left at start (ms), p10-p90"], rows_g)
    L.append("")
    L.append("Collective kind (corrected) and class of the last starter (mb1..3 forward):\n")
    L += kind_lines
    L.append("")
    return L


def sec5(data, insts, maxstart):
    L = ["## 5. GEMM slowdown and early arrival of the overlapping AG-I (mb1..3 forward)", ""]
    L.append("Each GEMM on the compute stream is matched to the AG-I on the same rank that overlaps it most. That "
             "AG-I is `early` when its rank started at least 0.1 ms before the last rank of its node (corrected "
             "times). Otherwise it is `last or near-last`. Slow means duration above 4.5 ms. Per-microbatch values "
             "average over the 6 instances (2 steps x mb1..3).\n")
    n_mb = 6
    cross_rows, r_lines = [], []
    per_rank_tables = {}
    last_of = {}
    for arm, _ in ARMS:
        for x in insts[arm]:
            last_of[(arm, x["node"], x["k"])] = x["last"]
    for arm, _ in ARMS:
        cats = defaultdict(list)
        pts_x, pts_y = [], []
        rank_rows = []
        for r in RANKS:
            n = r // 8
            recs = data[arm][r]["recs"]
            for g in data[arm][r]["gemm"]:
                if g["ag_k"] is None:
                    cat = "no AG-I overlap"
                else:
                    x = recs[g["ag_k"]]
                    lag = (maxstart[arm][(n, x["k"])] - x["cstart"]) / NS
                    cat = "overlapping AG-I early" if lag >= LAG_MS else "overlapping AG-I last or near-last"
                cats[cat].append(g)
            fwd = [x for x in recs if x["phase"] in PHASES_FWD]
            lag_ag = [x for x in fwd if (maxstart[arm][(n, x["k"])] - x["cstart"]) / NS >= LAG_MS]
            last_share = sum(1 for x in fwd if last_of[(arm, n, x["k"])] == r) / len(fwd)
            slow = sum(1 for g in data[arm][r]["gemm"] if g["dur"] > SLOW_MS)
            rank_rows.append((r, f"node {n}", f"{slow / n_mb:.2f}", f"{len(data[arm][r]['gemm']) / n_mb:.1f}",
                              f"{len(lag_ag) / n_mb:.2f}", f"{last_share:.3f}",
                              fmt(med([x["dur"] for x in fwd]))))
            for step, ph in sorted({(x["step"], x["phase"]) for x in fwd}):
                pts_x.append(sum(1 for x in lag_ag if x["step"] == step and x["phase"] == ph))
                pts_y.append(sum(1 for g in data[arm][r]["gemm"]
                                 if g["dur"] > SLOW_MS and g["step"] == step and g["phase"] == ph))
        per_rank_tables[arm] = rank_rows
        for cat in ["overlapping AG-I early", "overlapping AG-I last or near-last", "no AG-I overlap"]:
            gs = cats[cat]
            slow = [g for g in gs if g["dur"] > SLOW_MS]
            cross_rows.append((arm, cat, f"{len(gs) / (16 * n_mb):.2f}", f"{len(slow) / (16 * n_mb):.2f}",
                               f"{len(slow) / len(gs):.3f}" if gs else "n/a",
                               fmt(med([g["dur"] for g in gs])),
                               fmt(med([g["dur"] for g in slow]))))
        r_lines.append((arm, len(pts_x), fmt(pearson(pts_x, pts_y), 3)))
    for arm, _ in ARMS:
        L.append(f"### {arm}: per rank, per microbatch\n")
        L += md(["rank", "node", "slow GEMM per mb", "GEMM per mb", "early AG-I per mb",
                 "rank was last starter (share of its AG-I)", "AG-I median dur (ms)"], per_rank_tables[arm])
    L.append("### GEMM by the overlapping AG-I on the same rank (16 ranks, per microbatch)\n")
    L += md(["arm", "GEMM category", "GEMM per mb", "slow per mb", "slow fraction", "GEMM median dur (ms)",
             "slow GEMM median dur (ms)"], cross_rows)
    L.append("Pearson r over (rank, step, mb) points, x = early AG-I count and y = slow GEMM count:\n")
    L += md(["arm", "points", "Pearson r"], r_lines)
    return L


def sec6(data, offs):
    L = ["## 6. Timeline excerpt: node 1, step index 1, mb1 forward, first 6 AG-I (corrected times)", ""]
    L.append("Step index 1 is the second profiled step (NVTX `train_T4096_ga4_step4` in these files; the first is "
             "`step3`). t0 is the earliest corrected start of the first AG-I of the excerpt across the 8 ranks. "
             "Launch is the CUPTI runtime-call start. `running at start` is the compute-stream op on that rank at the "
             "AG-I start.\n")
    rks = [8 + i for i in range(8)]
    for arm, _ in ARMS:
        recs0 = data[arm][rks[0]]["recs"]
        k0 = next(x["k"] for x in recs0 if x["step"] == 1 and x["phase"] == "mb1_fwd" and x["idx"] == 0)
        t0 = min(data[arm][r]["recs"][k0]["cstart"] for r in rks)
        rows = []
        for i in range(6):
            k = k0 + i
            tags = {(data[arm][r]["recs"][k]["step"], data[arm][r]["recs"][k]["phase"]) for r in rks}
            if len(tags) != 1:
                raise SystemExit(f"tags differ across ranks at k={k}: {tags}")
            for r in rks:
                x = data[arm][r]["recs"][k]
                rows.append((i + 1, r, f"{(x['claunch'] - t0) / NS:+.3f}", f"{(x['cstart'] - t0) / NS:+.3f}",
                             f"{(x['cend'] - t0) / NS:+.3f}", f"{x['dur']:.3f}",
                             x["run"].replace(E.GEMM_PREFIX, "")))
        L.append(f"### {arm} (collective indices k = {k0}..{k0 + 5} on each rank)\n")
        L += md(["AG-I #", "rank", "launch (ms)", "start (ms)", "end (ms)", "dur (ms)", "running at start"], rows)
    return L


def anomalies(data, meta, insts, counts, tag_mis):
    L = ["## Anomalies and checks", ""]
    for arm, _ in ARMS:
        for n in NODES:
            c = counts[arm][n]
            if len(set(c.values())) > 1:
                L.append(f"- {arm} node {n}: AG-I count mismatch across ranks {c}; matched the first {min(c.values())}.")
            else:
                L.append(f"- {arm} node {n}: AG-I count {next(iter(c.values()))} on each of the 8 ranks; no mismatch.")
    L.append(f"- Collectives whose 8 ranks disagree on (step, phase): {sum(tag_mis[a] for a, _ in ARMS)}.")
    for arm, _ in ARMS:
        inner = sorted({meta[arm][r]["inner"] for r in RANKS})
        outer = sorted({str(meta[arm][r]["outer"]) for r in RANKS})
        L.append(f"- {arm}: inner AG (stream, gridX) over the 16 files: {inner}. Outer AG: {outer}.")
        L.append(f"- {arm}: unmatched ops {sum(meta[arm][r]['unmatched'] for r in RANKS)}; multi-phase matches "
                 f"{sum(meta[arm][r]['multi'] for r in RANKS)}; duplicate runtime correlation ids "
                 f"{sum(meta[arm][r]['rt_dup'] for r in RANKS)}; duplicate event records "
                 f"{sum(meta[arm][r]['rec_dup'] for r in RANKS)}; cudaStreamWaitEvent rows with another name "
                 f"{sum(meta[arm][r]['wait_bad'] for r in RANKS)}; train NVTX ranges per file "
                 f"{sorted({meta[arm][r]['n_steps'] for r in RANKS})}.")
        outside = sum(1 for r in RANKS for x in data[arm][r]["recs"] if x["phase"] == "outside")
        bad_step = sum(1 for r in RANKS for x in data[arm][r]["recs"] if x["step"] < 0)
        unc = sum(1 for x in sel(insts[arm], PHASES_FWD) if x["last_rec"]["cls"] == "u")
        unres = sum(1 for r in RANKS for x in data[arm][r]["recs"]
                    if x["info"] is not None and x["info"]["cat"] in ("record not in trace", "recording stream had no earlier op"))
        after = sum(1 for r in RANKS for x in data[arm][r]["recs"]
                    if x["info"] is not None and x["info"]["cat"] == "event completes after AG start")
        L.append(f"- {arm}: AG-I outside any phase {outside}; outside any step {bad_step}; last-starter "
                 f"unclassifiable {unc} (mb1..3 forward); waited event completes after the AG start {after}; "
                 f"AG-I whose waited event has no gating op (record not in trace, or no earlier op on the recording stream) {unres} (all phases).")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    data, meta, offs, insts, maxstart, counts, tag_mis = {}, {}, {}, {}, {}, {}, {}
    for arm, adir in ARMS:
        data[arm], meta[arm] = load_arm(adir)
        offs[arm] = apply_offsets({r: data[arm][r]["recs"] for r in RANKS})
        insts[arm], maxstart[arm], counts[arm], tag_mis[arm] = build_instances(data[arm])
        print(f"[instances] {arm}: {len(insts[arm])} collectives", flush=True)

    L = ["# E2 skew: inner all-gather spin versus late arrival, 2-node nsys", ""]
    L.append("Script: `bench/e2skew_compare.py`. Inputs: nsys sqlite exports of `d4096_L32_T4096_ga4.rank{0..15}` "
             "for `01-v2-off` (fast) and `02-v2-p1` (slow). Inner all-gather (AG-I) = NCCL AllGather on the inner "
             "stream with gridX 24. Phase tags come from CPU-launch membership inside the NVTX ranges. Cross-rank "
             "comparisons use per-rank clock offsets estimated from AG-I end times (section 1). Node 0 and node 1 are "
             "never compared.\n")
    L += sec1(data, meta, offs, insts)
    L += sec2(insts)
    L += sec3(insts)
    L += sec4(insts)
    L += sec5(data, insts, maxstart)
    L += sec6(data, offs)
    L.append("## 7. Findings\n")
    L.append(FINDINGS.strip() if FINDINGS.strip() else "(not written yet)")
    L.append("")
    L += anomalies(data, meta, insts, counts, tag_mis)
    text = "\n".join(L)
    print(text)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
