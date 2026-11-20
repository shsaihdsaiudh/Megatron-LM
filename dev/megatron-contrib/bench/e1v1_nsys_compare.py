#!/usr/bin/env python3
"""Compare the forward timeline of microbatches 1..3 across Nsight Systems traces.

Adapted from bench/e2_fwd_slowdown.py. Stream ids, gridX values and the compute stream are detected
per file. Nothing is hardcoded except the NVTX range names.

Inputs are sqlite exports (`nsys export --type sqlite`). Each input is given as
  --file ARM@GROUP=PATH
for example --file v2-off@rank8=/path/rank8.sqlite. Arms are compared inside the same GROUP against
--baseline (default: the first arm of the group). Groups are reported in the order rank8, then others.

Usage:
  python3 e1v1_nsys_compare.py --file v2-off@rank8=A.sqlite --file v2-p1@rank8=B.sqlite \
      --file v1-same@rank8=C.sqlite --file v2-off@rank0=D.sqlite ... --out NSYS.md [--findings FILE]

Definitions
- Phases are the NVTX ranges mb{i}_fwd and mb{i}_bwd. Each occurs once per profiled step (2 per file).
  Per-microbatch averages divide totals by the number of instances in the group: mb0 fwd 2,
  mb1..3 fwd 6, mb*_bwd 8.
- Membership (primary, used for all numbers in the report): an op belongs to a phase instance when its
  CPU launch time (CUPTI runtime call start joined by correlationId) lies inside the NVTX range. Forward
  ranges also require the same CPU thread (globalTid). Backward ranges match by time only, because
  backward kernels are launched from another thread (same as e2_fwd_slowdown.py).
- Membership (secondary, check only): the GPU window of an instance is [earliest GPU start, latest GPU end]
  of its primary members. Any op whose GPU start falls inside the window is a member.
- AG-I (inner all-gather): the NCCL AllGather family (stream, gridX, shortName) with the most kernels in
  mb1..3 forward. Outer AG: an AllGather family that appears in mb0 forward and never in mb1..3 forward.
- Compute stream: the stream with the most nvjet kernels. Compute ops are all ops on that stream
  (kernels and memsets). GEMM = nvjet kernel on the compute stream.
- Overlap: a compute-stream op is overlapped if any NCCL kernel on another stream overlaps it by a
  positive time. The AG-I start is attributed to the compute-stream op running at that instant. Overlap
  uses every NCCL and compute op in the file, not only the instance.
- Span: first compute-stream op start to last compute-stream op end, per instance. Busy: union of
  compute-stream op intervals. Idle: span minus busy. A gap is a stretch inside the span with no
  compute-stream op running.
"""

import argparse
import bisect
import sqlite3
import statistics
from collections import Counter, defaultdict

NS = 1e6
FWD_MB = ["mb1_fwd", "mb2_fwd", "mb3_fwd"]
BWD_ALL = ["mb0_bwd", "mb1_bwd", "mb2_bwd", "mb3_bwd"]
PHASE_NAMES = ["mb0_fwd"] + FWD_MB + BWD_ALL
GROUP_PHASES = {"mb0_fwd": ["mb0_fwd"], "mb1..3_fwd": FWD_MB, "mb*_bwd": BWD_ALL}
GEMM_PREFIX = "nvjet_sm90_tst_"


class Op:
    def __init__(self, kind, stream, start, end, corr, short, grid):
        self.idx = -1
        self.kind = kind            # KERNEL, MEMCPY or MEMSET
        self.stream = stream
        self.start = start
        self.end = end
        self.corr = corr
        self.short = short
        self.grid = grid
        self.launch = None
        self.tid = None
        self.cpu = None             # (phase, instance index) from primary membership

    @property
    def dur(self):
        return (self.end - self.start) / NS

    @property
    def is_nccl(self):
        return self.kind == "KERNEL" and "nccl" in self.short.lower()

    @property
    def is_gemm(self):
        return self.kind == "KERNEL" and "nvjet" in self.short


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def f2(x):
    return "n/a" if x != x else f"{x:.2f}"


def disp(o):
    if o.kind != "KERNEL":
        return o.kind.lower()
    return o.short.replace(GEMM_PREFIX, "")


def kname(o):
    """Kernel name plus gridX, used as the grouping key for compute ops."""
    if o.kind != "KERNEL":
        return disp(o)
    return f"{disp(o)} [gx{o.grid}]"


def phase_group(ph):
    if ph == "mb0_fwd":
        return "mb0_fwd"
    if ph in FWD_MB:
        return "mb1..3_fwd"
    return "mb*_bwd"


def merge(iv):
    ms, me = [], []
    for s, e in iv:
        if me and s <= me[-1]:
            me[-1] = max(me[-1], e)
        else:
            ms.append(s)
            me.append(e)
    return ms, me


def ov_merged(ms, me, a, b):
    tot = 0
    i = bisect.bisect_right(me, a)
    while i < len(ms) and ms[i] < b:
        tot += min(b, me[i]) - max(a, ms[i])
        i += 1
    return tot


def ov_compute(D, a, b):
    """(op, overlap_ns) for compute-stream ops that overlap [a, b)."""
    cops, cstarts = D["cops"], D["cstarts"]
    i = max(0, bisect.bisect_right(cstarts, a) - 1)
    out = []
    while i < len(cops) and cops[i].start < b:
        x = min(b, cops[i].end) - max(a, cops[i].start)
        if x > 0:
            out.append((cops[i], x))
        i += 1
    return out


def running_op(D, a):
    i = bisect.bisect_right(D["cstarts"], a) - 1
    if i >= 0 and D["cops"][i].end > a:
        return D["cops"][i]
    return None


def fwd_keys(D):
    return [(ph, k) for ph in FWD_MB for k in range(D["n_inst"][ph])]


def fwd_ops(D):
    return [o for key in fwd_keys(D) for o in D["members"].get(key, [])]


def is_inner(D, o):
    return D["inner_key"] is not None and o.kind == "KERNEL" and (o.stream, o.grid, o.short) == D["inner_key"]


def load(path):
    c = sqlite3.connect(path)
    S = dict(c.execute("select id, value from StringIds"))

    nvtx = defaultdict(list)
    for text, text_id, s, e, g in c.execute(
        "select text, textId, start, end, globalTid from NVTX_EVENTS where end is not null"
    ):
        lab = text if text is not None else S.get(text_id)
        if lab in PHASE_NAMES:
            nvtx[lab].append((s, e, g))
    for v in nvtx.values():
        v.sort()

    rt = {}
    for corr, s, g in c.execute("select correlationId, start, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        rt[corr] = (s, g)

    ops = []
    for sid, s, e, corr, sn, gx in c.execute(
        "select streamId, start, end, correlationId, shortName, gridX from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        ops.append(Op("KERNEL", sid, s, e, corr, S.get(sn, ""), gx))
    for s, e, sid, corr in c.execute("select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ops.append(Op("MEMCPY", sid, s, e, corr, "memcpy", None))
    for s, e, sid, corr in c.execute("select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"):
        ops.append(Op("MEMSET", sid, s, e, corr, "memset", None))
    ops.sort(key=lambda o: o.start)
    for i, o in enumerate(ops):
        o.idx = i

    unmatched = 0
    for o in ops:
        r = rt.get(o.corr)
        if r is None:
            unmatched += 1
        else:
            o.launch, o.tid = r

    gemm_streams = Counter(o.stream for o in ops if o.is_gemm)
    compute = gemm_streams.most_common(1)[0][0]

    # Primary membership: CPU launch time inside the NVTX range (forward also requires the same thread).
    n_inst = {ph: len(nvtx.get(ph, [])) for ph in PHASE_NAMES}
    starts = {ph: [t[0] for t in nvtx.get(ph, [])] for ph in PHASE_NAMES}
    thread_match, any_thread, multi = Counter(), Counter(), 0
    members = defaultdict(list)
    for o in ops:
        if o.launch is None:
            continue
        hits = []
        for ph in PHASE_NAMES:
            L = nvtx.get(ph)
            if not L:
                continue
            k = bisect.bisect_right(starts[ph], o.launch) - 1
            if k < 0:
                continue
            s, e, g = L[k]
            if o.launch > e:
                continue
            any_thread[ph] += 1
            if g == o.tid:
                thread_match[ph] += 1
            if ph.endswith("_bwd") or g == o.tid:
                hits.append((ph, k))
        if len(hits) > 1:
            multi += 1
        if hits:
            o.cpu = hits[0]
            members[hits[0]].append(o)

    # Secondary membership check: GPU window of each instance.
    gstarts = [o.start for o in ops]
    gpu_members = {}
    for key, lst in members.items():
        lo = min(o.start for o in lst)
        hi = max(o.end for o in lst)
        a = bisect.bisect_left(gstarts, lo)
        b = bisect.bisect_right(gstarts, hi)
        gpu_members[key] = ops[a:b]

    # NCCL families by phase group.
    fam = defaultdict(Counter)
    for o in ops:
        if not o.is_nccl:
            continue
        g = phase_group(o.cpu[0]) if o.cpu else "outside"
        fam[(o.stream, o.grid, o.short)][g] += 1

    ag_keys = [k for k in fam if "AllGather" in k[2]]
    fwd_ag = [k for k in ag_keys if fam[k]["mb1..3_fwd"] > 0]
    inner_key = max(fwd_ag, key=lambda k: fam[k]["mb1..3_fwd"]) if fwd_ag else None
    outer_cands = [k for k in ag_keys if fam[k]["mb0_fwd"] > 0 and fam[k]["mb1..3_fwd"] == 0]
    outer_key = max(outer_cands, key=lambda k: fam[k]["mb0_fwd"]) if outer_cands else None

    cops = [o for o in ops if o.stream == compute]
    cstarts = [o.start for o in cops]
    overlapping_compute = sum(1 for a, b in zip(cops, cops[1:]) if b.start < a.end)
    nccl_on_compute = sum(1 for o in cops if o.is_nccl)
    ms, me = merge(sorted((o.start, o.end) for o in ops if o.is_nccl and o.stream != compute))

    return {
        "name": path.split("/")[-1],
        "ops": ops,
        "compute": compute,
        "gemm_streams": gemm_streams,
        "n_inst": n_inst,
        "members": members,
        "gpu_members": gpu_members,
        "thread_match": thread_match,
        "any_thread": any_thread,
        "multi": multi,
        "unmatched": unmatched,
        "fam": fam,
        "inner_key": inner_key,
        "outer_key": outer_key,
        "outer_cands": outer_cands,
        "cops": cops,
        "cstarts": cstarts,
        "overlapping_compute": overlapping_compute,
        "nccl_on_compute": nccl_on_compute,
        "nccl_ms": ms,
        "nccl_me": me,
    }


def ovl_nccl(D, o):
    return ov_merged(D["nccl_ms"], D["nccl_me"], o.start, o.end) > 0


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")
    return out


def busy_and_gaps(ops_i):
    """Union busy time and the gaps between compute-stream ops (ns). ops_i is sorted by start."""
    busy = 0
    gaps = []
    cover_end = None
    cover_op = None
    for o in ops_i:
        if cover_end is None:
            cover_end, cover_op = o.end, o
            busy += o.end - o.start
        elif o.start >= cover_end:
            if o.start > cover_end:
                gaps.append((o.start - cover_end, cover_op, o))
            busy += o.end - o.start
            cover_end, cover_op = o.end, o
        elif o.end > cover_end:
            busy += o.end - cover_end
            cover_end, cover_op = o.end, o
    return busy, gaps


def span_section(arms, baseline):
    L = []
    cols = {}
    gap_rows = {}
    for arm, D in arms.items():
        per = []
        allgaps = []
        for key in fwd_keys(D):
            ops_i = sorted((o for o in D["members"].get(key, []) if o.stream == D["compute"]), key=lambda o: o.start)
            if not ops_i:
                continue
            first = ops_i[0].start
            last = max(o.end for o in ops_i)
            busy, gaps = busy_and_gaps(ops_i)
            span = (last - first) / NS
            per.append({
                "n": len(ops_i),
                "span": span,
                "busy": busy / NS,
                "idle": span - busy / NS,
                "sum": sum(o.dur for o in ops_i),
                "gaps": len(gaps),
            })
            for g, prev, nxt in gaps:
                allgaps.append((g / NS, f"{key[0]}#{key[1]}", disp(prev), disp(nxt)))
        n = len(per)

        def mean(k):
            return sum(p[k] for p in per) / n if n else float("nan")

        cols[arm] = {
            "instances": n,
            "ops": mean("n"),
            "span": mean("span"),
            "busy": mean("busy"),
            "idle": mean("idle"),
            "idle_share": mean("idle") / mean("span") if n else float("nan"),
            "sum": mean("sum"),
            "gaps": mean("gaps"),
        }
        allgaps.sort(key=lambda x: -x[0])
        gap_rows[arm] = allgaps[:5]
    arm_names = list(arms)
    rows = [
        ("Compute-stream ops per microbatch", "ops", "{:.1f}"),
        ("GPU span (first op start to last op end), ms", "span", "{:.3f}"),
        ("Busy (union of op intervals), ms", "busy", "{:.3f}"),
        ("Idle inside span (span minus busy), ms", "idle", "{:.3f}"),
        ("Idle share of span", "idle_share", "{:.4f}"),
        ("Sum of op durations, ms", "sum", "{:.3f}"),
        ("Idle gaps per microbatch (count)", "gaps", "{:.1f}"),
        ("Instances used", "instances", "{}"),
    ]
    table_rows = []
    for label_, key, fmt in rows:
        table_rows.append([label_] + [fmt.format(cols[a][key]) for a in arm_names])
    L += md_table(["metric (mean per microbatch instance)"] + arm_names, table_rows)
    for arm in arm_names:
        L.append(f"Top 5 idle gaps, {arm} (ms, instance, previous op, next op):\n")
        L += md_table(["gap ms", "instance", "previous op", "next op"],
                      [(f"{g:.3f}", i, p, n) for g, i, p, n in gap_rows[arm]] or [("none", "", "", "")])
    return L


def gemm_section(arms, baseline):
    L = []
    for arm, D in arms.items():
        keys = fwd_keys(D)
        n = len(keys)
        comp = [o for o in fwd_ops(D) if o.stream == D["compute"]]
        gemm = [o for o in comp if o.is_gemm]
        other = [o for o in comp if not o.is_gemm]
        L.append(f"### {arm}\n")
        by = defaultdict(list)
        for o in comp:
            by[kname(o)].append(o)
        top = sorted(by.items(), key=lambda kv: -sum(o.dur for o in kv[1]))[:10]
        L.append("Top 10 compute-stream kernel names by total time (all compute-stream ops, per microbatch):\n")
        L += md_table(
            ["kernel", "count per mb", "total ms per mb", "median ms"],
            [(k, f"{len(v) / n:.2f}", f2(sum(o.dur for o in v) / n), f3(med([o.dur for o in v])))
             for k, v in top],
        )
        gf = defaultdict(list)
        for o in gemm:
            gf[kname(o)].append(o)
        gfam = sorted(gf.items(), key=lambda kv: -sum(o.dur for o in kv[1]))
        split_rows = []
        for k, v in gfam:
            ov = [o for o in v if ovl_nccl(D, o)]
            nov = [o for o in v if not ovl_nccl(D, o)]
            split_rows.append((k, f"{len(ov) / n:.2f}", f3(med([o.dur for o in ov])),
                               f2(sum(o.dur for o in ov) / n),
                               f"{len(nov) / n:.2f}", f3(med([o.dur for o in nov])),
                               f2(sum(o.dur for o in nov) / n)))
        L.append("GEMM families on the compute stream, split by whether any NCCL kernel on another stream overlaps them:\n")
        L += md_table(["GEMM kernel", "overlapped count per mb", "overlapped median ms", "overlapped total ms per mb",
                       "not overlapped count per mb", "not overlapped median ms", "not overlapped total ms per mb"],
                      split_rows)
        ov_o = [o for o in other if ovl_nccl(D, o)]
        nov_o = [o for o in other if not ovl_nccl(D, o)]
        ov_g = [o for o in gemm if ovl_nccl(D, o)]
        nov_g = [o for o in gemm if not ovl_nccl(D, o)]
        L.append("Aggregates on the compute stream (per microbatch):\n")
        L += md_table(
            ["group", "count per mb", "total ms per mb", "overlapped count per mb", "overlapped median ms",
             "overlapped total ms per mb", "not overlapped count per mb", "not overlapped median ms",
             "not overlapped total ms per mb"],
            [
                ("all GEMM", f"{len(gemm) / n:.2f}", f2(sum(o.dur for o in gemm) / n),
                 f"{len(ov_g) / n:.2f}", f3(med([o.dur for o in ov_g])), f2(sum(o.dur for o in ov_g) / n),
                 f"{len(nov_g) / n:.2f}", f3(med([o.dur for o in nov_g])), f2(sum(o.dur for o in nov_g) / n)),
                ("non-GEMM compute ops", f"{len(other) / n:.2f}", f2(sum(o.dur for o in other) / n),
                 f"{len(ov_o) / n:.2f}", f3(med([o.dur for o in ov_o])), f2(sum(o.dur for o in ov_o) / n),
                 f"{len(nov_o) / n:.2f}", f3(med([o.dur for o in nov_o])), f2(sum(o.dur for o in nov_o) / n)),
            ],
        )
        gd = sorted(o.dur for o in gemm)

        def pc(q):
            return gd[min(len(gd) - 1, int(q * (len(gd) - 1)))]

        L += md_table(
            ["GEMM duration (all GEMM on compute stream)", "mean ms", "p50 ms", "p90 ms", "p99 ms", "max ms"],
            [("all GEMM", f3(sum(gd) / len(gd)), f3(pc(0.5)), f3(pc(0.9)), f3(pc(0.99)), f3(gd[-1]))],
        )
        L.append("GEMMs split at 4.5 ms (a fixed cut between the two duration clusters in the data), crossed with "
                 "NCCL overlap:\n")
        cells = {}
        for cls, pred in (("<= 4.5 ms", lambda d: d <= 4.5), ("> 4.5 ms", lambda d: d > 4.5)):
            sel = [o for o in gemm if pred(o.dur)]
            ov = [o for o in sel if ovl_nccl(D, o)]
            nov = [o for o in sel if not ovl_nccl(D, o)]
            cells[cls] = (len(ov), len(nov), sum(o.dur for o in ov), sum(o.dur for o in nov))
        L += md_table(
            ["GEMM class", "overlapped count per mb", "overlapped total ms per mb", "overlapped mean ms",
             "not overlapped count per mb", "not overlapped total ms per mb", "not overlapped mean ms"],
            [
                (cls, f"{a / n:.2f}", f2(s_a / n), f3(s_a / a) if a else "n/a",
                 f"{b / n:.2f}", f2(s_b / n), f3(s_b / b) if b else "n/a")
                for cls, (a, b, s_a, s_b) in cells.items()
            ],
        )
    return L


def ag_section(arms):
    L = []
    for arm, D in arms.items():
        keys = fwd_keys(D)
        n = len(keys)
        ag = [o for o in fwd_ops(D) if is_inner(D, o)]
        L.append(f"### {arm}\n")
        if not ag:
            L.append("No AG-I kernels found in mb1..3 forward.\n")
            continue
        run = Counter()
        gemm_ov = 0
        any_ov = 0
        total = 0
        for o in ag:
            r = running_op(D, o.start)
            run[kname(r) if r is not None else "(compute stream idle)"] += 1
            for op, x in ov_compute(D, o.start, o.end):
                if op.is_gemm:
                    gemm_ov += x
                any_ov += x
            total += o.end - o.start
        durs = [o.dur for o in ag]
        L += md_table(
            ["metric", "value"],
            [
                ("AG-I kernels per microbatch", f"{len(ag) / n:.2f}"),
                ("AG-I median duration, ms", f3(med(durs))),
                ("AG-I total time per microbatch, ms", f2(sum(durs) / n)),
                ("Fraction of AG-I time overlapping a GEMM", f"{gemm_ov / total:.3f}"),
                ("Fraction of AG-I time overlapping any compute-stream op", f"{any_ov / total:.3f}"),
            ],
        )
        L.append("Compute op running when each AG-I starts (top 5, count over both steps and mb1..3):\n")
        L += md_table(["running op at AG-I start", "count"], run.most_common(5))
        cls = defaultdict(list)
        for o in ag:
            r = running_op(D, o.start)
            c_ = "compute stream idle" if r is None else ("GEMM" if r.is_gemm else "other compute op")
            cls[c_].append(o.dur)
        L.append("AG-I duration by what runs on the compute stream at its start:\n")
        L += md_table(
            ["class at AG-I start", "count per mb", "median ms", "total ms per mb"],
            [(c_, f"{len(v) / n:.2f}", f3(med(v)), f2(sum(v) / n)) for c_, v in sorted(cls.items())],
        )
    return L


def diff_section(arms, baseline):
    L = []
    base = arms[baseline]
    nb = len(fwd_keys(base))

    def per_key(D):
        n = len(fwd_keys(D))
        ms, cnt = defaultdict(float), Counter()
        for o in fwd_ops(D):
            if o.stream == D["compute"]:
                ms[kname(o)] += o.dur / n
                cnt[kname(o)] += 1 / n
        return ms, cnt

    bms, bcnt = per_key(base)
    for arm, D in arms.items():
        if arm == baseline:
            continue
        ams, acnt = per_key(D)
        keys = set(bms) | set(ams)
        rows = sorted(keys, key=lambda k: -abs(ams.get(k, 0) - bms.get(k, 0)))[:15]
        L.append(f"### {arm} minus {baseline} (compute-stream kernel time per microbatch)\n")
        table = []
        for k in rows:
            table.append((k, f"{bcnt.get(k, 0):.2f}", f"{acnt.get(k, 0):.2f}", f3(bms.get(k, 0)), f3(ams.get(k, 0)),
                          f"{ams.get(k, 0) - bms.get(k, 0):+.3f}"))
        tb = sum(bms.values())
        ta = sum(ams.values())
        table.append(("TOTAL compute stream (all kernels)", "", "", f2(tb), f2(ta), f"{ta - tb:+.3f}"))
        L += md_table(["kernel", f"{baseline} count/mb", f"{arm} count/mb", f"{baseline} ms/mb", f"{arm} ms/mb",
                       "difference ms/mb"], table)
    return L


def other_stream_section(arms):
    L = []
    for arm, D in arms.items():
        n = len(fwd_keys(D))
        grp = defaultdict(lambda: [0, 0.0])
        for o in fwd_ops(D):
            if o.stream == D["compute"] or o.is_nccl:
                continue
            key = (o.stream, o.kind, o.short)
            grp[key][0] += 1
            grp[key][1] += o.dur
        L.append(f"### {arm}\n")
        if not grp:
            L.append("None. Every mb1..3 forward kernel, memcpy and memset outside NCCL is on the compute stream.\n")
            continue
        L += md_table(["stream", "kind", "name", "count per mb", "total ms per mb"],
                      [(k[0], k[1], k[2], f"{v[0] / n:.2f}", f3(v[1] / n)) for k, v in sorted(grp.items())])
    return L


def detection_section(arms, baseline):
    L = []
    for arm, D in arms.items():
        L.append(f"### {arm} ({D['name']})\n")
        cnt = D["gemm_streams"]
        L.append(f"- Compute stream: {D['compute']} ({cnt[D['compute']]} nvjet kernels; "
                 f"other streams with nvjet: {dict((s, c) for s, c in cnt.items() if s != D['compute']) or 'none'}).")
        ik, ok = D["inner_key"], D["outer_key"]
        L.append(f"- AG-I: stream {ik[0]}, gridX {ik[1]}, {ik[2]}." if ik else "- AG-I: NOT FOUND.")
        if ok:
            L.append(f"- Outer AG: stream {ok[0]}, gridX {ok[1]}, {ok[2]}.")
        else:
            L.append("- Outer AG: none. No AllGather family appears in mb0 forward and not in mb1..3 forward.")
        L.append(f"- Forward CPU membership: thread-matched / any-thread launches inside the range: "
                 + ", ".join(f"{ph} {D['thread_match'][ph]}/{D['any_thread'][ph]}" for ph in ["mb0_fwd"] + FWD_MB)
                 + ".")
        L.append("")
        rows = []
        for k in sorted(D["fam"], key=lambda k: (k[0], k[1], k[2])):
            c = D["fam"][k]
            if k == ik:
                role = "AG-I"
            elif k == ok:
                role = "outer AG"
            elif "AllGather" in k[2]:
                role = "other AllGather"
            elif "ReduceScatter" in k[2]:
                role = "ReduceScatter"
            else:
                role = "other NCCL"
            rows.append((k[0], k[2].replace("ncclDevKernel_", ""), k[1], role,
                         f"{c['mb0_fwd'] / D['n_inst']['mb0_fwd']:.2f}" if D['n_inst']['mb0_fwd'] else "n/a",
                         f"{c['mb1..3_fwd'] / max(1, sum(D['n_inst'][p] for p in FWD_MB)):.2f}",
                         f"{c['mb*_bwd'] / max(1, sum(D['n_inst'][p] for p in BWD_ALL)):.2f}",
                         c["outside"], sum(c.values())))
        L += md_table(["stream", "NCCL kernel", "gridX", "role", "mb0 fwd per mb", "mb1..3 fwd per mb",
                       "mb*_bwd per mb", "outside any phase", "total"], rows)
    return L


def membership_section(arms, baseline):
    L = []
    cols = {}
    for arm, D in arms.items():
        keys = fwd_keys(D)
        n = len(keys)
        cpu = fwd_ops(D)
        gpu = [o for k in keys for o in D["gpu_members"].get(k, [])]
        both = sum(len({o.idx for o in D["members"].get(k, [])} & {o.idx for o in D["gpu_members"].get(k, [])})
                   for k in keys)

        def st(lst):
            comp = [o for o in lst if o.stream == D["compute"]]
            ag = [o for o in lst if is_inner(D, o)]
            return (f"{len(lst) / n:.1f}", f"{len(comp) / n:.1f}", f2(sum(o.dur for o in comp) / n),
                    f"{len(ag) / n:.1f}")

        cols[arm] = (st(cpu), st(gpu), f"{both / n:.1f}")
    arm_names = list(arms)
    rows = []
    labels = ["all ops per mb", "compute-stream ops per mb", "compute-stream op ms per mb", "AG-I per mb"]
    for i, lab in enumerate(labels):
        rows.append([f"CPU launch (primary): {lab}"] + [cols[a][0][i] for a in arm_names])
        rows.append([f"GPU window (secondary): {lab}"] + [cols[a][1][i] for a in arm_names])
    rows.append(["ops in both, per mb"] + [cols[a][2] for a in arm_names])
    L += md_table(["metric (mb1..3 forward, per microbatch)"] + arm_names, rows)
    return L


def anomalies(arms):
    lines = []
    for arm, D in arms.items():
        bits = []
        if D["unmatched"]:
            bits.append(f"{D['unmatched']} ops have no CUPTI runtime row")
        if D["multi"]:
            bits.append(f"{D['multi']} ops matched more than one phase (first match used)")
        if D["overlapping_compute"]:
            bits.append(f"{D['overlapping_compute']} overlapping pairs on the compute stream")
        if D["nccl_on_compute"]:
            bits.append(f"{D['nccl_on_compute']} NCCL kernels on the compute stream")
        bad = {ph: n for ph, n in D["n_inst"].items() if n != 2}
        if bad:
            bits.append(f"NVTX instance counts not equal to 2: {bad}")
        if D["outer_cands"] and len(D["outer_cands"]) > 1:
            bits.append("AllGather families appearing in mb0 forward only (candidates for outer AG): "
                        + ", ".join(f"stream {k[0]} gridX {k[1]}" for k in D["outer_cands"])
                        + f"; chosen: stream {D['outer_key'][0]} gridX {D['outer_key'][1]}")
        k = D["inner_key"]
        if k:
            per_mb = sum(1 for key in fwd_keys(D) for o in D["members"].get(key, []) if is_inner(D, o))
            per_mb /= len(fwd_keys(D))
            if abs(per_mb - 32) > 1e-9:
                bits.append(f"AG-I per mb is {per_mb:.2f}, not 32")
        lines.append(f"- {arm}: " + ("; ".join(bits) if bits else "no structural anomalies"))
    return lines


def build_report(groups, baseline_arm, findings):
    L = []
    L.append("# E1v1 Nsight Systems comparison: forward of microbatches 1..3\n")
    L.append("Script: `bench/e1v1_nsys_compare.py`. Inputs: sqlite exports of `d4096_L32_T4096_ga4.rank{0,8}.nsys-rep`"
             " for v2-off, v2-p1 and v1-same. Rank 8 is primary and rank 0 is secondary.\n")
    L.append("All per-microbatch numbers average over the 6 forward instances (mb1, mb2, mb3 in each of 2 steps)."
             " Membership is CPU launch time inside the NVTX range (primary). The GPU-window check is in section 1.\n")
    for group in groups:
        arms = groups[group]
        base = baseline_arm if baseline_arm in arms else next(iter(arms))
        title = "Rank 8 (primary)" if group == "rank8" else ("Rank 0 (secondary)" if group == "rank0" else group)
        L.append(f"## {title}\n")
        L.append("Arms in this group, baseline " + base + ": " + ", ".join(
            f"{a} = `{arms[a]['name']}`" for a in arms) + ".\n")
        L.append("### 1. Detection summary\n")
        L += detection_section(arms, base)
        L.append("Membership check (CPU launch primary vs GPU window secondary):\n")
        L += membership_section(arms, base)
        L.append("Structural checks:\n")
        L += anomalies(arms)
        L.append("")
        L.append("### 2. GPU span and idle time of mb1..3 forward\n")
        L += span_section(arms, base)
        L.append("### 3. GEMM and non-GEMM compute kernels (mb1..3 forward)\n")
        L += gemm_section(arms, base)
        L.append("### 4. AG-I timing (mb1..3 forward)\n")
        L += ag_section(arms)
        L.append("### 5. Compute-stream kernel diff vs baseline (mb1..3 forward)\n")
        L += diff_section(arms, base)
        L.append("### 6. Other streams (non-compute, non-NCCL), mb1..3 forward\n")
        L += other_stream_section(arms)
    L.append("## 7. Findings\n")
    L.append(findings.strip() if findings else "(findings not provided)")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", required=True, metavar="ARM@GROUP=PATH")
    ap.add_argument("--baseline", default="v2-off")
    ap.add_argument("--findings", default="", help="text file with the findings paragraph (section 7)")
    ap.add_argument("--out", default="", help="write the Markdown report here")
    args = ap.parse_args()

    groups = defaultdict(dict)
    for item in args.file:
        label_, path = item.split("=", 1)
        arm, group = label_.split("@", 1) if "@" in label_ else (label_, "all")
        D = load(path)
        D["name"] = path
        groups[group][arm] = D
        print(f"[detect] {arm}@{group}: compute stream {D['compute']}, "
              f"inner AG {D['inner_key']}, outer AG {D['outer_key']}, "
              f"instances {D['n_inst']}, unmatched {D['unmatched']}, multi {D['multi']}")

    order = sorted(groups, key=lambda g: (g != "rank8", g != "rank0", g))
    ordered = {g: groups[g] for g in order}
    findings = open(args.findings).read() if args.findings else ""
    text = build_report(ordered, args.baseline, findings)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
