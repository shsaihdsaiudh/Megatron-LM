#!/usr/bin/env python3
"""E2 stream analysis: Megatron-FSDP DP-outer (O) vs DP-inner (I) all-gathers, flag OFF vs ON.

Usage: python3 e2_nsys_streams.py OFF.sqlite ON.sqlite      (prints Markdown)
Export first: nsys export --type sqlite --force-overwrite true -o X.sqlite X.nsys-rep

Method (times from the Nsight Systems GPU timeline, reported in ms):
- Each file has two profiled steps (NVTX train_*_step3 and step4). Each step
  contains mb0..mb3 forward/backward ranges and an optimizer range. Per-step values
  are averaged over the steps.
- Membership: a GPU kernel belongs to an NVTX range when its launching runtime call
  (CUPTI_ACTIVITY_KIND_RUNTIME, joined on correlationId) starts inside the range.
  NVTX ranges are CPU-side. The GPU runs well behind them (about 150 ms behind at
  T=1024), so NVTX durations are not GPU time. Use the "GPU span" for GPU time.
- O vs I for all-gather (and reduce-scatter): NCCL kernels with gridX 4 are the
  2-rank outer communicator, gridX 24 the 8-rank inner communicator. gridX is the
  channel count NCCL picks per communicator. The data supports this: gridX-4
  all-gathers occur only in mb0_fwd (interleaved with gridX-24), and mb1..3 forward
  and all backward ranges contain only gridX-24 all-gathers.
- gridX 1 kernels are tiny root-level unshards. Their class is decided by ordinal:
  the k-th gridX-1 kernel of a collective (in GPU launch order) is O when, in the ON
  file, that kernel runs on the outer stream (the stream carrying gridX-4 kernels).
  The ON file sets the ordinals, and the OFF file reuses them.
- GEMM = kernels whose name contains "nvjet". splitKreduce is not counted.
- NCCL overlap uses all NCCL kernels in the file (any stream) on the GPU timeline.
- Compute stream = stream carrying the most nvjet kernels.
- GPU span of a range = first kernel start to last kernel end of its kernels.
"""

import argparse
import bisect
import os
import sqlite3
import statistics
from collections import defaultdict

GAP_EPS_MS = 0.05
NS_PER_MS = 1e6


class Kern:
    __slots__ = ("stream", "start", "end", "grid", "name", "launch", "type", "comm")

    def __init__(self, stream, start, end, grid, name, launch):
        self.stream, self.start, self.end = stream, start, end
        self.grid, self.name, self.launch = grid, name, launch
        if name.startswith("ncclDevKernel_AllGather"):
            self.type = "AG"
        elif name.startswith("ncclDevKernel_ReduceScatter"):
            self.type = "RS"
        elif "nvjet" in name:
            self.type = "GEMM"
        else:
            self.type = "other"
        self.comm = None

    @property
    def dur(self):
        return (self.end - self.start) / NS_PER_MS

    @property
    def cls(self):
        if self.type in ("AG", "RS"):
            return f"{self.type}-{self.comm}"
        return self.type


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


class Union:
    """Merged, sorted, disjoint intervals with overlap queries."""

    def __init__(self, iv):
        self.m = merge(iv)
        self.s = [x[0] for x in self.m]
        self.e = [x[1] for x in self.m]

    def overlap(self, s, e):
        i = bisect.bisect_right(self.e, s)
        tot = 0
        while i < len(self.s) and self.s[i] < e:
            tot += min(e, self.e[i]) - max(s, self.s[i])
            i += 1
        return tot

    def overlap_union(self, other):
        return sum(other.overlap(s, e) for s, e in self.m)


def load(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    nv = []
    for s, e, t, tid in c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"):
        lab = t if t is not None else names.get(tid)
        if not lab:
            continue
        if (lab.startswith("mb") and (lab.endswith("_fwd") or lab.endswith("_bwd"))) or lab.startswith("train_"):
            nv.append((s, e, lab))
    kerns = []
    for sid, s, e, gx, nm, ls in c.execute(
        "select k.streamId, k.start, k.end, k.gridX, k.shortName, r.start "
        "from CUPTI_ACTIVITY_KIND_KERNEL k left join CUPTI_ACTIVITY_KIND_RUNTIME r "
        "on k.correlationId = r.correlationId"
    ):
        kerns.append(Kern(sid, s, e, gx, names.get(nm, ""), ls))
    return nv, kerns


def assign_comm(kerns, ref=None):
    """Set Kern.comm for AG and RS kernels. Returns the gridX-1 ordinal sets used."""
    used = {}
    for typ in ("AG", "RS"):
        ks = sorted([k for k in kerns if k.type == typ], key=lambda k: k.start)
        g1 = [k for k in ks if k.grid == 1]
        if ref is None:
            outer_streams = {k.stream for k in ks if k.grid == 4}
            ords = {i for i, k in enumerate(g1) if k.stream in outer_streams}
        else:
            ords = ref[typ]
        used[typ] = ords
        for k in ks:
            if k.grid == 4:
                k.comm = "O"
            elif k.grid == 24:
                k.comm = "I"
            elif k.grid == 1:
                k.comm = "O" if g1.index(k) in ords else "I"
            else:
                raise SystemExit(f"unexpected gridX {k.grid} for {typ}")
    return used


def build(path, ref=None):
    nv, kerns = load(path)
    used = assign_comm(kerns, ref)
    nvjet_n = defaultdict(int)
    for k in kerns:
        if k.type == "GEMM":
            nvjet_n[k.stream] += 1
    compute = max(nvjet_n, key=nvjet_n.get)
    steps = []
    for tlo, thi, tl in sorted([x for x in nv if x[2].startswith("train_")]):
        d = {"lo": tlo, "hi": thi, "label": tl, "fwd": {}, "bwd": {}}
        for s, e, lab in nv:
            if not (tlo <= s and e <= thi):
                continue
            if lab.startswith("mb") and lab.endswith("_fwd"):
                d["fwd"][int(lab[2:-4])] = (s, e)
            elif lab.startswith("mb") and lab.endswith("_bwd"):
                d["bwd"][int(lab[2:-4])] = (s, e)
        steps.append(d)
    launched = sorted([k for k in kerns if k.launch is not None], key=lambda k: k.launch)
    return {
        "name": os.path.basename(path),
        "kerns": kerns,
        "steps": steps,
        "compute": compute,
        "launched": launched,
        "launch_t": [k.launch for k in launched],
        "nccl_union": Union([(k.start, k.end) for k in kerns if k.type in ("AG", "RS")]),
        "comm_ref": used,
    }


def in_range(D, lo, hi):
    a = bisect.bisect_left(D["launch_t"], lo)
    b = bisect.bisect_right(D["launch_t"], hi)
    return D["launched"][a:b]


def gpu_span(ks):
    if not ks:
        return None
    return min(k.start for k in ks), max(k.end for k in ks)


def fmean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmed(xs):
    return statistics.median(xs) if xs else float("nan")


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def f2(x):
    return "n/a" if x != x else f"{x:.2f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def bucket_of(frac):
    if frac == 0:
        return "0%"
    if frac < 0.5:
        return "<50%"
    if frac < 0.999:
        return "50-99%"
    return "100%"


def validation_rows(D, tag):
    rows = []
    for label, kind, idx in [
        ("mb0_fwd", "fwd", 0),
        ("mb1_fwd", "fwd", 1),
        ("mb3_fwd", "fwd", 3),
        ("mb0_bwd", "bwd", 0),
        ("mb3_bwd", "bwd", 3),
    ]:
        per_step = []
        for s in D["steps"]:
            lo, hi = s[kind][idx]
            c = defaultdict(int)
            for k in in_range(D, lo, hi):
                if k.type == "AG":
                    c[f"AG-{k.comm}@s{k.stream}"] += 1
            per_step.append(c)
        keys = sorted({k for c in per_step for k in c})
        desc = ", ".join(f"{k}={fmean([c.get(k, 0) for c in per_step]):.0f}" for k in keys) or "none"
        rows.append((tag, label, desc))
    return rows


def q1_streams(D):
    per = []
    for s in D["steps"]:
        agg = defaultdict(lambda: [0, 0.0])
        for k in in_range(D, *s["fwd"][0]):
            if k.type == "AG":
                agg[(k.stream, k.comm)][0] += 1
                agg[(k.stream, k.comm)][1] += k.dur
        per.append(agg)
    keys = sorted({k for p in per for k in p})
    rows = []
    for key in keys:
        n = fmean([p[key][0] if key in p else 0 for p in per])
        t = fmean([p[key][1] if key in p else 0.0 for p in per])
        rows.append((f"stream {key[0]}", key[1], f"{n:.1f}", f2(t)))
    return rows


def q2_overlap(D):
    res = []
    for s in D["steps"]:
        lo, hi = s["fwd"][0]
        ks = in_range(D, lo, hi)
        O = [k for k in ks if k.type == "AG" and k.comm == "O"]
        I = [k for k in ks if k.type == "AG" and k.comm == "I"]
        UO = Union([(k.start, k.end) for k in O])
        UI = Union([(k.start, k.end) for k in I])
        conc = UO.overlap_union(UI) / NS_PER_MS
        o_total = sum(k.dur for k in O)
        i13 = []
        for m in (1, 2, 3):
            mlo, mhi = s["fwd"][m]
            i13 += [k.dur for k in in_range(D, mlo, mhi) if k.type == "AG" and k.comm == "I"]
        res.append(
            {
                "nO": len(O),
                "nI": len(I),
                "o_total": o_total,
                "conc": conc,
                "frac": conc / o_total if o_total else float("nan"),
                "o_with_i": sum(1 for k in O if UI.overlap(k.start, k.end) > 0),
                "o_mean": fmean([k.dur for k in O]),
                "o_med": fmed([k.dur for k in O]),
                "i_mean": fmean([k.dur for k in I]),
                "i_med": fmed([k.dur for k in I]),
                "i13_mean": fmean(i13),
                "i13_med": fmed(i13),
            }
        )
    return {k: fmean([r[k] for r in res]) for k in res[0]}


def q3_gemm(D):
    res = []
    for s in D["steps"]:
        lo0, hi0 = s["fwd"][0]
        ks0 = in_range(D, lo0, hi0)
        g0 = [k for k in ks0 if k.type == "GEMM"]
        g13 = []
        for m in (1, 2, 3):
            mlo, mhi = s["fwd"][m]
            g13 += [k for k in in_range(D, mlo, mhi) if k.type == "GEMM"]
        sp = gpu_span(ks0)
        res.append(
            {
                "wall0": (hi0 - lo0) / NS_PER_MS,
                "span0": (sp[1] - sp[0]) / NS_PER_MS,
                "n0": len(g0),
                "sum0": sum(k.dur for k in g0),
                "ov0": sum(D["nccl_union"].overlap(k.start, k.end) for k in g0) / NS_PER_MS,
                "mean0": fmean([k.dur for k in g0]),
                "n13": len(g13) / 3,
                "mean13": fmean([k.dur for k in g13]),
                "fwd_walls": [round((s["fwd"][m][1] - s["fwd"][m][0]) / NS_PER_MS, 2) for m in (0, 1, 2, 3)],
            }
        )
    keys = [k for k in res[0] if k != "fwd_walls"]
    return {k: fmean([r[k] for r in res]) for k in keys}, [r["fwd_walls"] for r in res]


def q4_rest(D):
    """Forward of mb1..3 (NVTX mb1_fwd..mb3_fwd), per step."""
    out = {"gemm": [], "nccl": [], "idle": [], "busy": [], "gaps": [], "span": []}
    for s in D["steps"]:
        wins, gemm = [], []
        for m in (1, 2, 3):
            lo, hi = s["fwd"][m]
            ks = in_range(D, lo, hi)
            sp = gpu_span(ks)
            if sp:
                wins.append(sp)
            for k in ks:
                if k.type == "GEMM":
                    gemm.append((k, D["nccl_union"].overlap(k.start, k.end) / (k.end - k.start)))
        out["gemm"].append(gemm)
        out["span"].append(sum(b - a for a, b in wins) / NS_PER_MS)

        seen = defaultdict(lambda: [0, 0.0])
        for k in D["kerns"]:
            if k.type in ("AG", "RS") and any(k.end > a and k.start < b for a, b in wins):
                seen[(k.cls, k.stream, k.name[:60])][0] += 1
                seen[(k.cls, k.stream, k.name[:60])][1] += k.dur
        out["nccl"].append(seen)

        comp = [(k.start, k.end) for k in D["kerns"] if k.stream == D["compute"]]
        idle, gap_n, gap_sum, busy_sum = 0.0, 0, 0.0, 0.0
        for a, b in wins:
            busy = merge([(max(x, a), min(y, b)) for x, y in comp if y > a and x < b])
            cur, busy_t = a, 0
            for x, y in busy:
                if x - cur >= GAP_EPS_MS * NS_PER_MS:
                    gap_n += 1
                    gap_sum += (x - cur) / NS_PER_MS
                busy_t += y - x
                cur = y
            if b - cur >= GAP_EPS_MS * NS_PER_MS:
                gap_n += 1
                gap_sum += (b - cur) / NS_PER_MS
            idle += ((b - a) - busy_t) / NS_PER_MS
            busy_sum += busy_t / NS_PER_MS
        out["idle"].append(idle)
        out["busy"].append(busy_sum)
        out["gaps"].append((gap_n, gap_sum))
    return out


def bucket_table(R_off, R_on):
    """mb1..3 GEMMs bucketed by the share of their duration covered by any NCCL kernel."""
    def bucket(R):
        b = defaultdict(list)
        for step in R["gemm"]:
            for k, frac in step:
                b[bucket_of(frac)].append(k.dur)
        return b

    off_b, on_b = bucket(R_off), bucket(R_on)
    rows = []
    for key in ("0%", "<50%", "50-99%", "100%"):
        o, e = off_b.get(key, []), on_b.get(key, [])
        rows.append((
            key,
            f"{len(o) / len(R_off['gemm']):.1f}",
            f3(fmed(o)),
            f"{len(e) / len(R_on['gemm']):.1f}",
            f3(fmed(e)),
        ))
    return md_table(
        ["share of GEMM duration covered by NCCL", "OFF count/step", "OFF median ms", "ON count/step", "ON median ms"],
        rows,
    )


def name_table(R_off, R_on, top=5):
    """Per-name medians over all mb1..3 GEMMs, top names by total time."""
    def collect(R):
        d = defaultdict(list)
        for step in R["gemm"]:
            for k, _ in step:
                d[k.name[:72]].append(k.dur)
        return d

    off_d, on_d = collect(R_off), collect(R_on)
    total = defaultdict(float)
    for d in (off_d, on_d):
        for n, v in d.items():
            total[n] += sum(v)
    rows = []
    for n in sorted(total, key=lambda n: -total[n])[:top]:
        o, e = off_d.get(n, []), on_d.get(n, [])
        rows.append((
            n,
            f"{len(o) / len(R_off['gemm']):.1f}",
            f3(fmed(o)),
            f"{len(e) / len(R_on['gemm']):.1f}",
            f3(fmed(e)),
        ))
    return md_table(["GEMM kernel (first 72 chars)", "OFF count/step", "OFF median ms", "ON count/step", "ON median ms"], rows)


def nccl_table(off4, on4):
    keys = set()
    for R in (off4, on4):
        for m in R["nccl"]:
            keys |= set(m.keys())
    rows = []
    for key in sorted(keys, key=lambda x: (x[0], x[1], x[2])):
        def stats(R):
            n = [m.get(key, [0, 0.0])[0] for m in R["nccl"]]
            t = [m.get(key, [0, 0.0])[1] for m in R["nccl"]]
            return fmean(n), fmean(t)
        on_n, on_t = stats(on4)
        off_n, off_t = stats(off4)
        rows.append((key[0], key[1], key[2], f"{off_n:.1f}", f2(off_t), f"{on_n:.1f}", f2(on_t)))
    return md_table(["class", "stream", "kernel (first 60 chars)", "OFF count/step", "OFF total ms", "ON count/step", "ON total ms"], rows)


def report(off, on):
    p = print
    p(f"- OFF file: `{off['name']}`; ON file: `{on['name']}`")
    p(f"- Profiled steps per file: OFF {len(off['steps'])}, ON {len(on['steps'])}. "
      f"Compute stream: OFF {off['compute']}, ON {on['compute']}.\n")

    p("### Validation: all-gather kernels per NVTX range (mean per step)\n")
    p(md_table(["file", "range", "AG kernels, comm-stream = count"], validation_rows(off, "OFF") + validation_rows(on, "ON")))
    p("")

    p("### Q1. CUDA streams with all-gather kernels in mb0_fwd\n")
    p(md_table(["file", "stream", "comm", "kernels/step", "total ms/step"],
               [("OFF",) + r for r in q1_streams(off)] + [("ON",) + r for r in q1_streams(on)]))
    p("")

    p("### Q2. O/I concurrency in mb0_fwd and single-kernel durations\n")
    o2, n2 = q2_overlap(off), q2_overlap(on)
    p(md_table(["metric", "OFF", "ON"], [
        ("O kernels per step", f"{o2['nO']:.1f}", f"{n2['nO']:.1f}"),
        ("I kernels per step", f"{o2['nI']:.1f}", f"{n2['nI']:.1f}"),
        ("total O kernel time, ms", f2(o2["o_total"]), f2(n2["o_total"])),
        ("time an O and an I kernel run together, ms", f2(o2["conc"]), f2(n2["conc"])),
        ("concurrent time / total O kernel time", f"{o2['frac']:.3f}", f"{n2['frac']:.3f}"),
        ("O kernels overlapping any I kernel", f"{o2['o_with_i']:.1f} of {o2['nO']:.1f}", f"{n2['o_with_i']:.1f} of {n2['nO']:.1f}"),
        ("mean / median O kernel duration in mb0_fwd, ms", f"{o2['o_mean']:.3f} / {o2['o_med']:.3f}", f"{n2['o_mean']:.3f} / {n2['o_med']:.3f}"),
        ("mean / median I kernel duration in mb0_fwd, ms", f"{o2['i_mean']:.3f} / {o2['i_med']:.3f}", f"{n2['i_mean']:.3f} / {n2['i_med']:.3f}"),
        ("mean / median I kernel duration in mb1..3 fwd, ms (no O traffic)", f"{o2['i13_mean']:.3f} / {o2['i13_med']:.3f}", f"{n2['i13_mean']:.3f} / {n2['i13_med']:.3f}"),
    ]))
    p("")

    p("### Q3. mb0_fwd wall time and GEMM work\n")
    o3, o3w = q3_gemm(off)
    n3, n3w = q3_gemm(on)
    p(md_table(["metric", "OFF", "ON"], [
        ("NVTX mb0_fwd duration (CPU-side), ms", f2(o3["wall0"]), f2(n3["wall0"])),
        ("GPU span of kernels launched in mb0_fwd, ms", f2(o3["span0"]), f2(n3["span0"])),
        ("GEMM kernels in mb0_fwd per step", f"{o3['n0']:.1f}", f"{n3['n0']:.1f}"),
        ("sum GEMM kernel time in mb0_fwd, ms", f2(o3["sum0"]), f2(n3["sum0"])),
        ("GEMM time overlapping any NCCL kernel, ms", f2(o3["ov0"]), f2(n3["ov0"])),
        ("GEMM time overlapping NCCL / GEMM time", f"{o3['ov0'] / o3['sum0']:.3f}", f"{n3['ov0'] / n3['sum0']:.3f}"),
        ("mean GEMM kernel duration in mb0_fwd, ms", f3(o3["mean0"]), f3(n3["mean0"])),
        ("GEMM kernels per microbatch in mb1..3 fwd", f"{o3['n13']:.1f}", f"{n3['n13']:.1f}"),
        ("mean GEMM kernel duration in mb1..3 fwd, ms", f3(o3["mean13"]), f3(n3["mean13"])),
    ]))
    p("")
    p("NVTX forward range durations (CPU-side), ms, per profiled step (mb0 / mb1 / mb2 / mb3):\n")
    p(md_table(["file", "step 1", "step 2"],
               [("OFF", *[" / ".join(f"{x:.2f}" for x in w) for w in o3w]),
                ("ON", *[" / ".join(f"{x:.2f}" for x in w) for w in n3w])]))
    p("")

    p("### Q4. Forward of mb1..3 (NVTX mb1_fwd, mb2_fwd, mb3_fwd)\n")
    o4, n4 = q4_rest(off), q4_rest(on)
    rows = []
    for tag, R in (("OFF", o4), ("ON", n4)):
        gem = [k for step in R["gemm"] for k, _ in step]
        free = [k.dur for step in R["gemm"] for k, f in step if f == 0]
        ovl = [k.dur for step in R["gemm"] for k, f in step if f > 0]
        rows.append((
            tag,
            f"{len(gem) / len(R['gemm']):.1f}",
            f"{len(free) / len(R['gemm']):.1f} / {f3(fmean(free))}",
            f"{len(ovl) / len(R['gemm']):.1f} / {f3(fmean(ovl))}",
            f2(fmean(R["span"])),
            f2(fmean(R["busy"])),
            f2(fmean(R["idle"])),
            f"{fmean([g for g, _ in R['gaps']]):.1f} / {f2(fmean([s for _, s in R['gaps']]))}",
        ))
    p(md_table([
        "file", "GEMM kernels per step", "GEMM with no NCCL overlap per step / mean ms",
        "GEMM with any NCCL overlap per step / mean ms", "GPU span of mb1..3 fwd, ms/step",
        "compute-stream busy in span, ms/step", "compute-stream idle in span, ms/step",
        "idle gaps >= 0.05 ms per step / sum ms",
    ], rows))
    p("")
    p("mb1..3 GEMM durations bucketed by how much of each GEMM overlaps any NCCL kernel:\n")
    p(bucket_table(o4, n4))
    p("")
    p("mb1..3 GEMM kernels, top 5 names by total time (median ms, all overlap states):\n")
    p(name_table(o4, n4))
    p("")
    p("NCCL kernels whose GPU interval overlaps the mb1..3 forward GPU windows (mean per step, full kernel durations):\n")
    p(nccl_table(o4, n4))
    p("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off", help="OFF sqlite export")
    ap.add_argument("on", help="ON sqlite export")
    args = ap.parse_args()
    on = build(args.on)
    off = build(args.off, ref=on["comm_ref"])
    report(off, on)


if __name__ == "__main__":
    main()
