"""Per-microbatch GPU timeline stats from an E1 nsys report (exported to sqlite).

For every NVTX range ``mbK_fwd`` (CPU-side), select the kernels whose launch API
call falls in that range (via correlationId), then report on the GPU timeline:

- span: first kernel start -> last kernel end of the forward
- compute busy: union of non-NCCL kernel intervals
- compute idle inside the span (bubbles = span - busy)
- NCCL all-gather kernels on the all-gather stream, in launch order, with
  durations. In mb0 each group issues outer AG then inner AG, so they alternate.

Usage: python e1_nsys_stats.py report.sqlite [--max-kernels 12]
(export with: nsys export --type sqlite -o report.sqlite report.nsys-rep)
"""

import argparse
import sqlite3
from collections import defaultdict


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--max-kernels", type=int, default=12)
    args = ap.parse_args()
    c = sqlite3.connect(args.sqlite)
    names = dict(c.execute("select id, value from StringIds"))
    ranges = []
    for start, end, text_id, text in c.execute(
        "select start, end, textId, text from NVTX_EVENTS where end is not null"
    ):
        label = text if text is not None else names.get(text_id)
        if label and label.startswith("mb") and label.endswith("_fwd"):
            ranges.append((start, end, label))
    ranges.sort()
    # kernel rows joined with their launching runtime call
    kernels = list(
        c.execute(
            "select r.start, k.start, k.end, k.streamId, k.shortName from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId"
        )
    )
    step = defaultdict(int)
    for start, end, label in ranges:
        step[label] += 1
        ks = [k for k in kernels if start <= k[0] <= end]
        if not ks:
            continue
        nccl = [k for k in ks if "nccl" in names[k[4]].lower()]
        comp = [k for k in ks if "nccl" not in names[k[4]].lower()]
        span_s = min(k[1] for k in ks)
        span_e = max(k[2] for k in ks)
        busy = union_len([(k[1], k[2]) for k in comp])
        comp_streams = sorted({k[3] for k in comp})
        nccl_streams = sorted({k[3] for k in nccl})
        print(
            f"{label} #{step[label]}: span={(span_e - span_s) / 1e6:7.2f} ms "
            f"compute_busy={busy / 1e6:7.2f} ms bubbles={(span_e - span_s - busy) / 1e6:7.2f} ms "
            f"nccl_kernels={len(nccl)} nccl_sum={sum(k[2] - k[1] for k in nccl) / 1e6:7.2f} ms "
            f"compute_streams={comp_streams} nccl_streams={nccl_streams}"
        )
        for k in sorted(nccl, key=lambda k: k[1])[: args.max_kernels]:
            print(
                f"    stream {k[3]:3d} t={(k[1] - span_s) / 1e6:7.2f} dur={(k[2] - k[1]) / 1e6:6.3f} ms "
                f"{names[k[4]]}"
            )


if __name__ == "__main__":
    main()
