"""Race check on an E4 nsys sqlite export, same question as #6714's benchmark analysis.

For every DP-inner all-gather (AG-I: NCCL AllGather with the largest gridX), find the
last compute-stream kernel X that ended before AG-I started and the next compute kernel G
after X. AG-I "wins the race" when it starts before G, i.e. inside the X -> G gap.
Also reports compute-stream GEMM time and how much of it overlaps any NCCL kernel.

Usage: python e4_nsys_race.py <file.sqlite> [...]
"""

import bisect
import collections
import sqlite3
import statistics
import sys


def main():
    print(f"{'file':28s} {'AG-I':>5s} {'AG-I first':>11s} {'gap_us med':>10s} "
          f"{'GEMM ms':>9s} {'GEMM n':>7s} {'GEMM w/ AG-I ms':>15s} {'compute busy ms':>15s} "
          f"{'outer AG ms':>11s} {'outer AG exposed ms':>19s}")
    for path in sys.argv[1:]:
        db = sqlite3.connect(path)
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        rows = db.execute(
            "SELECT start, end, streamId, shortName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ).fetchall()
        busy = collections.Counter(r[2] for r in rows)
        compute = busy.most_common(1)[0][0]
        comp = [(s, e, names[n]) for s, e, sid, n, gx in rows if sid == compute]
        cstart = [c[0] for c in comp]
        cend = sorted((c[1], i) for i, c in enumerate(comp))
        cend_t = [x[0] for x in cend]
        ag = [(s, e, gx) for s, e, sid, n, gx in rows if sid != compute and "AllGather" in names[n]]
        gx_inner = max(g for _, _, g in ag)
        agi = [(s, e) for s, e, g in ag if g == gx_inner]
        ago = [(s, e) for s, e, g in ag if g != gx_inner]

        first, gaps = 0, []
        for s, _ in agi:
            k = bisect.bisect_left(cend_t, s) - 1
            if k < 0:
                continue
            x_end, xi = cend[k]
            j = bisect.bisect_left(cstart, x_end)
            if j >= len(comp):
                continue
            g_start = comp[j][0]
            if g_start > s:
                first += 1
                gaps.append((g_start - x_end) / 1e3)

        def overlaps(s, e, ivs):
            return any(a < e and b > s for a, b in ivs)

        gemm = [(s, e) for s, e, n in comp if "gemm" in n.lower() or "nvjet" in n.lower() or "sm90" in n.lower()]
        agi_sorted = sorted(agi)
        agi_starts = [a for a, _ in agi_sorted]

        def near(s, e):
            k = bisect.bisect_left(agi_starts, e)
            return [agi_sorted[i] for i in range(max(0, k - 3), k)]

        gemm_ms = sum(e - s for s, e in gemm) / 1e6
        gemm_ov = sum(e - s for s, e in gemm if overlaps(s, e, near(s, e))) / 1e6
        comp_ms = sum(e - s for s, e, _ in comp) / 1e6
        # Outer AG time not covered by any compute kernel (upper bound on exposure).
        exposed = 0.0
        for s, e in ago:
            covered = 0
            k = bisect.bisect_left(cstart, e)
            for cs, ce, _ in comp[max(0, k - 50):k]:
                covered += max(0, min(e, ce) - max(s, cs))
            exposed += max(0, (e - s) - covered) / 1e6
        ago_ms = sum(e - s for s, e in ago) / 1e6
        gap = f"{statistics.median(gaps):10.1f}" if gaps else f"{'-':>10s}"
        name = path.rsplit("/", 1)[-1].replace(".sqlite", "")
        print(f"{name:28s} {len(agi):5d} {first:5d} ({first / max(1, len(agi)):4.0%}) {gap} "
              f"{gemm_ms:9.1f} {len(gemm):7d} {gemm_ov:15.1f} {comp_ms:15.1f} {ago_ms:11.1f} {exposed:19.1f}")


if __name__ == "__main__":
    main()
