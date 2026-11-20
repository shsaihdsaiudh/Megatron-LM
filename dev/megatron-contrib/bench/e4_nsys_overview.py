"""Overview of an E4 nsys sqlite export: kernels per stream, NCCL kernels, NVTX ranges.

Usage: python e4_nsys_overview.py <file.sqlite> [...]
"""

import collections
import sqlite3
import sys


def main():
    for path in sys.argv[1:]:
        db = sqlite3.connect(path)
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        rows = db.execute(
            "SELECT start, end, streamId, shortName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchall()
        t0 = min(r[0] for r in rows)
        t1 = max(r[1] for r in rows)
        print(f"=== {path}: {len(rows)} kernels over {(t1 - t0) / 1e6:.1f} ms")
        per_stream = collections.defaultdict(lambda: [0, 0.0])
        nccl = collections.defaultdict(lambda: [0, 0.0])
        for s, e, sid, name, gx in rows:
            per_stream[sid][0] += 1
            per_stream[sid][1] += (e - s) / 1e6
            n = names[name]
            if "nccl" in n.lower():
                nccl[(sid, n, gx)][0] += 1
                nccl[(sid, n, gx)][1] += (e - s) / 1e6
        print("stream  kernels  busy_ms")
        for sid, (c, ms) in sorted(per_stream.items()):
            print(f"{sid:6d} {c:8d} {ms:9.1f}")
        print("NCCL kernels: stream, name, gridX, count, total ms")
        for (sid, n, gx), (c, ms) in sorted(nccl.items()):
            print(f"  {sid:4d} {n[:60]:60s} {gx:4d} {c:6d} {ms:9.1f}")
        try:
            nv = db.execute(
                "SELECT text, COUNT(*), SUM(end - start) FROM NVTX_EVENTS "
                "WHERE text IS NOT NULL AND end IS NOT NULL GROUP BY text ORDER BY 2 DESC LIMIT 40"
            ).fetchall()
            print("NVTX ranges (text, count, total ms):")
            for t, c, d in nv:
                print(f"  {t[:70]:70s} {c:6d} {(d or 0) / 1e6:10.1f}")
        except sqlite3.OperationalError as err:
            print("no NVTX:", err)


if __name__ == "__main__":
    main()
