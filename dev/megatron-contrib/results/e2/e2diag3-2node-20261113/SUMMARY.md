# E2-diag3 (job t-20261113150210-cuk58)

d4096_L32, mesh (2,8), GA=4, all arms overlap_outer_communication=True.
Order: main-on, p1-on, f5, f5, p1-on, main-on. node0 sweep logs, mean of 2 repeats, ms.

| arm | fwd_rest T=1024 | fwd_rest T=4096 | step T=1024 | step T=4096 | nostep step T=4096 |
|---|---|---|---|---|---|
| main-on | 78.7 | 269.1 | 954.0 | 3252.2 | 3236.8 |
| p1-on | 85.5 | 296.4 | 973.3 | 3360.4 | 3351.3 |
| f5 (P1 stream, serialized with inner AG) | 85.6 | 296.4 | 978.5 | 3351.4 | 3352.6 |

nsys (rank 8, T=1024, f5 run 03): DP-outer all-gathers (gridX 4) on stream 47, inner
(gridX 24) on stream 39, O/I kernel overlap 0.000 ms (P1: 24.9 ms; flag off: 0).
So f5 isolates "outer all-gather on another stream" from "O and I run concurrently".

Conclusion: the slowdown does not need O/I concurrency. Launching the DP-outer all-gather
on a stream other than the DP-inner all-gather stream is enough; the effect then persists
for later forward passes (including nostep configs). With f3 (no new stream) and f4
(cross-stream waits without moving the NCCL op) from E2-diag2, the trigger is the NCCL
launch stream of the outer collective.

Lead: NCCL logs show the outer (2-rank) and inner (8-rank) comms are both ncclCommSplit
children of the 16-rank world comm, with split timings "kernels 0.00, alloc 0.00",
which suggests shared resources (shared internal streams). Untested.
