# E2-diag2 (job t-20261113125843-z96iu)

d4096_L32, mesh (2,8), GA=4, all arms overlap_outer_communication=True.
Order: main-on, p1-on, f3, f4, f4, f3, p1-on, main-on. Values: node0 sweep logs, mean of 2 repeats, ms.

| arm | fwd_rest T=1024 | fwd_rest T=4096 | step T=4096 | nostep step T=4096 | E T=1024 / 4096 |
|---|---|---|---|---|---|
| main-on | 78.8 | 269.1 | 3255.0 | 3237.7 | 14.8 / 18.9 |
| p1-on | 85.6 | 296.9 | 3362.9 | 3349.7 | 5.7 / 8.0 |
| f3 (P1, outer AG on outer RS stream) | 85.5 | 296.6 | 3361.9 | 3350.8 | 4.8 / 7.1 |
| f4 (main + fork/1-element kernel/join on extra stream) | 78.8 | 269.1 | 3254.7 | 3235.1 | 15.3 / 20.2 |

Repeat differences are below 0.4% for step and fwd_rest.
Losses (raw_rank0.losses, all-rank loss_sum) are bit-identical across all 8 runs; peak memory
is equal except +512 B in f4 (its 1-element scratch tensor).

Conclusion: a new stream is not needed for the slowdown (f3 has none and is as slow as P1),
and cross-stream event waits around the first-microbatch unshard do not cause it (f4 is as
fast as main). The trigger is running the DP-outer all-gather NCCL kernel on a stream other
than the DP-inner all-gather stream. Not yet separated: whether the cause is the stream
identity of the NCCL op or the O/I kernels actually running concurrently.
