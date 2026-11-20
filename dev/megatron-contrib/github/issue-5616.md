# Issue #5616: MFSDP v2 should support HSDP and HFSDP placements

- URL: https://github.com/NVIDIA/Megatron-LM/issues/5616
- Author: @wujingyue  State: closed
- Created: 2026-07-02T04:53:40Z

## Body

MFSDP v2 currently rejects the placement differences needed for HSDP/HFSDP.

`megatron_fsdp/experimental/parameter_group.py` requires `main_grad` and `main_weight` to have identical placements and raises until HSDP/HFSDP support is implemented. That blocks hybrid sharding modes where weights, gradients, and optimizer state intentionally use different inner-DP / outer-DP layouts.

Implementation detail:

- First microbatch: all-gather across DP-outer.
- Last microbatch: reduce-scatter or all-reduce across DP-outer.
- DP-outer communication should probably run on a separate stream so it can overlap with DP-inner communication.

Expected outcome:

- Define MFSDP v2 layout semantics for HSDP and HFSDP.
- Support HSDP outer-DP replication / gradient synchronization.
- Support HFSDP optimizer-state sharding across inner and outer DP ranks.
- Add focused tests for the required placement and collective transitions.

Development PRs:
- #5652


## Conversation

### @Achyuthan-S 2026-07-02T11:25:03Z

Hi @wujingyue — I dug into the experimental MFSDP v2 path and I'd be glad to help with this if it's open for outside contribution.

The blocker looks fairly well-scoped from the code:

- `FsdpParameterGroup` guards on `main_grad.placements == main_weight.placements` (`experimental/parameter_group.py`), which is exactly the invariant HSDP/HFSDP need to relax.
- `DBuffer.redistribute()` currently composes only a single-axis transition, so a 2-D `[Partial, Partial] → [Replicate, Flat]` reduction isn't yet expressible.
- `reduce_gradients()` reduces across all mesh axes at once, with no inner-reduce-scatter / outer-reduce split.
- #5441 already dropped the DBuffer mesh-axis validation, so the collectives can run on the outer/inner axes — that groundwork looks ready.

If a PR would be welcome, I'd suggest landing this incrementally rather than as one change:

1. **HSDP first** — outer-DP-replicated params/optimizer + outer all-reduce of gradients on a single stream. Smallest slice that replaces the guard with real support, and straightforward to test.
2. **HFSDP** (optimizer sharded across both inner and outer DP) as a follow-up.
3. The **separate outer-DP overlap stream** as its own PR, since it's orthogonal to correctness and overlaps your comm-overlap work (#5416 / #5513).

I'd add the first 2-D mesh unit tests along the way — the current `fully_shard` tests are all 1-D.

Two questions before I start:
- Are you already working on this, or would a PR be welcome?
- Anything in flight (e.g. the optimizer work in #5411) I should build on or wait for to avoid conflicts?

Happy to open a draft PR for step 1 if that direction sounds right.

### @wujingyue 2026-07-02T15:29:30Z

Thanks for the interest, @Achyuthan-S!

I'd wait for https://github.com/NVIDIA/Megatron-LM/pull/5513 and the microbatch-context work from https://github.com/NVIDIA/Megatron-LM/pull/5378 to land. #5513 adds `FsdpContext`, and the #5378 work adds state for deciding which microbatch we are processing.

I don't think you need https://github.com/NVIDIA/Megatron-LM/pull/5411 to move forward. You can work around it by using a precision-aware optimizer, e.g. SGD with `foreach=False`.

### @Achyuthan-S 2026-07-02T15:36:44Z

Thanks @wujingyue, that's really helpful — I'll wait on #5513 and the microbatch-context work before building the overlap/optimized path.

One thing: #5378 is closed and was based on `pull-request/4976`, so I don't have a live PR to track for the microbatch-context piece. Is that work being re-homed onto `main` in a new PR I should follow?

In the meantime, would you be open to a correctness-only HSDP slice that's independent of the microbatch context? I'd reduce gradients across DP-outer every backward on the default stream — numerically identical to a last-microbatch reduction since AVG is linear — which only needs the `main_grad`/`main_weight` guard relaxed plus a multi-axis `reduce_gradients`/`redistribute`, no `FsdpContext` or `is_first_microbatch`. Once #5513 and the microbatch work land, I'd follow up to move the outer-DP reduction to the last microbatch on a separate stream.

Either way I'll start on the HSDP/HFSDP layout semantics and the first 2-D-mesh unit tests, since those are unblocked. Thanks again for the pointers on the optimizer workaround.

### @wujingyue 2026-07-02T16:03:37Z

@Achyuthan-S

> Is that work being re-homed onto `main` in a new PR I should follow?

I'll make the new PR, and when it's ready I'll link it to this issue.

> I'd reduce gradients across DP-outer every backward on the default stream

I worry about the microbatch-specific logic more than streams. So even just for functionality, I wouldn't recommend proceeding without the microbatch PR.

If you follow https://google.github.io/eng-practices/review/developer/small-cls.html, there **might** be a chance I reuse some code from you.

### @Achyuthan-S 2026-07-02T16:38:32Z

Thanks @wujingyue , that's clear — I'll hold the HSDP functionality until your microbatch PR lands and watch the issue for the link. Agreed the microbatch-boundary logic (first-microbatch outer all-gather, last-microbatch outer reduce) is the substantive part; reduce-every-backward sidesteps exactly what's worth getting right, so I'll drop that framing.

I'll keep to small, self-contained CLs. Two unblocked pieces I'll prep in the meantime:

1. The first 2-D-mesh parity test for `fully_shard` (current tests are all 1-D) — it asserts final numerics, so it holds regardless of reduction cadence.
2. A short write-up of the HSDP/HFSDP layout semantics (parameter / gradient / optimizer placement tables across the outer/inner axes) for the issue's first deliverable.

One scoping question so I don't collide with your microbatch PR: would a small standalone CL for the multi-axis gradient reduction — composing inner reduce-scatter + outer reduce, [Partial, Partial] -> [Replicate, Flat] / [Flat, Flat] — be useful, or will that reduction live inside the microbatch PR? Happy to shape it whichever way slots in best.

### @wujingyue 2026-07-02T16:57:26Z

> Two unblocked pieces I'll prep in the meantime

SGTM

### @Achyuthan-S 2026-07-08T12:09:54Z

Hey @wujingyue , now that #5652 has landed I'll pick up the DP-outer gradient reduction on top of it — reduce-scatter across DP-inner every backward, reduce DP-outer only at is_last_microbatch, with the main_grad/main_weight guard relaxed for the outer-Partial accumulation — and put it up as a small draft PR. Shout if that's already in progress on your end.

### @wujingyue 2026-07-20T23:33:23Z

Hey @Achyuthan-S, congrats on getting #5743 merged!

If you'd like to keep contributing, here are a couple of areas we could work on next:

1. HFSDP support.
2. Communication overlap. The DP-inner and DP-outer communication happen across different network domains, so they can potentially overlap to improve latency.

cc @Autumn1998 for awareness and potential feedback on what features are most urgent. 

### @Achyuthan-S 2026-07-21T03:47:39Z

Thanks @wujingyue — really appreciate you shepherding this all the way through, and for the thorough reviews along the way. I learned a lot about the DBuffer/redistribute model and the reduce_scatter_stream overlap path.

I'd be glad to keep going. Between the two, I'd lean toward starting HFSDP next since it builds directly on this PR — the finalize is already a redistribute, so extending it to reduce-scatter the DP-outer axis (and re-allocating the [Partial, Flat] accumulation buffer, since that path shrinks the buffer) is the natural follow-on. Happy to take the DP-inner/DP-outer communication overlap after, or in parallel if that's more urgent.

@Autumn1998 — happy to hear which is the higher priority on your end, and I'll open an issue to scope whichever you'd prefer before I start.

### @wujingyue 2026-07-21T03:50:39Z

> I'll open an issue to scope whichever you'd prefer before I start.

This issue is good enough for tracking -- no need to open a new issue

### @Achyuthan-S 2026-07-21T04:04:43Z

Sounds good — I'll keep it under #5616. I'd be glad to take HFSDP as the next piece: optimizer -> [Flat, Flat], the finalize switching from all-reduce to reduce-scatter on the DP-outer axis, the fresh-buffer reset on that shrunk path, and extending the parity + event-count tests to cover it.

Before I dig in, though — is HFSDP the one you'd like me to pick up, or is the DP-inner/DP-outer overlap higher priority? Happy to go with whatever you and @Autumn1998 think is most urgent, and I'll scope it in #5616 once you point me at one.

### @wujingyue 2026-07-21T17:03:01Z

> is HFSDP the one you'd like me to pick up

SGTM. I'm pretty sure both are important. HFSDP will probably be easier for you at this moment. 

### @Achyuthan-S 2026-07-21T18:05:50Z

Great — I'll pick up HFSDP. It slots right onto this PR: the finalize is already a redistribute, so it's mostly swapping the DP-outer all-reduce for a reduce-scatter and handling the fresh-buffer reset on the shrunk path. I'll scope it in #5616 and open a draft PR against it once it's in shape — will tag you then. Thanks!

### @Achyuthan-S 2026-07-22T16:14:47Z

@wujingyue — opened the HFSDP follow-up we discussed (continues #5616, builds on #5743). It came out small: the only new code is the reset allocating a fresh [Partial, Flat] accumulation buffer for HFSDP's reduce-scattered finalize — the finalize and sync_model_weight already work through the existing redistribute path (Partial->Flat reduce-scatter and Flat->Replicate all-gather respectively). Added HFSDP placements, a 2-D parity test (both set_to_none modes × {1,3} microbatches), and an event-count test asserting the DP-outer finalize is a reduce-scatter, not an all-reduce.

### @wujingyue 2026-07-23T20:25:54Z

FYI, https://github.com/NVIDIA/Megatron-LM/pull/5985, a V1 optimization for better overlapping. 

### @Achyuthan-S 2026-07-24T04:33:47Z

Thanks for the pointer. Makes sense — #5985 is the V1 take on overlapping the DP-outer all-gather, which is exactly the gather this HFSDP path introduces via sync_model_weight. This PR is the V2 correctness foundation; prefetching that DP-outer AG post-optimizer-step would be the natural overlap follow-on once it lands (and it's a clean reference for doing it on the V2 side). Different code paths, so no interaction with this one — I'll keep it focused on correctness.

### @Autumn1998 2026-07-29T06:15:54Z

Hi @wujingyue  ,
In the HSDP test, the outer-DP gradient placement is explicitly initialized as Partial(dist.ReduceOp.AVG):

https://github.com/NVIDIA/Megatron-LM/blob/bc67abde107ca0a7751208ae142d09ce602e45be/tests/unit_tests/distributed/mfsdp_v2/test_fully_shard.py#L118

Only when symmetric memory is enabled, the temporary gradient buffer instead uses Partial(dist.ReduceOp.SUM) because the symmetric-memory reduce-scatter kernel currently requires SUM:

  https://github.com/NVIDIA/Megatron-LM/blob/66b592a2cb3cc4007ae0be06fc86a8ab78430ce2/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L262

  The result is subsequently divided by the reduction-group size:

  https://github.com/NVIDIA/Megatron-LM/blob/66b592a2cb3cc4007ae0be06fc86a8ab78430ce2/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L327-L341

  Therefore, SUM does not represent different user-visible gradient semantics here. It is only a backend-specific implementation of AVG: the symmetric-memory path lowers AVG into SUM + division.

 From a user’s perspective, choosing the sharding strategy should not require understanding this internal AVG versus SUM distinction. It also seems easy  to configure incorrectly because the most natural spelling, Partial(), silently selects SUM.

  Is there a good way to encapsulate this reduction-op handling inside FSDP, so users can define the placement and sharding strategy without needing to specify or understand the AVG/SUM implementation details?

### @wujingyue 2026-07-29T14:07:12Z

> Is there a good way to encapsulate this reduction-op handling inside FSDP, so users can define the placement and sharding strategy without needing to specify or understand the AVG/SUM implementation details?

Sorry, what's the problem exactly? Isn't `avg` equivalent to `sum / n`? cc @Autumn1998 

### @Autumn1998 2026-07-30T02:28:20Z

cc @wujingyue  What I mean is that a configuration like `gradient = [Partial(dist.ReduceOp.AVG), Flat()] `looks a bit odd. Ideally, when users initialize different strategies via placements, they shouldn’t have to think about whether it’s using AVG or SUM. Instead, they should be able to write `gradient = [Partial(), Flat()]` without worrying about the reduction op.

### @wujingyue 2026-07-30T03:16:22Z

> Instead, they should be able to write gradient = [Partial(), Flat()] without worrying about the reduction op.

Got it. See nv/mfsdp-design => API. We will provide helper functions to create common placements, e.g., HFSDP. This way, most of the times, the user doesn't need to worry about reduction ops. cc @Autumn1998 

### @Autumn1998 2026-07-30T10:04:37Z

cc @wujingyue
 I think that also makes sense, though I wonder if the following might be a bit better: instead of using SUM and AVG flag, when a reduce shows up during redistribution in dbuffer, we could check whether the input is sym memory, and if so, perform the reduce and then add a manual avg.

### @wujingyue 2026-07-30T17:32:23Z

Good question—it’s not obvious which approach is better. 

Semantically, users should specify AVG through the **API** because that reflects the intended operation: averaging gradients across ranks for example. At the implementation level, however, because symmetric memory doesn’t support AVG directly, the code can **internally** create or update a DBuffer using SUM, then divide the result by the number of ranks.

What do you think, @Autumn1998? 

### @Autumn1998 2026-07-31T02:26:54Z

I prefer to encapsulate this logic inside the dbuffer. Although this adds some complexity to the dbuffer, it reduces the amount of thinking we have to do when using dbuffer 🤔. cc @wujingyue 

### @wujingyue 2026-07-31T02:57:37Z

> I prefer to encapsulate this logic inside the dbuffer. 

I like it. I think I misread your earlier comment. Let me see if I can hide the if-symm-mem-then-sum-and-div logic inside DBuffer. This way, code outside can simply use AVG unconditionally. 

### @Achyuthan-S 2026-08-16T05:52:02Z

Thanks @wujingyue  for getting #5972  across the line — the GPU repro that pinned it to the compute-weight sync, and pushing it through the merge queue. Glad HFSDP is in. Happy to pick up the DP-outer reduction overlap next if that's still the plan — let me know how you'd like to scope it.

### @wujingyue 2026-08-16T07:34:38Z

> Happy to pick up the DP-outer reduction overlap next if that's still the plan 

Yes but let's build on top of https://github.com/NVIDIA/Megatron-LM/pull/6137

### @Achyuthan-S 2026-08-16T12:49:27Z

@wujingyue That sounds good — I'll base the DP-outer overlap on #6137 so it sits on the reworked reduction path.

One sequencing question: would you rather I wait for #6137 to land and build on main, or start now on @Autumn1998's branch and rebase as it evolves? Since the overlap touches the same reduce path #6137 restructures, I'd lean toward building on it once it merges to avoid churn — but happy to start on the branch in parallel if you'd prefer. Either way I'll sync with @Autumn1998 on the design so the overlap slots into the new structure cleanly.

### @wujingyue 2026-08-16T13:39:09Z

Also, be aware of https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/docs/runtime_schedule.md

> would you rather I wait for https://github.com/NVIDIA/Megatron-LM/pull/6137 to land and build on main, or start now on @Autumn1998's branch and rebase as it evolves?

Up to you. 

### @Achyuthan-S 2026-08-16T14:59:17Z

Thanks @wujingyue — read through runtime_schedule.md; that's the design I'll target.

On sequencing: I'll wait for #6137 to land and build on main — since it's approved and reworking the same reduce path, rebasing over its conflict-resolution churn would be noisier than waiting. Meanwhile I'll sketch the overlap design against #6137's structure + the runtime schedule and share it for a sanity check before writing code, and sync with @Autumn1998 so it slots into the new layout cleanly.

### @Achyuthan-S 2026-08-20T18:21:54Z

@wujingyue Sketched the DP-outer overlap against #6137's reduce path. My read: the last-microbatch finalize (parameter_group.py) runs on reduce_scatter_stream, shared with the DP-inner reductions — and since the two axes already resolve to distinct NCCL communicators via the mesh, moving the finalize to its own stream should let it progress independently, no separate communicator needed. Does that match your model, or is there a shared-communicator path that would still serialize them?

 If that holds: an optional DP-outer reduction stream on FsdpContext (off by default, mirroring unify_communication_stream), finalize buffer alloc/free kept on that stream (no record_stream), and a wait before optimizer.step(). Standalone flag + benchmark, or fold into a broader schedule change?

### @wujingyue 2026-08-20T19:18:04Z

I suspect there have to be separate streams for dp-outer otherwise dp-outer communications and dp-inner can't overlap. Is it yet to be measured how much this overlap buy us. Anecdotes say it's significant. 

### @wujingyue 2026-08-20T19:18:40Z

I'll close this bug and file a separate one for overlapping. 
