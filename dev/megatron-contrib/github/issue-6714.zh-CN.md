# Issue #6714：MFSDP v2 中重叠 DP-inner 与 DP-outer 通信

- URL: https://github.com/NVIDIA/Megatron-LM/issues/6714
- 作者：@wujingyue  状态：open
- 创建时间：2026-08-20T19:22:47Z

## 正文

本 issue 是 #5616 的后续：探索在 MFSDP v2 中重叠 DP-inner 与 DP-outer 通信。

这两种通信走不同的网络域，早期经验表明让它们重叠可以显著降低延迟。实现应遵循
`megatron/core/distributed/fsdp/src/docs/runtime_schedule.md` 中的运行时调度模型。

我们应当先测量收益，再用结果指导实现和调度设计。

## 讨论

### @Achyuthan-S 2026-08-21T06:10:36Z

@wujingyue 谢谢，我来接这个任务。同意先测量：我会按 runtime_schedule.md 在一个
HSDP/HFSDP 配置上，量化 DP-outer 归约当前在关键路径上的开销，以及使用独立的
DP-outer stream 能找回多少延迟，并在提出实现方案之前把数据贴在这里。我只会把独立
stream 的原型做到能拿到真实的前后对比，然后让结果决定生产级设计。

### @Achyuthan-S 2026-08-24T06:28:02Z

@wujingyue，

我测量了 DP-inner/DP-outer 集合通信重叠带来的收益（先测量）。设置：2×2 DP mesh，
每个轴使用独立的 NCCL communicator（`mesh.get_group`），DP-inner reduce-scatter 与
DP-outer all-reduce 分别放在独立的 CUDA stream 上，用 CUDA event 计时。

 单节点，4×A100-80GB / 4×H100-80GB：

| 大小    | 串行           | 重叠           | 节省      |
|---------|----------------|----------------|-----------|
| 64 MiB  | 1.65 / 1.07 ms | 1.05 / 0.73 ms | 37% / 31% |
| 256 MiB | 6.14 / 3.83 ms | 3.92 / 2.56 ms | 36% / 33% |
| 1 GiB   | 23.3 / 14.4 ms | 14.8 / 9.8 ms  | 37% / 32% |

重叠后 ≈ max(RS, AR)——reduce-scatter 完全隐藏在 all-reduce 之后，找回了约三分之一的
合计集合通信时间，在各大小和两种 GPU 上结果一致。而在真实的 HSDP 路径中，DP-outer
all-reduce 目前完全排在 DP-inner 之后（profiler 中重叠为 0/N，两者都在
reduce-scatter stream 上），所以这三分之一的时间完全落在关键路径上。

注意事项：单节点（两个轴都在 NVLink 上）——在这里它们能干净地重叠，争用很小；多节点
（DP-outer 跨节点走 IB）是拿到跨域数字的自然后续。另外微基准每轮迭代都同时跑两种
集合通信，而训练中 DP-outer 只在最后一个 microbatch 发生，所以每步实际能省多少，需要
真正的独立 stream 实现才能确定。

考虑到这些集合通信在独立 communicator + stream 上确实能重叠、而目前是串行的，我想
继续做独立的 DP-outer stream 实现。这个顺序可以吗，还是你希望先做多节点测量？

### @wujingyue 2026-08-24T06:57:27Z

> 我想继续做独立的 DP-outer stream 实现。

就这么做。然后我可以在更大、更真实的配置上测量影响。

### @Achyuthan-S 2026-08-24T07:59:49Z

@wujingyue 开始做 DP-outer 重叠（#6714）。为了在独立 stream 上执行最后一个
microbatch 的 finalize 又不触发 #6441 的分配 stream 断言，我打算把 `main_grad` 的
所有权移到 dp_outer 归约 stream 上（在 `__init__` 中分配，HFSDP reset 时也在那里分配，
并在 DP-inner 归约后加一个 RS→dp_outer 的等待）。

这是你想要的所有权模型吗？还是你更希望 DP-outer 集合通信写入一个独立的 finalize
缓冲区，而 `main_grad` 继续留在 reduce-scatter stream 上？

### @wujingyue 2026-08-24T18:02:44Z

你能把几种方案都做个原型吗？当我们把 outer 和 inner 重叠时，似乎不可避免地要同时让
outer 归约后的缓冲和 all-reduce 后的缓冲存活。因此，我们可以在 outer-rs stream 上分配
一个 DBuffer，在 inner-rs stream 上分配另一个。

### @Achyuthan-S 2026-08-26T12:56:37Z

@wujingyue 明白了——就两个缓冲：`main_grad`（累加）留在 inner stream 上，另加一个
DP-outer 归约后的缓冲放在 outer stream 上，通过 `redistribute(out=...)` 做 finalize，
这样两者都保持存活，分配/释放也留在同一 stream。附带的好处是 `main_grad` 永远不会被
重新赋值，reset 路径保持简单。我会为 finalize 后的缓冲做两种生命周期方案的原型——
持久化 vs 每个最后 microbatch 分配、step 后释放——并比较 parity、重叠情况（event 计数）
和峰值内存。数据会同步给大家。

### @wujingyue 2026-08-26T15:13:49Z

> 数据会同步给大家

提醒一下：

我不知道你是否有合适的硬件来测量。这种重叠主要在 dp-outer 和 dp-inner 映射到不同
通信通道时有用，例如 RDMA vs nvlink。

等你有了原型并确认重叠确实发生之后，@rapatel 和我可以在更大、更真实的配置上测量影响。

### @rapatel 2026-08-26T15:23:52Z

这不就是 #5985 做的事情吗——它把 outer-DP 的 AG 和 inner-DP 的 AG 重叠起来。

### @Achyuthan-S 2026-08-26T15:37:37Z

> 这不就是 [#5985](https://github.com/NVIDIA/Megatron-LM/pull/5985) 做的事情吗——它把
> outer-DP 的 AG 和 inner-DP 的 AG 重叠起来。

@rapatel 谢谢指路——我看了 #5985。我认为两者是互补关系，而不是同一件事：

- #5985 在 DP-outer/DP-inner 两个域上流水化 HFSDP 的参数 all-gather——把 DP-outer
  的 gather 提前到比 DP-inner 前沿多一个 unit（`hfsdp_param_gather_overlap`）——位于
  mfsdp_v1 路径（`param_and_grad_buffer.py`），关联 #4130。
- 本工作（#6714）重叠的是 DP-outer 的梯度归约——最后一个 microbatch 的 all-reduce
  （HSDP）/ reduce-scatter（HFSDP）——与 DP-inner 的归约重叠，位于 mfsdp_v2 的
  experimental 路径。

  所以是不同的集合通信（参数 gather vs 梯度归约）、不同的代码路径（v1 vs v2）——
  一个是 DP-outer/DP-inner 重叠的 gather 侧，另一个是 reduce 侧。如果你看到它们真的会
  冲突的地方，请告诉我。

### @Achyuthan-S 2026-08-26T15:39:07Z

> > 数据会同步给大家
>
> 提醒一下：
>
> 我不知道你是否有合适的硬件来测量。这种重叠主要在 dp-outer 和 dp-inner 映射到不同
> 通信通道时有用，例如 RDMA vs nvlink。
>
> 等你有了原型并确认重叠确实发生之后，[@rapatel](https://github.com/rapatel) 和我
> 可以在更大、更真实的配置上测量影响。

@wujingyue 谢谢——硬件的限制我明白。

计划：用原型验证重叠确实发生（通过 profiler trace 确认 DP-outer 归约在自己的 stream
上与 DP-inner 并发执行），并保证 parity 逐位一致，然后交给 @rapatel 和你在真实的多通道
配置上测量影响。parity 和重叠验证的结果会贴在这里。

### @rapatel 2026-08-26T16:10:21Z

@Achyuthan-S 明白了——谢谢解释，说得通。那太好了——你的 PR 加上我提到的那个 PR，我们
就能把所有 outer-DP 通信和 inner-DP 通信重叠起来！

### @wujingyue 2026-08-26T16:57:21Z

@rapatel 这个 issue 是针对 v2 的，不是 v1。（我觉得你已经意识到了，只是确认一下）

### @rapatel 2026-08-26T17:35:55Z

@wujingyue 是的，我明白。只是想指出另一个 PR，因为那是类似的优化（只是发生在优化器
step 的另一侧、第一个 microbatch 期间）。这两个特性在 V2 中都应该支持。

### @Achyuthan-S 2026-08-27T01:43:26Z

同意——这里处理的是 v2 中 reduce 侧（最后一个 microbatch 的 DP-outer 归约）；gather 侧
是自然的对应部分。先在这里落地 reduce 侧重叠，parity 和重叠验证很快会跟上。


### @Achyuthan-S 2026-08-29T11:21:15Z

@wujingyue——reduce 侧的 DP-outer 重叠已经开好了（双缓冲方案，默认关闭，按你的建议）。

方便的时候，能否帮忙标成 ready 并跑一下 /ok to test？mfsdp_v2 单元测试桶会跑两种
zero_grad 模式下的 HSDP/HFSDP 重叠 parity——我想在开始做持久化 vs 临时缓冲生命周期的
内存对比之前，先确认正确性。重叠是否真正实现的基准测试我会另外跟进。

### @shsaihdsaiudh 2026-11-09T14:21:02Z

@wujingyue 你好，我来接 #5985 在 v2 中的 gather 侧对应工作：让优化器更新后的
DP-outer all-gather 在自己的 stream 上运行，从而与 DP-inner all-gather 重叠。
`context.py` 里仍然注明 outer all-gather 分离尚未实现。

我有 2 节点 × 8 张 H20（节点内 NVLink，节点间 RDMA）。按照这里的先测量原则，我会先
量化 HFSDP 中第一个 microbatch 的 DP-outer all-gather 有多少时间暴露在关键路径上
（分别在启用和禁用 CUDA graphs 的情况下），在写实现之前先把数据贴出来。
