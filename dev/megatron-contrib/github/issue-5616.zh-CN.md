# Issue #5616：MFSDP v2 应支持 HSDP 和 HFSDP 的 placement

- URL: https://github.com/NVIDIA/Megatron-LM/issues/5616
- 作者：@wujingyue  状态：closed
- 创建时间：2026-07-02T04:53:40Z

## 正文

MFSDP v2 目前会拒绝 HSDP/HFSDP 所需的 placement 差异。

`megatron_fsdp/experimental/parameter_group.py` 要求 `main_grad` 和 `main_weight`
具有完全相同的 placement，在 HSDP/HFSDP 支持实现之前会直接抛错。这阻塞了混合分片模式
——在这些模式中，权重、梯度和优化器状态会有意采用不同的 inner-DP / outer-DP 布局。

实现细节：

- 第一个 microbatch：在 DP-outer 上做 all-gather。
- 最后一个 microbatch：在 DP-outer 上做 reduce-scatter 或 all-reduce。
- DP-outer 通信最好放在单独的 stream 上，以便与 DP-inner 通信重叠。

期望结果：

- 定义 HSDP 和 HFSDP 的 MFSDP v2 布局语义。
- 支持 HSDP 的 outer-DP 复制 / 梯度同步。
- 支持 HFSDP 在 inner 和 outer DP rank 上分片优化器状态。
- 为所需的 placement 和集合通信状态转换添加有针对性的测试。

开发 PR：
- #5652


## 讨论

### @Achyuthan-S 2026-07-02T11:25:03Z

你好 @wujingyue——我研究了 experimental 的 MFSDP v2 路径，如果这个问题还接受外部
贡献，我很乐意帮忙。

从代码看，这个障碍的范围比较清晰：

- `FsdpParameterGroup` 校验 `main_grad.placements == main_weight.placements`
  （`experimental/parameter_group.py`），而这正是 HSDP/HFSDP 需要放宽的不变量。
- `DBuffer.redistribute()` 目前只能组合单轴转换，所以二维的
  `[Partial, Partial] → [Replicate, Flat]` 归约还无法表达。
- `reduce_gradients()` 一次性在所有 mesh 轴上归约，没有 inner-reduce-scatter /
  outer-reduce 的拆分。
- #5441 已经去掉了 DBuffer 的 mesh 轴校验，因此集合通信可以在 outer/inner 轴上运行
  ——这部分基础看起来已就绪。

如果欢迎 PR，我建议增量落地，而不是一次做完：

1. **先做 HSDP**——outer-DP 复制的参数/优化器 + 在单条 stream 上对梯度做 outer
   all-reduce。这是用真正的支持替换该 guard 的最小切片，也便于测试。
2. **HFSDP**（优化器在 inner 和 outer DP 上都分片）作为后续。
3. **独立的 outer-DP 重叠 stream** 单独一个 PR，因为它与正确性正交，并与你的通信
   重叠工作（#5416 / #5513）重叠。

过程中我会补上最早的一批二维 mesh 单元测试——现有的 `fully_shard` 测试全是一维的。

开始前有两个问题：
- 你是否已经在做这件事，还是欢迎 PR？
- 有没有正在进行的工作（比如 #5411 里的优化器工作）我应当基于它来做，或者等它落地
  以避免冲突？

如果方向没问题，我很乐意先为第 1 步开一个草稿 PR。

### @wujingyue 2026-07-02T15:29:30Z

感谢关注，@Achyuthan-S！

我会等 https://github.com/NVIDIA/Megatron-LM/pull/5513 以及来自
https://github.com/NVIDIA/Megatron-LM/pull/5378 的 microbatch-context 工作落地。
其中 #5513 增加了 `FsdpContext`，而 #5378 的工作增加了用于判断当前处理哪个 microbatch 的
状态。

我认为你不需要 https://github.com/NVIDIA/Megatron-LM/pull/5411 才能推进。你可以用
精度感知的优化器绕过它，例如 `foreach=False` 的 SGD。

### @Achyuthan-S 2026-07-02T15:36:44Z

谢谢 @wujingyue，很有帮助——我会先等 #5513 和 microbatch-context 工作，再构建
重叠/优化路径。

有一个问题：#5378 已关闭，且基于 `pull-request/4976`，所以 microbatch-context 这部分
我没有一个可跟踪的活跃 PR。那部分工作是否会在新 PR 中重新落到 `main` 上，我该关注哪个？

同时，你是否愿意接受一个只保证正确性、与 microbatch context 无关的 HSDP 切片？我会在
每次反向时于默认 stream 上跨 DP-outer 归约梯度——由于 AVG 是线性的，数值上等价于在最后
一个 microbatch 归约——只需要放宽 `main_grad`/`main_weight` 的 guard，加上多轴的
`reduce_gradients`/`redistribute`，不需要 `FsdpContext` 或 `is_first_microbatch`。
等 #5513 和 microbatch 工作落地后，我再把 outer-DP 归约改到最后一个小 batch、放到独立
stream 上。

无论哪种方式，我都会先开始做 HSDP/HFSDP 布局语义和第一批二维 mesh 单元测试，因为这些
没有阻塞。再次感谢关于优化器绕行方案的指点。

### @wujingyue 2026-07-02T16:03:37Z

@Achyuthan-S

> 那部分工作是否会在新 PR 中重新落到 `main` 上，我该关注哪个？

我会开新 PR，准备好后链接到本 issue。

> 我会在每次反向时于默认 stream 上跨 DP-outer 归约梯度

比起 stream，我更担心 microbatch 相关的逻辑。所以即使只考虑功能，我也不建议在没有
microbatch PR 的情况下推进。

如果你遵循 https://google.github.io/eng-practices/review/developer/small-cls.html，
**也许**有机会复用你的一些代码。

### @Achyuthan-S 2026-07-02T16:38:32Z

谢谢 @wujingyue，很清楚——我会等你的 microbatch PR 落地再做 HSDP 功能，并关注 issue
里的链接。同意 microbatch 边界逻辑（第一个 microbatch 做 outer all-gather，最后一个
microbatch 做 outer reduce）才是实质部分；每次反向都归约恰好绕开了真正值得做对的地方，
所以我不再用那个说法。

我会坚持小而自包含的 CL。与此同时准备两个没有阻塞的部分：

1. `fully_shard` 的第一个二维 mesh parity 测试（现有测试全是一维）——它断言最终数值，
   因此无论归约节奏如何都成立。
2. 一份简短的 HSDP/HFSDP 布局语义说明（跨 outer/inner 轴的参数 / 梯度 / 优化器
   placement 表），作为本 issue 的第一个交付物。

一个范围问题，以免与你的 microbatch PR 冲突：一个独立的小 CL，只做多轴梯度归约——组合
inner reduce-scatter + outer reduce，`[Partial, Partial] -> [Replicate, Flat]` /
`[Flat, Flat]`——会有用吗，还是这个归约会放在 microbatch PR 里？无论哪种安排最合适，
我都可以配合。

### @wujingyue 2026-07-02T16:57:26Z

> 与此同时准备两个没有阻塞的部分

SGTM（听起来不错）

### @Achyuthan-S 2026-07-08T12:09:54Z

嗨 @wujingyue，#5652 已经落地，我会在它之上接 DP-outer 梯度归约这部分——每次反向跨
DP-inner 做 reduce-scatter，只在 `is_last_microbatch` 时归约 DP-outer，同时为
outer-Partial 累加放宽 `main_grad`/`main_weight` 的 guard——并作为一个小的草稿 PR 提交。
如果你那边已经在做，请告诉我一声。

### @wujingyue 2026-07-20T23:33:23Z

嗨 @Achyuthan-S，恭喜 #5743 合并！

如果你想继续贡献，下面有几个我们可以接着做的方向：

1. HFSDP 支持。
2. 通信重叠。DP-inner 和 DP-outer 通信走不同的网络域，因此有可能重叠以改善延迟。

cc @Autumn1998，让你了解情况，也欢迎你就哪些特性最紧急给些反馈。

### @Achyuthan-S 2026-07-21T03:47:39Z

谢谢 @wujingyue——非常感谢你一路把关，也感谢过程中细致的评审。我对
DBuffer/redistribute 模型和 reduce_scatter_stream 重叠路径学到了很多。

我很乐意继续。两者之间，我倾向于先做 HFSDP，因为它直接构建在本次 PR 之上——finalize
本来就是一个 redistribute，把它扩展为对 DP-outer 轴做 reduce-scatter（并重新分配
`[Partial, Flat]` 累加缓冲，因为那条路径会让缓冲变小）是很自然的后续。之后我很乐意做
DP-inner/DP-outer 通信重叠；如果那边更紧急，也可以并行。

@Autumn1998——想听听你那边哪个优先级更高，开始前我会按你的偏好开 issue 界定范围。

### @wujingyue 2026-07-21T03:50:39Z

> 开始前我会按你的偏好开 issue 界定范围。

用这个 issue 跟踪就足够了——不用另开新 issue

### @Achyuthan-S 2026-07-21T04:04:43Z

好的——我会继续放在 #5616 下。我很乐意把 HFSDP 作为下一部分：优化器 -> `[Flat, Flat]`，
finalize 从 all-reduce 改为在 DP-outer 轴上做 reduce-scatter，在缩小后的路径上重置新
缓冲，并把 parity + event 计数测试扩展到覆盖它。

不过开始深入之前——你是希望我接 HFSDP，还是 DP-inner/DP-outer 重叠优先级更高？我乐意
按你和 @Autumn1998 认为最紧急的方向来，等你指一个方向我就在 #5616 里界定范围。

### @wujingyue 2026-07-21T17:03:01Z

> 你是希望我接 HFSDP 吗

SGTM。我很确定两者都重要。目前对你来说 HFSDP 可能更容易一些。

### @Achyuthan-S 2026-07-21T18:05:50Z

好——我来接 HFSDP。它正好接在本次 PR 上：finalize 本来就是 redistribute，所以主要就是
把 DP-outer 的 all-reduce 换成 reduce-scatter，并处理缩小路径上的新缓冲重置。我会在
在本 issue（#5616）里界定范围，成形后针对它开草稿 PR——到时候 @ 你。谢谢！

### @Achyuthan-S 2026-07-22T16:14:47Z

@wujingyue——我们讨论过的 HFSDP 后续已经开好（延续 #5616，基于 #5743）。改动很小：
唯一的新代码是在 reset 时为 HFSDP 的 reduce-scatter finalize 分配一个全新的
`[Partial, Flat]` 累加缓冲——finalize 和 sync_model_weight 已经可以走现有的
redistribute 路径（分别是 Partial->Flat 的 reduce-scatter 和 Flat->Replicate 的
all-gather）。新增了 HFSDP placement、二维 parity 测试（两种 set_to_none 模式 ×
{1,3} 个 microbatch），以及一个 event 计数测试，断言 DP-outer finalize 是
reduce-scatter 而不是 all-reduce。

### @wujingyue 2026-07-23T20:25:54Z

供参考：https://github.com/NVIDIA/Megatron-LM/pull/5985，一个用于更好重叠的 V1 优化。

### @Achyuthan-S 2026-07-24T04:33:47Z

谢谢指路。有道理——#5985 是对"重叠 DP-outer all-gather"的 V1 方案，而这正是本 HFSDP
路径通过 sync_model_weight 引入的 gather。本 PR 是 V2 的正确性基础；在优化器 step 之后
预取那个 DP-outer AG，等它落地后就是自然的重叠后续（也是 V2 侧实现的干净参考）。两者
代码路径不同，因此与本 PR 没有交互——我会让本 PR 专注于正确性。

### @Autumn1998 2026-07-29T06:15:54Z

嗨 @wujingyue，
在 HSDP 测试中，outer-DP 的梯度 placement 被显式初始化为
`Partial(dist.ReduceOp.AVG)`：

https://github.com/NVIDIA/Megatron-LM/blob/bc67abde107ca0a7751208ae142d09ce602e45be/tests/unit_tests/distributed/mfsdp_v2/test_fully_shard.py#L118

只有在启用对称内存（symmetric memory）时，临时梯度缓冲才改用
`Partial(dist.ReduceOp.SUM)`，因为对称内存的 reduce-scatter kernel 目前要求 SUM：

  https://github.com/NVIDIA/Megatron-LM/blob/66b592a2cb3cc4007ae0be06fc86a8ab78430ce2/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L262

  结果随后会除以归约组的大小：

  https://github.com/NVIDIA/Megatron-LM/blob/66b592a2cb3cc4007ae0be06fc86a8ab78430ce2/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py#L327-L341

  因此，SUM 在这里并不代表不同的用户可见梯度语义。它只是 AVG 的一种后端特定实现：
  对称内存路径把 AVG 降低为 SUM + 除法。

 从用户角度看，选择分片策略不应该需要理解这个内部的 AVG vs SUM 区别。而且这很容易
 配错，因为最自然的写法 `Partial()` 会静默地选择 SUM。

 有没有好的办法把这种 reduction op 的处理封装在 FSDP 内部，让用户定义 placement 和
 分片策略时无需指定或理解 AVG/SUM 的实现细节？

### @wujingyue 2026-07-29T14:07:12Z

> 有没有好的办法把这种 reduction op 的处理封装在 FSDP 内部，让用户定义 placement 和
> 分片策略时无需指定或理解 AVG/SUM 的实现细节？

抱歉，具体问题是什么？`avg` 不就等价于 `sum / n` 吗？cc @Autumn1998

### @Autumn1998 2026-07-30T02:28:20Z

cc @wujingyue 我的意思是，像 `gradient = [Partial(dist.ReduceOp.AVG), Flat()]` 这样的
配置看起来有点奇怪。理想情况下，用户通过 placement 初始化不同策略时，不应该需要思考
用的是 AVG 还是 SUM。相反，他们应该能直接写 `gradient = [Partial(), Flat()]`，而不用
关心 reduction op。

### @wujingyue 2026-07-30T03:16:22Z

> 相反，他们应该能直接写 gradient = [Partial(), Flat()]，而不用关心 reduction op。

明白了。见 nv/mfsdp-design => API。我们会提供辅助函数来创建常见 placement，例如
HFSDP。这样大多数时候用户不需要关心 reduction op。cc @Autumn1998

### @Autumn1998 2026-07-30T10:04:37Z

cc @wujingyue
 我觉得那也有道理，不过我在想下面这种方式会不会更好一点：与其使用 SUM 和 AVG 标志，
 不如当 dbuffer 在 redistribute 期间出现 reduce 时，检查输入是否为 sym memory，如果是，
 就执行归约然后手动加一个 avg。

### @wujingyue 2026-07-30T17:32:23Z

好问题——很难说哪种方案更好。

从语义上讲，用户应该通过 **API** 指定 AVG，因为这反映了意图中的操作：例如在各 rank
之间平均梯度。然而在实现层面，由于对称内存并不直接支持 AVG，代码可以**在内部**用 SUM
创建或更新 DBuffer，然后把结果除以 rank 数。

你怎么看，@Autumn1998？

### @Autumn1998 2026-07-31T02:26:54Z

我更倾向于把这段逻辑封装在 dbuffer 内部。虽然这会给 dbuffer 增加一些复杂度，但能减少
我们使用 dbuffer 时需要思考的东西 🤔。cc @wujingyue

### @wujingyue 2026-07-31T02:57:37Z

> 我更倾向于把这段逻辑封装在 dbuffer 内部。

我喜欢。我想我之前误读了你的上一条评论。我看看能不能把"如果是 symm mem 就用 sum 再
除"的逻辑藏进 DBuffer 里。这样外面的代码就可以无条件使用 AVG。

### @Achyuthan-S 2026-08-16T05:52:02Z

谢谢 @wujingyue 把 #5972 推过终点线——那个把问题定位到 compute-weight sync 的 GPU
复现，以及推动它通过合并队列。很高兴 HFSDP 已经合入。如果计划不变，我很乐意接下来做
DP-outer 归约重叠——请告诉我你希望如何界定范围。

### @wujingyue 2026-08-16T07:34:38Z

> 如果计划不变，我很乐意接下来做 DP-outer 归约重叠

可以，但请基于 https://github.com/NVIDIA/Megatron-LM/pull/6137 来做。

### @Achyuthan-S 2026-08-16T12:49:27Z

@wujingyue 好的——我会把 DP-outer 重叠建立在 #6137 之上，这样它就落在重构后的归约
路径上。

一个顺序问题：你是希望我等 #6137 落地后基于 main 来做，还是现在就在 @Autumn1998 的
分支上开始、随它演进而 rebase？由于重叠会触及 #6137 重构的同一段 reduce 路径，我倾向
于等它合并后再基于它做，以避免反复折腾——但如果你更希望，我也可以并行在分支上开始。
无论哪种方式，我都会与 @Autumn1998 对齐设计，让重叠能干净地嵌入新结构。

### @wujingyue 2026-08-16T13:39:09Z

另外，注意一下
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/docs/runtime_schedule.md

> 你是希望我等 https://github.com/NVIDIA/Megatron-LM/pull/6137 落地后基于 main 来做，
> 还是现在就在 @Autumn1998 的分支上开始、随它演进而 rebase？

你决定。

### @Achyuthan-S 2026-08-16T14:59:17Z

谢谢 @wujingyue——我读完了 runtime_schedule.md；那就是我要对齐的设计。

关于顺序：我会等 #6137 落地后基于 main 来做——既然它已获批且重构的是同一段 reduce
路径，基于它 rebase 并处理冲突解法的噪声比等待更大。与此同时，我会对照 #6137 的结构和
runtime schedule 勾画重叠设计，在写代码前先分享出来做个合理性检查，并与 @Autumn1998
对齐，让它干净地嵌入新布局。

### @Achyuthan-S 2026-08-20T18:21:54Z

@wujingyue 我对照 #6137 的 reduce 路径勾画了 DP-outer 重叠。我的理解是：最后一个
microbatch 的 finalize（parameter_group.py）跑在 reduce_scatter_stream 上，与 DP-inner
归约共用——而由于两个轴通过 mesh 已经解析到不同的 NCCL communicator，把 finalize 移到
自己的 stream 上应该能让它独立推进，无需单独的 communicator。这符合你的模型吗，还是
存在某条共享 communicator 的路径仍会让它们串行？

 如果成立：在 FsdpContext 上加一个可选的 DP-outer 归约 stream（默认关闭，仿照
 unify_communication_stream），finalize 缓冲的分配/释放留在该 stream 上（不用
 record_stream），并在 optimizer.step() 之前加一次等待。是做成独立开关 + 基准测试，
 还是并入更大的调度改动？

### @wujingyue 2026-08-20T19:18:04Z

我怀疑 dp-outer 必须有独立的 stream，否则 dp-outer 和 dp-inner 通信无法重叠。这个重叠
能带来多少收益还有待测量。轶事说法是相当可观。

### @wujingyue 2026-08-20T19:18:40Z

我会关掉这个 bug，另外开一个关于重叠的 issue。
