# Issue #7814：MFSDP v2：静态预取顺序无法表达模块的重复调用

- URL: https://github.com/NVIDIA/Megatron-LM/issues/7814
- 作者：@wujingyue  状态：open
- 创建时间：2026-10-02T02:17:15Z

## 正文

**描述 bug**

MFSDP v2 依据"唯一模块对象"的静态顺序推导预取调度。当
`mtp_use_repeated_layer=True` 时，同一个 FSDP unit 会在多个预测深度上执行，因此这个
顺序会漏掉各深度之间的切换。

对于共享 block 和输出 head（忽略 root）：

```text
Static forward order: block -> head
Actual forward:       block(depth 1) -> head(depth 1)
                   -> block(depth 2) -> head(depth 2)
```

在 `head(depth 1)` 之后，静态后继查找找不到下一个模块，因此不会预取 `block(depth 2)`，
后者的权重只能按需 gather。

在配合 #7757 的 backward-entry guard 测试过的普通共享 MTP 模式下，这只是一个错过了
的预取机会。五步的 loss 仍与不分片（unsharded）的参考实现一致，反向结束后也没有残留
未使用的全权重缓冲。反向过程中，共享权重会一直保持已 gather 状态，直到两个使用点都
贡献完梯度。这一观察并不能证明确切的正确性对任意复用或重计算调度都成立。

**复现步骤 / 代码**

仅一次前向就足以暴露这个漏掉的预取。把下面内容保存为 `repro_static_prefetch.py`，
放在 Megatron-LM 的 checkout 中：

```python
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)


class SharedMTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.head = nn.Linear(8, 32)

    def forward(self, tokens, hidden):
        logits = []
        length = hidden.size(1)
        for depth in (1, 2):
            # 每个深度消费偏移后的 token 和前一个深度的 hidden states。
            hidden = hidden + self.embedding(tokens[:, depth : depth + length])
            hidden = hidden + self.block(hidden)
            logits.append(self.head(hidden))
        return tuple(logits)


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
torch.manual_seed(1234)
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
placements = Placements(
    dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
)
model = SharedMTP().to(device)
with fully_shard_context(device=device) as context:
    fully_shard(model.block, mesh=mesh, placements=placements)
    fully_shard(model.head, mesh=mesh, placements=placements)
    fully_shard(model, mesh=mesh, placements=placements)

trace = []


def record_prefetch(module, inputs):
    # Run before MFSDP's hook can gather the weights on demand.
    trace.append((module.name, module._unshard_event is not None))


model.block.register_forward_pre_hook(record_prefetch, prepend=True)
model.head.register_forward_pre_hook(record_prefetch, prepend=True)
tokens = torch.randint(32, (2, 6), device=device)
hidden = torch.randn(2, 4, 8, device=device)
model(tokens, hidden)
torch.cuda.synchronize(device)
if dist.get_rank() == 0:
    print("Static order:", [unit.name or "<root>" for unit in context.forward_order])
    print("(module, already prefetched):", trace)
dist.destroy_process_group()
```

在两卡上运行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 repro_static_prefetch.py
```

观察到的输出：

```text
Static order: ['<root>', 'block', 'head']
(module, already prefetched): [('block', True), ('head', True), ('block', False), ('head', True)]
```

第二次 block 调用没有被预取。它正常的 MFSDP pre-forward hook 只能按需 gather 权重。

已在两张 RTX A6000、PyTorch `2.13.0a0+8145d630e8.nv26.06` 上验证，使用 main 提交
`05fb87bcd050e83ec89c13937df9a59041f168e5` 中未改动的 MFSDP 实现。CUDA graphs 处于
关闭状态。尚未量化对 step 时间的影响。

**期望行为**

预取应考虑到下一次模块调用，包括重复 MTP 深度之间的切换，同时遵守其内存预算，并保留
按需 gather 作为正确性兜底。

**补充信息**

代码里已经标注了这个限制：

- [The `FsdpContext` comment](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L54-L58) 提出"capturing runtime order if static module order proves too fragile"。
- [`finalize()`](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L102-L125) 从模块层级构造顺序；[`IndexedOrder`](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/indexed_order.py#L32-L59) 拒绝重复的模块对象，并按固定的后继做查找。
- [#7757 中的 guard 注释](https://github.com/wujingyue/Megatron-LM/blob/2b5893ed4ea30b1abbd22a1393bc5ffc88bbfa4b/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L499-L521) 指出，在把"按调用释放权重"与"梯度就绪"解耦之前，必须先做重复调用的预取。

相关：#7764 已经在更大的 backward-hook 重构中讨论了这个调度限制。本报告只隔离出当前
可观察到的、前向预取缺失的问题。#7809 通过修正 `ShortcutMoEBlock` 的注册顺序解决了
另一个不匹配问题；仅靠调整注册顺序无法表达同一模块的重复调用。


## 讨论
