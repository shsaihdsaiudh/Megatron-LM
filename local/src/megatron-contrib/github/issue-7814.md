# Issue #7814: MFSDP v2: static prefetch order does not represent repeated module calls

- URL: https://github.com/NVIDIA/Megatron-LM/issues/7814
- Author: @wujingyue  State: open
- Created: 2026-10-02T02:17:15Z

## Body

**Describe the bug**

MFSDP v2 derives its prefetch schedule from a static order of unique module objects. With `mtp_use_repeated_layer=True`, the same FSDP units execute at multiple prediction depths, so this order misses transitions between depths.

For a shared block and output head, ignoring the root:

```text
Static forward order: block -> head
Actual forward:       block(depth 1) -> head(depth 1)
                   -> block(depth 2) -> head(depth 2)
```

After `head(depth 1)`, static successor lookup returns no next module. It therefore does not prefetch `block(depth 2)`, which gathers its weights on demand.

In the ordinary shared-MTP pattern tested with #7757's backward-entry guard, this is a missed prefetch opportunity. Five-step losses still match the unsharded reference, and no unused full-weight buffers remain after backward. During backward, shared weights stay gathered until both uses contribute their gradients. This observation does not establish correctness for arbitrary reuse or recomputation schedules.

**Steps/Code to reproduce bug**

A forward pass is sufficient to expose the missed prefetch. Save the following as `repro_static_prefetch.py` in a Megatron-LM checkout:

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
            # Each depth consumes shifted tokens and the preceding hidden states.
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

Run on two GPUs:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 repro_static_prefetch.py
```

Observed output:

```text
Static order: ['<root>', 'block', 'head']
(module, already prefetched): [('block', True), ('head', True), ('block', False), ('head', True)]
```

The second block invocation is not prefetched. Its normal MFSDP pre-forward hook must gather its weights on demand.

Verified on two RTX A6000 GPUs with PyTorch `2.13.0a0+8145d630e8.nv26.06`, using the unchanged MFSDP implementation from main commit `05fb87bcd050e83ec89c13937df9a59041f168e5`. CUDA graphs are disabled. No step-time impact has been quantified yet.

**Expected behavior**

Prefetch should account for the next module invocation, including transitions between repeated MTP depths, while respecting its memory budget and retaining on-demand gathering as a correctness fallback.

**Additional context**

The code already flags this limitation:

- [The `FsdpContext` comment](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L54-L58) suggests “capturing runtime order if static module order proves too fragile.”
- [`finalize()`](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L102-L125) builds the orders from the module hierarchy; [`IndexedOrder`](https://github.com/NVIDIA/Megatron-LM/blob/05fb87bcd050e83ec89c13937df9a59041f168e5/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/indexed_order.py#L32-L59) rejects duplicate module objects and looks up a fixed successor.
- [The guard comment in #7757](https://github.com/wujingyue/Megatron-LM/blob/2b5893ed4ea30b1abbd22a1393bc5ffc88bbfa4b/megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py#L499-L521) identifies repeated-invocation prefetch as work needed before separating per-invocation weight release from gradient readiness.

Related: #7764 already discusses this scheduling limitation within the broader backward-hook redesign. This report isolates the currently observable missed forward prefetch. #7809 addresses a different mismatch by correcting `ShortcutMoEBlock` registration order; registration reordering cannot represent repeated invocations of the same module.


## Conversation
