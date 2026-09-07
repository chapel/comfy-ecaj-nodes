"""Successful-operation lifetimes, measured before subsequent allocations."""

import weakref

import pytest
import torch

from lib import gpu_ops as gpu
from lib import per_block
from lib import recipe_eval as re
from lib.block_classify import classify_key
from lib.recipe import BlockConfig, RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel
from lib.widen import WIDEN


def owner_ref(tensor):
    while tensor._base is not None:
        tensor = tensor._base
    return weakref.ref(tensor)


# AC: @full-model-execution ac-5, ac-7, ac-14, ac-15
# AC: @memory-management ac-1
@pytest.mark.parametrize("domain", ["diffusion", "clip"])
@pytest.mark.parametrize("kind", ["model", "lora", "mixed"])
@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_compiled_dead_outputs_released_before_next_loader(
    monkeypatch, domain, kind, compose, scaled
):
    key = (
        "diffusion_model.input_blocks.0.0.weight"
        if domain == "diffusion"
        else "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
    )
    config = (
        BlockConfig(arch="sdxl", block_overrides=((classify_key(key, "sdxl", domain), 0.5),))
        if scaled
        else None
    )
    count = 3 if compose else 2
    leaves = [
        RecipeModel(str(i), strength=0.7 if scaled else 1.0, block_config=config)
        if kind == "model" or (kind == "mixed" and i % 2 == 0)
        else RecipeLoRA(({"path": str(i), "strength": 1.0},), block_config=config)
        for i in range(count)
    ]
    base = RecipeBase(object(), "sdxl", domain)
    first = RecipeCompose(tuple(leaves[:-1])) if compose else leaves[0]
    root = RecipeMerge(RecipeMerge(base, first, None, 1.0), leaves[-1], None, 1.0)
    ids = {id(n): str(i) for i, n in enumerate(leaves)}
    plan = re.compile_plan(root, ids, "sdxl", ids)
    refs, cpu_refs, observations = [], [], []

    def inspect(i):
        if i == count - 1:
            observations.append(([r() is None for r in refs], [r() is None for r in cpu_refs]))

    class ModelLoader:
        def __init__(self, i):
            self.i = i

        def get_weights(self, keys):
            inspect(self.i)
            t = torch.full((3, 4), float(self.i + 2))
            cpu_refs.append(owner_ref(t))
            return [t]

    class LoraLoader:
        def get_delta_specs(self, keys, indices, set_id):
            inspect(int(set_id))
            return [
                gpu.DeltaSpec(
                    "standard",
                    0,
                    up=torch.ones(3, 1),
                    down=torch.ones(1, 4),
                    scale=int(set_id) + 1,
                )
            ]

    # Capture the actual tensors consumed by real WIDEN, not register metadata.
    real_filter, real_merge = WIDEN.filter_delta_batched, WIDEN.merge_weights_batched

    def filter_(self, applied, backbone):
        refs.append(owner_ref(applied))
        return real_filter(self, applied, backbone)

    def merge_(self, branches, backbone):
        refs.extend(owner_ref(t) for t in branches)
        return real_merge(self, branches, backbone)

    monkeypatch.setattr(WIDEN, "filter_delta_batched", filter_)
    monkeypatch.setattr(WIDEN, "merge_weights_batched", merge_)
    out = re.execute_plan(
        plan,
        [key],
        torch.ones(1, 3, 4),
        LoraLoader(),
        WIDEN(),
        "cpu",
        torch.float32,
        arch="sdxl",
        model_loaders={str(i): ModelLoader(i) for i in range(count)},
        domain=domain,
    )
    assert observations and observations[0][0]
    assert all(observations[0][0]) and all(observations[0][1])
    # Constant matrices have zero directional divergence; explicit scalar control.
    strength = 0.35 if scaled else 1.0

    def applied(i, current):
        if isinstance(leaves[i], RecipeModel):
            return current + strength * (i + 2 - current)
        return current + (0.5 if scaled else 1.0) * (i + 1)

    initial = sum(applied(i, 1.0) for i in range(count - 1)) / (count - 1)
    torch.testing.assert_close(out, torch.full_like(out, applied(count - 1, initial)))


# AC: @batched-executor ac-2
@pytest.mark.parametrize("first_kind", ["standard", "qkv_q"])
@pytest.mark.parametrize("next_kind", ["standard", "direct", "lokr"])
def test_partition_storage_dies_before_next_allocation(monkeypatch, first_kind, next_kind):
    refs, observed = [], []
    compute = gpu._compute_deltas

    def checked_compute(group, device, dtype):
        observed.append(all(r() is None for r in refs))
        pairs = compute(group, device, dtype)
        refs.extend(owner_ref(t) for _, t in pairs)
        return pairs

    monkeypatch.setattr(gpu, "_compute_deltas", checked_compute)
    # Tensor.to is the first allocation boundary for direct/kron factors.
    sentinel = torch.ones(3, 4)
    real_to = torch.Tensor.to

    def checked_to(self, *args, **kwargs):
        if self is sentinel:
            observed.append(all(r() is None for r in refs))
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", checked_to)
    specs = [
        gpu.DeltaSpec(first_kind, i, up=torch.ones(3, 1), down=torch.ones(1, 4), offset=(0, 3))
        for i in range(2)
    ]
    if next_kind == "standard":
        specs.append(gpu.DeltaSpec("standard", 0, up=torch.ones(3, 2), down=torch.ones(2, 4)))
        extra = 2
    elif next_kind == "direct":
        specs.append(gpu.DeltaSpec("direct", 0, up=sentinel))
        extra = 1
    else:
        specs.append(gpu.DeltaSpec("lokr", 0, w1=sentinel, w2=torch.ones(1, 1)))
        extra = 1
    out = gpu.apply_lora_batch_gpu(["a", "b"], torch.zeros(2, 3, 4), specs, "cpu", torch.float32)
    assert observed == [True, True]
    torch.testing.assert_close(out[0], torch.full((3, 4), 1.0 + extra))
    torch.testing.assert_close(out[1], torch.ones(3, 4))


# AC: @batched-executor ac-2
def test_offset_mlp_dispatches_to_slice():
    spec = gpu.DeltaSpec(
        "offset_mlp", 0, up=torch.ones(2, 1), down=torch.ones(1, 4), offset=(3, 2)
    )
    out = gpu.apply_lora_batch_gpu(["k"], torch.zeros(1, 6, 4), [spec], "cpu", torch.float32)
    expected = torch.zeros_like(out)
    expected[:, 3:5] = 1
    torch.testing.assert_close(out, expected)


# AC: @merge-block-config ac-1
@pytest.mark.parametrize("merge", [False, True])
def test_subgroups_release_indexed_storage(monkeypatch, merge):
    monkeypatch.setattr(per_block, "_get_block_t_factors", lambda *a: {1.0: [0], 2.0: [1]})
    refs, observations = [], []
    real_getitem = torch.Tensor.__getitem__
    base = torch.ones(2, 3, 4)

    def getitem(self, index):
        if isinstance(index, list) and index == [1]:
            observations.append(all(r() is None for r in refs))
        out = real_getitem(self, index)
        if isinstance(index, list) and index == [0]:
            refs.append(owner_ref(out))
        return out

    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    if merge:
        out = per_block._apply_widen_merge_per_block(
            ["a", "b"], [base + 1, base + 3], base, None, "sdxl", 1.0, None
        )
        expected = base + 2
    else:
        out = per_block._apply_widen_filter_per_block(
            ["a", "b"], base + 1, base, None, "sdxl", 1.0, None
        )
        expected = base + 1
    assert observations and all(observations)
    torch.testing.assert_close(out, expected)


# AC: @accurate-ram-preflight ac-2, ac-7
def test_chunk_estimate_keeps_compute_capacity_separate_from_storage(monkeypatch):
    from lib.batch_groups import OpSignature

    calls = []

    def capacity(shape, count, dtype, budget):
        calls.append((shape, count, dtype, budget))
        return 2

    monkeypatch.setattr(gpu, "compute_batch_size", capacity)
    groups = {OpSignature((4,), 1): ["a", "b", "c"]}
    assert (
        gpu.estimate_worst_chunk_bytes(groups, 3, torch.float32, 1.0, storage_dtype=torch.float16)
        == 16
    )
    assert calls == [((4,), 3, torch.float32, 1.0)]
    assert gpu.estimate_worst_chunk_bytes({}, 3, torch.float32) == 0


# AC: @accurate-ram-preflight ac-2, ac-7
@pytest.mark.parametrize("count,capacity,expected", [(1, 1000000, 16), (100, 3, 48), (0, 3, 0)])
def test_chunk_estimate_counts_only_actual_members(monkeypatch, count, capacity, expected):
    from lib.batch_groups import OpSignature

    monkeypatch.setattr(gpu, "compute_batch_size", lambda *a, **kw: capacity)
    sig = OpSignature((4,), 1)
    assert (
        gpu.estimate_worst_chunk_bytes(
            {sig: [str(i) for i in range(count)]}, 2, torch.float32, 1.0
        )
        == expected
    )
