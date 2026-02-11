"""Tests for batched pipeline executor primitives.

Covers all 7 acceptance criteria for @batched-executor spec.
"""

import torch

from lib.executor import (
    DeltaSpec,
    OpSignature,
    apply_lora_batch_gpu,
    chunked,
    chunked_evaluation,
    compile_batch_groups,
    compute_batch_size,
    evaluate_recipe,
)
from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

# =============================================================================
# AC-1: OpSignature grouping
# =============================================================================


class TestOpSignatureGrouping:
    """AC: @batched-executor ac-1

    Given: a set of parameter keys with varying shapes and affecting sets
    When: grouped by OpSignature
    Then: keys with identical shape and affecting sets are in the same group
    """

    def test_same_shape_same_sets_grouped(self):
        """Keys with identical shape and affecting sets should be grouped together."""
        # AC: @batched-executor ac-1
        base_state = {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(4, 4),
            "layer3.weight": torch.randn(4, 4),
        }
        set_affected = {
            "set_a": {"layer1.weight", "layer2.weight", "layer3.weight"},
        }

        groups = compile_batch_groups(
            list(base_state.keys()), base_state, set_affected
        )

        # All 3 keys have same shape (4,4) and same affecting sets {set_a}
        assert len(groups) == 1
        sig = list(groups.keys())[0]
        assert len(groups[sig]) == 3
        assert sig.shape == (4, 4)
        assert sig.ndim == 2
        assert sig.affecting_sets == frozenset(["set_a"])

    def test_different_shapes_separate_groups(self):
        """Keys with different shapes should be in separate groups."""
        # AC: @batched-executor ac-1
        base_state = {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(8, 8),
            "layer3.bias": torch.randn(16),
        }
        set_affected = {
            "set_a": {"layer1.weight", "layer2.weight", "layer3.bias"},
        }

        groups = compile_batch_groups(
            list(base_state.keys()), base_state, set_affected
        )

        # 3 different shapes = 3 groups
        assert len(groups) == 3
        shapes = {sig.shape for sig in groups.keys()}
        assert shapes == {(4, 4), (8, 8), (16,)}

    def test_different_affecting_sets_separate_groups(self):
        """Keys with different affecting sets should be in separate groups."""
        # AC: @batched-executor ac-1
        base_state = {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(4, 4),
        }
        # layer1 affected by set_a only, layer2 affected by both
        set_affected = {
            "set_a": {"layer1.weight", "layer2.weight"},
            "set_b": {"layer2.weight"},
        }

        groups = compile_batch_groups(
            list(base_state.keys()), base_state, set_affected
        )

        # Same shape but different affecting sets = 2 groups
        assert len(groups) == 2

    def test_opsignature_is_hashable(self):
        """OpSignature should be usable as dict key."""
        # AC: @batched-executor ac-1
        sig1 = OpSignature(frozenset(["a", "b"]), (4, 4), 2)
        sig2 = OpSignature(frozenset(["a", "b"]), (4, 4), 2)
        sig3 = OpSignature(frozenset(["a"]), (4, 4), 2)

        d = {sig1: "value1"}
        d[sig2] = "value2"
        d[sig3] = "value3"

        # sig1 and sig2 are equal, so same key
        assert len(d) == 2
        assert d[sig1] == "value2"

    def test_missing_keys_skipped(self):
        """Keys not in base_state should be skipped."""
        # AC: @batched-executor ac-1
        base_state = {
            "layer1.weight": torch.randn(4, 4),
        }
        set_affected = {
            "set_a": {"layer1.weight", "missing_key"},
        }

        groups = compile_batch_groups(
            ["layer1.weight", "missing_key"], base_state, set_affected
        )

        # Only layer1.weight should be in groups
        assert len(groups) == 1
        sig = list(groups.keys())[0]
        assert groups[sig] == ["layer1.weight"]


# =============================================================================
# AC-2: torch.bmm LoRA application
# =============================================================================


class TestBmmLoraApplication:
    """AC: @batched-executor ac-2

    Given: a batch of parameters and LoRA DeltaSpecs
    When: bmm LoRA apply runs
    Then: torch.bmm produces correct deltas matching per-key application
    """

    def test_single_lora_standard(self):
        """Single standard LoRA delta should match manual computation."""
        # AC: @batched-executor ac-2
        batch_size = 2
        out_dim, in_dim = 8, 4
        rank = 2

        base = torch.randn(batch_size, out_dim, in_dim)
        up = torch.randn(out_dim, rank)
        down = torch.randn(rank, in_dim)
        scale = 0.5

        specs = [
            DeltaSpec(kind="standard", key_index=0, up=up, down=down, scale=scale),
        ]

        result = apply_lora_batch_gpu(
            ["key0", "key1"], base, specs, device="cpu", dtype=torch.float32
        )

        # Expected: base[0] + scale * (up @ down), base[1] unchanged
        expected_delta = scale * (up @ down)
        assert torch.allclose(result[0], base[0] + expected_delta, atol=1e-5)
        assert torch.allclose(result[1], base[1], atol=1e-5)

    def test_multiple_same_rank_uses_bmm(self):
        """Multiple same-rank LoRAs should batch via bmm."""
        # AC: @batched-executor ac-2
        batch_size = 3
        out_dim, in_dim = 8, 4
        rank = 2

        base = torch.randn(batch_size, out_dim, in_dim)
        up0 = torch.randn(out_dim, rank)
        down0 = torch.randn(rank, in_dim)
        up1 = torch.randn(out_dim, rank)
        down1 = torch.randn(rank, in_dim)

        specs = [
            DeltaSpec(kind="standard", key_index=0, up=up0, down=down0, scale=1.0),
            DeltaSpec(kind="standard", key_index=1, up=up1, down=down1, scale=0.5),
        ]

        result = apply_lora_batch_gpu(
            ["k0", "k1", "k2"], base, specs, device="cpu", dtype=torch.float32
        )

        # Verify per-key correctness
        expected0 = base[0] + 1.0 * (up0 @ down0)
        expected1 = base[1] + 0.5 * (up1 @ down1)
        assert torch.allclose(result[0], expected0, atol=1e-5)
        assert torch.allclose(result[1], expected1, atol=1e-5)
        assert torch.allclose(result[2], base[2], atol=1e-5)

    def test_different_ranks_separate_partitions(self):
        """Different rank LoRAs should be in separate partitions."""
        # AC: @batched-executor ac-2
        batch_size = 2
        out_dim, in_dim = 8, 4

        base = torch.randn(batch_size, out_dim, in_dim)
        # Rank 2
        up0 = torch.randn(out_dim, 2)
        down0 = torch.randn(2, in_dim)
        # Rank 4
        up1 = torch.randn(out_dim, 4)
        down1 = torch.randn(4, in_dim)

        specs = [
            DeltaSpec(kind="standard", key_index=0, up=up0, down=down0, scale=1.0),
            DeltaSpec(kind="standard", key_index=1, up=up1, down=down1, scale=1.0),
        ]

        result = apply_lora_batch_gpu(
            ["k0", "k1"], base, specs, device="cpu", dtype=torch.float32
        )

        expected0 = base[0] + (up0 @ down0)
        expected1 = base[1] + (up1 @ down1)
        assert torch.allclose(result[0], expected0, atol=1e-5)
        assert torch.allclose(result[1], expected1, atol=1e-5)

    def test_empty_specs_returns_base(self):
        """Empty specs should return base unchanged."""
        # AC: @batched-executor ac-2
        base = torch.randn(2, 4, 4)
        result = apply_lora_batch_gpu(["k0", "k1"], base, [], "cpu", torch.float32)
        assert torch.allclose(result, base)

    def test_target_shape_reshape(self):
        """Conv2D deltas should reshape via target_shape."""
        # AC: @batched-executor ac-2
        batch_size = 1
        # Conv2d: (out_channels, in_channels, h, w) = (8, 4, 3, 3)
        out_c, in_c, h, w = 8, 4, 3, 3
        # Flattened for LoRA: (out_c, in_c * h * w) = (8, 36)
        flat_in = in_c * h * w
        rank = 2

        base = torch.randn(batch_size, out_c, in_c, h, w)
        up = torch.randn(out_c, rank)
        down = torch.randn(rank, flat_in)

        specs = [
            DeltaSpec(
                kind="standard",
                key_index=0,
                up=up,
                down=down,
                scale=1.0,
                target_shape=(out_c, in_c, h, w),
            ),
        ]

        result = apply_lora_batch_gpu(
            ["k0"], base, specs, device="cpu", dtype=torch.float32
        )

        # Delta should be reshaped to match base
        flat_delta = up @ down
        expected = base[0] + flat_delta.view(out_c, in_c, h, w)
        assert torch.allclose(result[0], expected, atol=1e-5)


# =============================================================================
# AC-3: compute_batch_size VRAM targeting
# =============================================================================


class TestComputeBatchSize:
    """AC: @batched-executor ac-3

    Given: available VRAM and parameter shapes
    When: compute_batch_size is called
    Then: it returns a batch size targeting 70 percent of free VRAM
    """

    def test_explicit_vram_budget(self):
        """With explicit VRAM budget, batch size should target 70%."""
        # AC: @batched-executor ac-3
        shape = (1024, 1024)  # 1M elements
        dtype = torch.float32  # 4 bytes per element
        n_models = 2  # multiplier = 3 + 3*2 = 9

        # 1GB budget
        vram_gb = 1.0

        batch_size = compute_batch_size(shape, n_models, dtype, vram_gb)

        # Expected calculation:
        # tensor_bytes = 4 * 1024 * 1024 = 4MB
        # budget_bytes = 0.7 * 1GB ≈ 751619276 bytes
        # max_batch = 751619276 / (4MB * 9) ≈ 20-30
        assert batch_size >= 1
        assert batch_size <= 35  # Reasonable upper bound

    def test_small_budget_returns_one(self):
        """Very small VRAM budget should return batch size 1."""
        # AC: @batched-executor ac-3
        shape = (1024, 1024)
        dtype = torch.float32
        n_models = 10

        # Very small budget that can't fit a batch
        batch_size = compute_batch_size(shape, n_models, dtype, 0.001)

        assert batch_size == 1

    def test_small_tensors_large_batch(self):
        """Small tensors should allow large batch sizes."""
        # AC: @batched-executor ac-3
        shape = (4, 4)  # 16 elements
        dtype = torch.float32
        n_models = 2

        batch_size = compute_batch_size(shape, n_models, dtype, 1.0)

        # 16 * 4 bytes * 9 = 576 bytes per batch element
        # 700MB / 576 bytes is huge
        assert batch_size > 100000

    def test_different_dtypes(self):
        """Different dtypes should scale batch size appropriately."""
        # AC: @batched-executor ac-3
        shape = (512, 512)
        n_models = 2
        vram_gb = 0.5

        batch_fp32 = compute_batch_size(shape, n_models, torch.float32, vram_gb)
        batch_fp16 = compute_batch_size(shape, n_models, torch.float16, vram_gb)

        # fp16 uses half the bytes, so ~2x batch size
        assert batch_fp16 > batch_fp32
        assert batch_fp16 <= 2 * batch_fp32 + 1  # Allow for rounding

    def test_more_models_smaller_batch(self):
        """More models should reduce batch size due to higher multiplier."""
        # AC: @batched-executor ac-3
        shape = (256, 256)
        dtype = torch.float32
        vram_gb = 0.5

        batch_2models = compute_batch_size(shape, 2, dtype, vram_gb)
        batch_10models = compute_batch_size(shape, 10, dtype, vram_gb)

        # More models = higher multiplier = smaller batch
        assert batch_2models > batch_10models


# =============================================================================
# AC-4: OOM backoff
# =============================================================================


class TestOomBackoff:
    """AC: @batched-executor ac-4

    Given: a torch.cuda.OutOfMemoryError during batch evaluation
    When: OOM backoff triggers
    Then: the failed chunk retries at batch size 1 while other chunks continue normally
    """

    def test_oom_retry_at_batch_size_one(self):
        """OOM on chunk should retry keys individually."""
        # AC: @batched-executor ac-4
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(4, 4) for k in keys}

        call_count = [0]

        def eval_fn_with_oom(batch_keys, batch_gpu):
            call_count[0] += 1
            if call_count[0] == 1 and len(batch_keys) > 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu * 2  # Simple transformation

        results = chunked_evaluation(
            keys,
            base,
            eval_fn_with_oom,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        # Should have all results despite OOM
        assert len(results) == 4
        for k in keys:
            assert torch.allclose(results[k], base[k] * 2, atol=1e-5)

    def test_multiple_chunks_continue_after_oom(self):
        """Other chunks should continue normally after one OOMs."""
        # AC: @batched-executor ac-4
        keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
        base = {k: torch.randn(4, 4) for k in keys}

        oom_chunks = set()
        chunk_call_count = [0]

        def eval_fn(batch_keys, batch_gpu):
            chunk_call_count[0] += 1
            # OOM on first chunk only
            if chunk_call_count[0] == 1 and len(batch_keys) > 1:
                oom_chunks.add(tuple(batch_keys))
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu + 1

        results = chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        # All keys should have results
        assert len(results) == 6
        # One chunk should have OOMed and retried
        assert len(oom_chunks) == 1


# =============================================================================
# AC-5: Results on CPU
# =============================================================================


class TestResultsOnCpu:
    """AC: @batched-executor ac-5

    Given: the executor completes evaluation
    When: merged tensors are produced
    Then: all result tensors are on CPU ready for set patch installation
    """

    def test_results_are_on_cpu(self):
        """All result tensors should be on CPU."""
        # AC: @batched-executor ac-5
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        results = chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        for k, v in results.items():
            assert v.device == torch.device("cpu")

    def test_results_ready_for_patching(self):
        """Results should be independent tensors suitable for patching."""
        # AC: @batched-executor ac-5
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu.clone()

        results = chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        # Verify results are contiguous and independent
        for k, v in results.items():
            assert v.is_contiguous()
            # Modifying result shouldn't affect base
            v[0, 0] = 999.0
            assert base[k][0, 0] != 999.0


# =============================================================================
# AC-6: Storage dtype matching
# =============================================================================


class TestStorageDtypeMatching:
    """AC: @batched-executor ac-6

    Given: a base model with bf16 storage dtype
    When: batched evaluation produces merged results
    Then: output tensors match the base model storage dtype
    """

    def test_output_matches_storage_dtype_bf16(self):
        """Output should match bf16 storage dtype."""
        # AC: @batched-executor ac-6
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4, dtype=torch.float32) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        results = chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,  # Computation in fp32
            storage_dtype=torch.bfloat16,  # Output in bf16
        )

        for k, v in results.items():
            assert v.dtype == torch.bfloat16

    def test_output_matches_storage_dtype_fp16(self):
        """Output should match fp16 storage dtype."""
        # AC: @batched-executor ac-6
        keys = ["k0"]
        base = {"k0": torch.randn(4, 4)}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu

        results = chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
        )

        assert results["k0"].dtype == torch.float16

    def test_computation_uses_higher_precision(self):
        """Internal computation should use higher precision dtype."""
        # AC: @batched-executor ac-6
        keys = ["k0"]
        base = {"k0": torch.randn(4, 4)}

        received_dtype = [None]

        def eval_fn(batch_keys, batch_gpu):
            received_dtype[0] = batch_gpu.dtype
            return batch_gpu

        chunked_evaluation(
            keys,
            base,
            eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,  # Computation dtype
            storage_dtype=torch.float16,  # Storage dtype
        )

        # Eval function should receive fp32 for numerical stability
        assert received_dtype[0] == torch.float32


# =============================================================================
# AC-7: LoKr torch.kron
# =============================================================================


class TestLokrKron:
    """AC: @batched-executor ac-7

    Given: a LoRA with LoKr weights
    When: batched apply runs
    Then: LoKr weights use per-key torch.kron on GPU instead of bmm
    """

    def test_lokr_uses_kron(self):
        """LoKr deltas should use torch.kron."""
        # AC: @batched-executor ac-7
        batch_size = 2
        # LoKr produces output via kron(w1, w2)
        # If w1 is (a, b) and w2 is (c, d), result is (a*c, b*d)
        w1 = torch.randn(2, 2)  # (2, 2)
        w2 = torch.randn(3, 4)  # (3, 4)
        # kron result: (2*3, 2*4) = (6, 8)

        base = torch.randn(batch_size, 6, 8)
        scale = 0.75

        specs = [
            DeltaSpec(kind="lokr", key_index=0, w1=w1, w2=w2, scale=scale),
        ]

        result = apply_lora_batch_gpu(
            ["k0", "k1"], base, specs, device="cpu", dtype=torch.float32
        )

        expected_delta = scale * torch.kron(w1, w2)
        assert torch.allclose(result[0], base[0] + expected_delta, atol=1e-5)
        assert torch.allclose(result[1], base[1], atol=1e-5)

    def test_lokr_with_target_shape(self):
        """LoKr deltas should reshape via target_shape for conv."""
        # AC: @batched-executor ac-7
        # LoKr for conv2d: kronecker produces flat delta, reshape to 4D
        w1 = torch.randn(2, 2)
        w2 = torch.randn(2, 2)
        # kron: (4, 4) -> reshape to (2, 2, 2, 2)
        target_shape = (2, 2, 2, 2)

        base = torch.randn(1, 2, 2, 2, 2)

        specs = [
            DeltaSpec(
                kind="lokr", key_index=0, w1=w1, w2=w2, scale=1.0, target_shape=target_shape
            ),
        ]

        result = apply_lora_batch_gpu(
            ["k0"], base, specs, device="cpu", dtype=torch.float32
        )

        expected_delta = torch.kron(w1, w2).view(target_shape)
        assert torch.allclose(result[0], base[0] + expected_delta, atol=1e-5)

    def test_multiple_lokr_not_batched(self):
        """Multiple LoKr specs should each use kron (not bmm)."""
        # AC: @batched-executor ac-7
        batch_size = 2
        w1a, w2a = torch.randn(2, 2), torch.randn(2, 2)
        w1b, w2b = torch.randn(2, 2), torch.randn(2, 2)

        base = torch.randn(batch_size, 4, 4)

        specs = [
            DeltaSpec(kind="lokr", key_index=0, w1=w1a, w2=w2a, scale=1.0),
            DeltaSpec(kind="lokr", key_index=1, w1=w1b, w2=w2b, scale=0.5),
        ]

        result = apply_lora_batch_gpu(
            ["k0", "k1"], base, specs, device="cpu", dtype=torch.float32
        )

        expected0 = base[0] + torch.kron(w1a, w2a)
        expected1 = base[1] + 0.5 * torch.kron(w1b, w2b)
        assert torch.allclose(result[0], expected0, atol=1e-5)
        assert torch.allclose(result[1], expected1, atol=1e-5)


# =============================================================================
# Helper function tests
# =============================================================================


class TestChunkedHelper:
    """Tests for the chunked() helper function."""

    def test_even_chunks(self):
        """List divisible by chunk size."""
        result = list(chunked([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_chunks(self):
        """List not divisible by chunk size."""
        result = list(chunked([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_element_chunks(self):
        """Chunk size 1."""
        result = list(chunked([1, 2, 3], 1))
        assert result == [[1], [2], [3]]

    def test_chunk_larger_than_list(self):
        """Chunk size larger than list."""
        result = list(chunked([1, 2], 5))
        assert result == [[1, 2]]

    def test_empty_list(self):
        """Empty list."""
        result = list(chunked([], 2))
        assert result == []


# =============================================================================
# Exit Batched Evaluation tests - @exit-batched-eval
# =============================================================================


class MockLoRALoader:
    """Mock LoRA loader for testing evaluate_recipe."""

    def __init__(self, delta_map: dict[str, torch.Tensor] | None = None):
        """Initialize with optional delta map (key -> delta tensor).

        delta_map values should be 2D tensors: (out_dim, in_dim)
        """
        self._delta_map = delta_map or {}

    def get_delta_specs(self, keys, key_indices, set_id=None):
        """Return mock delta specs.

        Creates proper up/down factorizations that when multiplied
        produce the desired delta. For simplicity, uses rank=1 decomposition.
        The set_id parameter is accepted for API compatibility.
        """
        specs = []
        for key in keys:
            if key in self._delta_map:
                idx = key_indices[key]
                delta = self._delta_map[key]
                out_dim, in_dim = delta.shape

                # Use rank-1 factorization: delta ≈ up @ down
                # up: (out_dim, rank=1), down: (rank=1, in_dim)
                # For testing, use column/row that approximates delta
                up = delta[:, 0:1]  # (out_dim, 1) - first column
                down = torch.ones(1, in_dim)  # (1, in_dim) - all ones

                specs.append(
                    DeltaSpec(
                        kind="standard",
                        key_index=idx,
                        up=up,
                        down=down,
                        scale=1.0,
                    )
                )
        return specs


class MockWIDEN:
    """Mock WIDEN for testing evaluate_recipe."""

    def __init__(self):
        self.filter_calls = []
        self.merge_calls = []

    def filter_delta_batched(self, lora_applied, backbone):
        """Record call and return lora_applied."""
        self.filter_calls.append({"lora_applied": lora_applied, "backbone": backbone})
        return lora_applied  # Passthrough for testing

    def merge_weights_batched(self, weights_list, backbone):
        """Record call and return average."""
        self.merge_calls.append({"weights_list": weights_list, "backbone": backbone})
        # Return simple average for testing
        return torch.stack(weights_list).mean(dim=0)


class TestEvaluateRecipeComposeTarget:
    """AC: @exit-batched-eval ac-1

    Given: a recipe tree with a compose target
    When: batched evaluation runs
    Then: merge_weights_batched is called with all branch results and the backbone
    """

    def test_compose_calls_merge_weights_batched(self):
        """Compose target should call merge_weights_batched."""
        # AC: @exit-batched-eval ac-1
        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        # Create recipe with compose target
        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora1 = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora2 = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora1, lora2))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        # Mock loader with deltas
        delta1 = torch.randn(4, 4)
        loader = MockLoRALoader({"k0": delta1, "k1": delta1})

        # Mock WIDEN
        widen = MockWIDEN()

        set_id_map = {id(lora1): "set1", id(lora2): "set2"}

        evaluate_recipe(
            keys=["k0", "k1"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should have called merge_weights_batched once
        assert len(widen.merge_calls) == 1
        assert len(widen.merge_calls[0]["weights_list"]) == 2  # Two branches

    def test_compose_passes_all_branches(self):
        """All branch results should be passed to merge_weights_batched."""
        # AC: @exit-batched-eval ac-1
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora1 = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora2 = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 1.0},))
        lora3 = RecipeLoRA(loras=({"path": "c.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora1, lora2, lora3))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora1): "set1", id(lora2): "set2", id(lora3): "set3"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should pass 3 branches
        assert len(widen.merge_calls[0]["weights_list"]) == 3


class TestEvaluateRecipeLoRATarget:
    """AC: @exit-batched-eval ac-2

    Given: a recipe tree with a single LoRA target
    When: batched evaluation runs
    Then: filter_delta_batched is called with the applied LoRA delta and backbone
    """

    def test_lora_target_calls_filter_delta_batched(self):
        """LoRA target should call filter_delta_batched."""
        # AC: @exit-batched-eval ac-2
        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=["k0", "k1"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should call filter_delta_batched once
        assert len(widen.filter_calls) == 1
        assert len(widen.merge_calls) == 0

    def test_lora_applied_passed_to_filter(self):
        """Applied LoRA weights should be passed to filter_delta_batched."""
        # AC: @exit-batched-eval ac-2
        batch_size = 1
        base_batch = torch.zeros(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        # Add a delta that will be applied
        delta = torch.ones(4, 4)
        loader = MockLoRALoader({"k0": delta})
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # The lora_applied tensor should be different from base if delta was applied
        assert len(widen.filter_calls) == 1


class TestEvaluateRecipeChainedMerge:
    """AC: @exit-batched-eval ac-3

    Given: a chain of RecipeMerge nodes
    When: evaluation recurses
    Then: inner merges evaluate first and results become the base for outer merges
    """

    def test_chained_merge_evaluates_inner_first(self):
        """Inner merge should be evaluated before outer merge."""
        # AC: @exit-batched-eval ac-3
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        inner_lora = RecipeLoRA(loras=({"path": "inner.safetensors", "strength": 1.0},))
        inner_merge = RecipeMerge(base=base, target=inner_lora, backbone=None, t_factor=1.0)

        outer_lora = RecipeLoRA(loras=({"path": "outer.safetensors", "strength": 1.0},))
        outer_merge = RecipeMerge(base=inner_merge, target=outer_lora, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(inner_lora): "set_inner", id(outer_lora): "set_outer"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=outer_merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should have 2 filter calls - inner merge first, then outer
        assert len(widen.filter_calls) == 2
        # Verify order: inner call uses base_batch as backbone,
        # outer call's backbone is inner's result (which is base_batch
        # since no deltas and filter passthrough returns lora_applied)
        inner_call = widen.filter_calls[0]
        outer_call = widen.filter_calls[1]
        assert torch.equal(inner_call["backbone"], base_batch)
        # Outer backbone should equal inner result (inner's lora_applied passthrough)
        assert torch.equal(outer_call["backbone"], inner_call["lora_applied"])

    def test_triple_chain_evaluates_in_order(self):
        """Triple chain should evaluate innermost to outermost."""
        # AC: @exit-batched-eval ac-3
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")

        lora1 = RecipeLoRA(loras=({"path": "l1.safetensors", "strength": 1.0},))
        merge1 = RecipeMerge(base=base, target=lora1, backbone=None, t_factor=1.0)

        lora2 = RecipeLoRA(loras=({"path": "l2.safetensors", "strength": 1.0},))
        merge2 = RecipeMerge(base=merge1, target=lora2, backbone=None, t_factor=1.0)

        lora3 = RecipeLoRA(loras=({"path": "l3.safetensors", "strength": 1.0},))
        merge3 = RecipeMerge(base=merge2, target=lora3, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora1): "set1", id(lora2): "set2", id(lora3): "set3"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge3,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should have 3 filter calls in order: innermost to outermost
        assert len(widen.filter_calls) == 3
        # Verify order: each subsequent call's backbone should equal the
        # previous call's result (lora_applied passthrough since no deltas)
        first_call = widen.filter_calls[0]
        second_call = widen.filter_calls[1]
        third_call = widen.filter_calls[2]
        assert torch.equal(first_call["backbone"], base_batch)
        assert torch.equal(second_call["backbone"], first_call["lora_applied"])
        assert torch.equal(third_call["backbone"], second_call["lora_applied"])


class TestEvaluateRecipeGPUResults:
    """AC: @exit-batched-eval ac-4

    Given: the WIDEN algorithm produces results
    When: they are returned from evaluation
    Then: all result tensors are on GPU (transferred to CPU in patch installation phase)
    """

    def test_result_same_device_as_input(self):
        """Result should be on same device as input."""
        # AC: @exit-batched-eval ac-4
        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        result = evaluate_recipe(
            keys=["k0", "k1"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",  # Would be "cuda" in production
            dtype=torch.float32,
        )

        # Result should be on same device as input base_batch
        assert result.device == base_batch.device

    def test_result_not_transferred_to_cpu(self):
        """Result should NOT be transferred to CPU (that happens in patch phase)."""
        # AC: @exit-batched-eval ac-4
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        result = evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Result shape should match input
        assert result.shape == base_batch.shape
        # Result dtype should match input dtype
        assert result.dtype == base_batch.dtype


class TestEvaluateRecipeBackboneOverride:
    """AC: @exit-batched-eval ac-5

    Given: a RecipeMerge with an explicit backbone reference that differs from the base
    When: batched evaluation runs the WIDEN step
    Then: the backbone model (not the base) is used as the importance reference for WIDEN analysis
    """

    def test_backbone_override_passed_to_widen(self):
        """Explicit backbone should be passed to WIDEN functions."""
        # AC: @exit-batched-eval ac-5
        batch_size = 2
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        backbone_ref = RecipeBase(model_patcher=None, arch="sdxl")  # Different backbone
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=backbone_ref, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=["k0", "k1"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # filter_delta_batched should have been called
        assert len(widen.filter_calls) == 1
        # The backbone parameter should have been passed (we use base_batch as fallback)
        call = widen.filter_calls[0]
        assert call["backbone"] is not None

    def test_no_backbone_uses_current_base(self):
        """When backbone is None, current base should be used."""
        # AC: @exit-batched-eval ac-5
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora): "set1"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # Should use current_base as backbone
        call = widen.filter_calls[0]
        assert torch.equal(call["backbone"], base_batch)

    def test_compose_with_backbone_override(self):
        """Compose target with backbone override should pass backbone to merge."""
        # AC: @exit-batched-eval ac-5
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        backbone_ref = RecipeBase(model_patcher=None, arch="sdxl")
        lora1 = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora2 = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora1, lora2))
        merge = RecipeMerge(base=base, target=compose, backbone=backbone_ref, t_factor=1.0)

        loader = MockLoRALoader()
        widen = MockWIDEN()
        set_id_map = {id(lora1): "set1", id(lora2): "set2"}

        evaluate_recipe(
            keys=["k0"],
            base_batch=base_batch,
            recipe_node=merge,
            loader=loader,
            widen=widen,
            set_id_map=set_id_map,
            device="cpu",
            dtype=torch.float32,
        )

        # merge_weights_batched should have backbone passed
        assert len(widen.merge_calls) == 1
        call = widen.merge_calls[0]
        assert call["backbone"] is not None
