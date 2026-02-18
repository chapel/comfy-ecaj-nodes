"""Tests for GPU memory management during merge execution.

Covers acceptance criteria for @memory-management spec.
"""

import gc
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open

from lib.batch_groups import OpSignature
from lib.executor import (
    chunked_evaluation,
    compile_batch_groups,
    compute_batch_size,
)
from lib.recipe import (
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
)
from lib.recipe_eval import EvalPlan
from nodes.exit import (
    WIDENExitNode,
    _incremental_cache,
)

# ===========================================================================
# Shared helper: build mocks for WIDENExitNode.execute()
# ===========================================================================

def _make_exit_mocks(mock_model_patcher, keys_to_process, *, recipe=None, arch="sdxl"):
    """Create mocks for WIDENExitNode.execute() — adapted from test_incremental_recompute."""
    mock_loader = MagicMock()
    mock_loader.affected_keys = set(keys_to_process)
    mock_loader.affected_keys_for_set = MagicMock(return_value=set(keys_to_process))
    mock_loader.cleanup = MagicMock()

    set_affected = {}
    if recipe is not None:
        def _find_loras(n):
            if isinstance(n, RecipeLoRA):
                set_affected[str(id(n))] = set(keys_to_process)
            elif isinstance(n, RecipeCompose):
                for b in n.branches:
                    _find_loras(b)
            elif isinstance(n, RecipeMerge):
                _find_loras(n.base)
                _find_loras(n.target)
                if n.backbone is not None:
                    _find_loras(n.backbone)
        _find_loras(recipe)
    if not set_affected:
        set_affected = {str(id(None)): set(keys_to_process)}

    mock_analyze = MagicMock(
        model_patcher=mock_model_patcher,
        arch=arch,
        loader=mock_loader,
        set_affected=set_affected,
        affected_keys=set(keys_to_process),
    )
    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()

    dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

    return mock_analyze, mock_model_analysis, mock_loader, dummy_plan


def _run_exit_node(recipe, mock_model_patcher, keys, *, extra_patches=None, **execute_kwargs):
    """Run WIDENExitNode.execute() with full mocking. Returns (result, mocks_dict)."""
    mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
        mock_model_patcher, keys, recipe=recipe,
    )
    merged = {k: torch.randn(4, 4) for k in keys}
    sig = OpSignature(shape=(4, 4), ndim=2)

    patches = {
        "nodes.exit.analyze_recipe": mock_analyze,
        "nodes.exit.analyze_recipe_models": mock_model_analysis,
        "nodes.exit.compile_plan": dummy_plan,
        "nodes.exit.compile_batch_groups": {sig: keys},
        "nodes.exit.chunked_evaluation": merged,
        "nodes.exit.compute_base_identity": "base_id",
        "nodes.exit.compute_lora_stats": {},
    }
    if extra_patches:
        patches.update(extra_patches)

    ctx_managers = []
    mock_refs = {}
    for target, value in patches.items():
        if isinstance(value, MagicMock) and hasattr(value, "return_value"):
            # Already a MagicMock used as return_value
            p = patch(target, return_value=value)
        elif isinstance(value, dict) or isinstance(value, EvalPlan):
            p = patch(target, return_value=value)
        elif isinstance(value, str):
            p = patch(target, return_value=value)
        elif isinstance(value, MagicMock):
            p = patch(target, value)
        else:
            p = patch(target, return_value=value)
        ctx_managers.append((target, p))

    # Enter all patches
    entered = []
    try:
        for target, p in ctx_managers:
            m = p.start()
            mock_refs[target] = m
            entered.append(p)

        node = WIDENExitNode()
        result = node.execute(recipe, **execute_kwargs)
    finally:
        for p in entered:
            p.stop()

    return result, {
        "mock_loader": mock_loader,
        "mock_analyze": mock_analyze,
        "mock_model_analysis": mock_model_analysis,
        "merged": merged,
    }

# =============================================================================
# AC-1: Per-chunk GPU tensor cleanup
# =============================================================================


class TestPerChunkCleanup:
    """AC: @memory-management ac-1

    Given: batched evaluation processes a chunk of parameters
    When: the chunk completes and results transfer to CPU
    Then: all GPU tensors for that chunk are deleted and freed
    """

    def test_no_gc_collect_per_chunk_in_normal_path(self):
        """gc.collect() should NOT be called per-chunk in normal (non-OOM) path.

        GPU tensors are freed via explicit del statements. gc.collect runs
        after all OpSignature groups complete (in exit.py), not per-chunk,
        to avoid GPU sync overhead that blocks kernel queuing.
        """
        # AC: @memory-management ac-1
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        gc_calls = []
        original_gc_collect = gc.collect

        def mock_gc_collect():
            gc_calls.append("gc.collect")
            return original_gc_collect()

        with patch("lib.executor.gc.collect", mock_gc_collect):
            chunked_evaluation(
                keys,
                base,
                eval_fn,
                batch_size=2,  # 2 chunks of 2
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        # No gc.collect calls in normal path (cleanup is after all groups in exit.py)
        assert len(gc_calls) == 0

    def test_no_empty_cache_per_chunk_in_normal_path(self):
        """torch.cuda.empty_cache() should NOT be called per-chunk in normal path."""
        # AC: @memory-management ac-1
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu

        empty_cache_calls = []

        def mock_empty_cache():
            empty_cache_calls.append("empty_cache")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache", mock_empty_cache):
                chunked_evaluation(
                    keys,
                    base,
                    eval_fn,
                    batch_size=2,
                    device="cpu",
                    dtype=torch.float32,
                    storage_dtype=torch.float32,
                )

        # No empty_cache calls in normal path
        assert len(empty_cache_calls) == 0

    def test_results_on_cpu_after_cleanup(self):
        """All result tensors should be on CPU after cleanup."""
        # AC: @memory-management ac-1
        keys = ["k0", "k1", "k2"]
        base = {k: torch.randn(8, 8) for k in keys}

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

        for key, tensor in results.items():
            assert tensor.device == torch.device("cpu")
            assert tensor.is_contiguous()

    def test_cleanup_in_oom_path(self):
        """Cleanup should also happen in OOM retry path."""
        # AC: @memory-management ac-1
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        call_count = [0]

        def eval_fn_with_oom(batch_keys, batch_gpu):
            call_count[0] += 1
            if call_count[0] == 1 and len(batch_keys) > 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu * 2

        gc_calls = []
        original_gc_collect = gc.collect

        def mock_gc_collect():
            gc_calls.append("gc.collect")
            return original_gc_collect()

        with patch("lib.executor.gc.collect", mock_gc_collect):
            results = chunked_evaluation(
                keys,
                base,
                eval_fn_with_oom,
                batch_size=2,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        # Should have gc.collect calls in OOM path too
        assert len(gc_calls) >= 1
        assert len(results) == 2


# =============================================================================
# AC-2: Between-group GC cycles
# =============================================================================


class TestBetweenGroupCleanup:
    """AC: @memory-management ac-2

    Given: an OpSignature group completes all chunks
    When: all groups complete
    Then: gc.collect() and torch.cuda.empty_cache() are called once after all
          groups complete (OOM backoff handles per-group memory pressure)
    """

    def test_multiple_groups_produce_distinct_signatures(self):
        """Different shapes produce distinct OpSignature groups."""
        # AC: @memory-management ac-2
        # Create parameters with different shapes to get multiple groups
        base_state = {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(8, 8),  # Different shape = different group
        }

        groups = compile_batch_groups(list(base_state.keys()), base_state)

        # Should have 2 groups due to different shapes
        assert len(groups) == 2

    def test_exit_node_calls_gc_after_groups(self, mock_model_patcher):
        """Exit node should call gc.collect after all OpSignature groups complete."""
        # AC: @memory-management ac-2
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        gc_calls = []
        original_gc = gc.collect

        def tracking_gc(*a, **kw):
            gc_calls.append(True)
            return original_gc(*a, **kw)

        with patch("nodes.exit.gc.collect", tracking_gc):
            _run_exit_node(recipe, mock_model_patcher, keys)

        assert len(gc_calls) >= 1, "gc.collect should be called after groups complete"


# =============================================================================
# AC-3: Loader resource cleanup
# =============================================================================


class TestLoaderCleanup:
    """AC: @memory-management ac-3

    Given: all LoRA files have been loaded and evaluation is complete
    When: cleanup runs
    Then: all loader resources are freed including delta caches and file handles
    """

    def test_sdxl_loader_cleanup_clears_data(self):
        """SDXLLoader.cleanup() should clear delta caches."""
        # AC: @memory-management ac-3
        from lib.lora.sdxl import SDXLLoader

        loader = SDXLLoader()
        # Simulate loaded data (using per-set storage)
        loader._lora_data_by_set["__default__"]["test_key"] = [
            (torch.randn(4, 4), torch.randn(4, 4), 1.0)
        ]
        loader._affected.add("test_key")

        assert len(loader._lora_data_by_set) > 0
        assert len(loader._affected) > 0

        loader.cleanup()

        assert len(loader._lora_data_by_set) == 0
        assert len(loader._affected) == 0

    def test_zimage_loader_cleanup_clears_data(self):
        """ZImageLoader.cleanup() should clear all delta caches including QKV."""
        # AC: @memory-management ac-3
        from lib.lora.zimage import ZImageLoader

        loader = ZImageLoader()
        # Simulate loaded data (using per-set storage)
        loader._lora_data_by_set["__default__"]["test_key"] = [
            (torch.randn(4, 4), torch.randn(4, 4), 1.0)
        ]
        loader._qkv_data_by_set["__default__"]["qkv_key"] = [
            (torch.randn(4, 4), torch.randn(4, 4), 1.0, "q")
        ]
        loader._affected.add("test_key")
        loader._affected.add("qkv_key")

        assert len(loader._lora_data_by_set) > 0
        assert len(loader._qkv_data_by_set) > 0
        assert len(loader._affected) > 0

        loader.cleanup()

        assert len(loader._lora_data_by_set) == 0
        assert len(loader._qkv_data_by_set) == 0
        assert len(loader._affected) == 0

    def test_loader_context_manager_calls_cleanup(self):
        """Loader context manager should call cleanup on exit."""
        # AC: @memory-management ac-3
        from lib.lora.sdxl import SDXLLoader

        cleanup_called = []
        original_cleanup = SDXLLoader.cleanup

        def mock_cleanup(self):
            cleanup_called.append(True)
            original_cleanup(self)

        with patch.object(SDXLLoader, "cleanup", mock_cleanup):
            with SDXLLoader() as loader:
                loader._lora_data_by_set["__default__"]["test"] = []

        assert len(cleanup_called) == 1

    def test_exit_node_cleanup_in_finally(self, mock_model_patcher):
        """Exit node should call loader.cleanup() even when evaluation raises."""
        # AC: @memory-management ac-3
        _incremental_cache.clear()

        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", side_effect=RuntimeError("boom")),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight"),
        ):
            node = WIDENExitNode()
            with pytest.raises(RuntimeError, match="boom"):
                node.execute(recipe)

        mock_loader.cleanup.assert_called_once()
        _incremental_cache.clear()


# =============================================================================
# AC-4: CPU-only final patches
# =============================================================================


class TestCpuOnlyPatches:
    """AC: @memory-management ac-4

    Given: the complete merge execution
    When: final merged patches are produced
    Then: all patch tensors are on CPU with no GPU tensor references remaining
    """

    def test_install_merged_patches_returns_cpu_tensors(self):
        """install_merged_patches should produce CPU-only tensors."""
        # AC: @memory-management ac-4
        from nodes.exit import install_merged_patches

        # Create mock model patcher
        mock_patcher = MagicMock()
        mock_patcher.model_state_dict.return_value = {
            "diffusion_model.layer.weight": torch.randn(4, 4, dtype=torch.float16)
        }
        mock_clone = MagicMock()
        mock_patcher.clone.return_value = mock_clone

        patches_received = {}

        def capture_patches(patches, strength_patch):
            patches_received.update(patches)

        mock_clone.add_patches = capture_patches

        # Merged state with CPU tensors (simulating output from chunked_evaluation)
        merged_state = {
            "layer.weight": torch.randn(4, 4, dtype=torch.float32),
        }

        install_merged_patches(mock_patcher, merged_state, torch.float16)

        # Verify patches are CPU tensors
        for key, (patch_type, (tensor,)) in patches_received.items():
            assert tensor.device == torch.device("cpu")

    def test_chunked_evaluation_returns_cpu_tensors(self):
        """chunked_evaluation should return tensors on CPU."""
        # AC: @memory-management ac-4
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

        for key, tensor in results.items():
            assert tensor.device == torch.device("cpu")
            # Verify it's a real CPU tensor, not just a view
            assert not tensor.is_cuda if hasattr(tensor, "is_cuda") else True

    def test_patch_dtype_matches_base(self):
        """Patch tensors should match base model dtype."""
        # AC: @memory-management ac-4
        from nodes.exit import install_merged_patches

        mock_patcher = MagicMock()
        mock_patcher.model_state_dict.return_value = {
            "diffusion_model.layer.weight": torch.randn(4, 4, dtype=torch.bfloat16)
        }
        mock_clone = MagicMock()
        mock_patcher.clone.return_value = mock_clone

        patches_received = {}

        def capture_patches(patches, strength_patch):
            patches_received.update(patches)

        mock_clone.add_patches = capture_patches

        # Input in float32, should be cast to bfloat16
        merged_state = {
            "layer.weight": torch.randn(4, 4, dtype=torch.float32),
        }

        install_merged_patches(mock_patcher, merged_state, torch.bfloat16)

        for key, (patch_type, (tensor,)) in patches_received.items():
            assert tensor.dtype == torch.bfloat16


# =============================================================================
# AC-5: GPU usage within estimate bounds
# =============================================================================


class TestGpuUsageBounds:
    """AC: @memory-management ac-5

    Given: peak GPU usage during a chunk
    When: compared to compute_batch_size estimate
    Then: actual usage does not exceed the estimate by more than 20 percent
    """

    def test_compute_batch_size_conservative(self):
        """compute_batch_size should be conservative to stay within bounds."""
        # AC: @memory-management ac-5
        shape = (512, 512)
        n_models = 3
        dtype = torch.float32
        vram_gb = 1.0

        batch_size = compute_batch_size(shape, n_models, dtype, vram_gb)

        # Calculate expected memory usage for this batch size
        element_bytes = 4  # float32
        tensor_bytes = element_bytes * 512 * 512
        multiplier = 3 + 3 * n_models  # = 12

        estimated_usage = batch_size * tensor_bytes * multiplier
        # Note: when explicit vram_gb is provided, no 0.7 factor is applied
        budget_bytes = int(vram_gb * 1024**3)

        # Estimated usage should be within budget
        assert estimated_usage <= budget_bytes

    def test_small_batch_size_for_large_tensors(self):
        """Large tensors should result in small batch sizes."""
        # AC: @memory-management ac-5
        shape = (4096, 4096)  # 16M elements
        n_models = 5
        dtype = torch.float32
        vram_gb = 2.0

        batch_size = compute_batch_size(shape, n_models, dtype, vram_gb)

        # Should be a small batch size due to large tensor
        # 4096*4096*4 bytes = 64MB per tensor
        # With multiplier of 18, that's ~1.1GB per batch element
        # With 2GB and 70%, budget is ~1.4GB, so batch_size should be 1
        assert batch_size == 1

    def test_large_batch_size_for_small_tensors(self):
        """Small tensors should allow larger batch sizes."""
        # AC: @memory-management ac-5
        shape = (64, 64)  # 4K elements
        n_models = 2
        dtype = torch.float32
        vram_gb = 1.0

        batch_size = compute_batch_size(shape, n_models, dtype, vram_gb)

        # 64*64*4 = 16KB per tensor
        # With multiplier of 9, 144KB per batch element
        # 700MB budget allows ~5000 batch elements
        assert batch_size > 1000

    def test_more_models_reduces_batch_size(self):
        """More models should reduce batch size to stay within bounds."""
        # AC: @memory-management ac-5
        shape = (256, 256)
        dtype = torch.float32
        vram_gb = 0.5

        batch_2 = compute_batch_size(shape, 2, dtype, vram_gb)
        batch_8 = compute_batch_size(shape, 8, dtype, vram_gb)

        # More models = higher multiplier = smaller batch
        assert batch_2 > batch_8

    def test_fp16_allows_larger_batch(self):
        """FP16 should allow roughly 2x batch size."""
        # AC: @memory-management ac-5
        shape = (512, 512)
        n_models = 2
        vram_gb = 0.5

        batch_fp32 = compute_batch_size(shape, n_models, torch.float32, vram_gb)
        batch_fp16 = compute_batch_size(shape, n_models, torch.float16, vram_gb)

        # FP16 uses half the bytes
        assert batch_fp16 >= batch_fp32 * 1.5  # Allow some margin


# =============================================================================
# Integration tests
# =============================================================================


class TestMemoryManagementIntegration:
    """Integration tests for memory management across the pipeline."""

    def test_full_pipeline_cleanup_pattern(self):
        """Verify cleanup pattern: no gc per chunk, results on CPU."""
        # Integration test covering AC-1, AC-4
        keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
        base = {k: torch.randn(8, 8) for k in keys}

        gc_calls = []
        original_gc_collect = gc.collect

        def mock_gc_collect():
            gc_calls.append("gc.collect")
            return original_gc_collect()

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu.clone() + 1

        with patch("lib.executor.gc.collect", mock_gc_collect):
            results = chunked_evaluation(
                keys,
                base,
                eval_fn,
                batch_size=2,  # 3 chunks
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        # No gc.collect in normal chunked_evaluation path
        # (gc.collect runs after all groups in exit.py, not per-chunk)
        assert len(gc_calls) == 0

        # All results on CPU
        for tensor in results.values():
            assert tensor.device == torch.device("cpu")

    def test_oom_recovery_maintains_cleanup(self):
        """OOM recovery should maintain cleanup invariants."""
        # Integration test for AC-1 OOM path
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(4, 4) for k in keys}

        oom_triggered = [False]

        def eval_fn_with_oom(batch_keys, batch_gpu):
            if not oom_triggered[0] and len(batch_keys) > 1:
                oom_triggered[0] = True
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu * 2

        results = chunked_evaluation(
            keys,
            base,
            eval_fn_with_oom,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        assert oom_triggered[0]  # OOM was triggered
        assert len(results) == 4  # All keys processed

        # All results on CPU
        for tensor in results.values():
            assert tensor.device == torch.device("cpu")


# =============================================================================
# AC-7: Streaming safetensors writer — peak memory +1 tensor
# =============================================================================


class TestStreamingSave:
    """AC: @memory-management ac-7

    Given save_model=True, when writing checkpoint, then tensors are
    streamed to disk one at a time (peak memory is +1 tensor, not +full model).
    """

    # AC: @memory-management ac-7
    def test_atomic_save_uses_streaming_writer(self):
        """atomic_save should call stream_save_file for streaming writes."""
        from lib.persistence import atomic_save

        tensors = {"weight": torch.randn(4, 4)}
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "h",
            "__ecaj_recipe__": "{}",
            "__ecaj_affected_keys__": '["weight"]',
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name

        try:
            with (
                patch("lib.persistence.stream_save_file") as mock_stream,
                patch("os.open", return_value=42),
                patch("os.fsync"),
                patch("os.close"),
                patch("os.replace"),
            ):
                atomic_save(tensors, path, metadata)

            mock_stream.assert_called_once()
            call_args = mock_stream.call_args
            assert call_args[0][0] is tensors  # first arg is tensors dict
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # AC: @memory-management ac-7
    def test_atomic_save_produces_valid_safetensors(self):
        """atomic_save output is readable by safe_open."""
        from lib.persistence import atomic_save

        tensors = {
            "weight": torch.randn(8, 8),
            "bias": torch.randn(8),
        }
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "test_hash",
            "__ecaj_recipe__": "{}",
            "__ecaj_affected_keys__": '["weight","bias"]',
        }

        with tempfile.NamedTemporaryFile(
            suffix=".safetensors", delete=False
        ) as f:
            path = f.name

        try:
            atomic_save(tensors, path, metadata)

            with safe_open(path, framework="pt", device="cpu") as sf:
                for name, expected in tensors.items():
                    loaded = sf.get_tensor(name)
                    assert torch.equal(loaded, expected)
                loaded_meta = sf.metadata()
                assert loaded_meta["__ecaj_version__"] == "1"
        finally:
            os.unlink(path)


# =============================================================================
# AC-8: GPU offload after save
# =============================================================================


class TestGpuOffloadAfterSave:
    """AC: @memory-management ac-8

    Given save_model=True, when save completes, then GPU models are
    offloaded and VRAM cache cleared.
    """

    # AC: @memory-management ac-8
    def test_exit_node_calls_free_memory_and_soft_empty_cache_after_save(self, mock_model_patcher):
        """Exit node calls free_memory then soft_empty_cache after save_model=True."""
        import sys

        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        # Track call order
        call_order = []

        mock_atomic_save = MagicMock(side_effect=lambda *a, **kw: call_order.append("atomic_save"))

        # Inject comfy.model_management with trackable functions
        mm_mod = sys.modules.get("comfy.model_management")
        mock_free = MagicMock(side_effect=lambda *a, **kw: call_order.append("free_memory"))
        mock_device = MagicMock(return_value="cpu")
        mock_sec = MagicMock(side_effect=lambda *a, **kw: call_order.append("soft_empty_cache"))

        if mm_mod is not None:
            mm_mod.free_memory = mock_free
            mm_mod.get_torch_device = mock_device
            mm_mod.soft_empty_cache = mock_sec

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="test.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value="/tmp/test.safetensors"),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.check_cache", return_value=None),
            patch("nodes.exit.build_metadata", return_value={"__ecaj_version__": "1"}),
            patch("nodes.exit.atomic_save", mock_atomic_save),
        ):
            node = WIDENExitNode()
            node.execute(recipe, save_model=True, model_name="test")

        assert "atomic_save" in call_order
        assert "free_memory" in call_order
        assert "soft_empty_cache" in call_order
        # Ordering: save before free_memory before soft_empty_cache
        assert call_order.index("atomic_save") < call_order.index("free_memory")
        assert call_order.index("free_memory") < call_order.index("soft_empty_cache")


# =============================================================================
# AC-9: Safe pin_memory gate
# =============================================================================


class TestPinMemoryGate:
    """AC: @memory-management ac-9

    Given: system RAM is below a safe threshold
    When: chunked_evaluation transfers tensors to GPU
    Then: pin_memory() is skipped and plain .to() is used instead
    """

    # AC: @memory-management ac-9
    def test_pin_memory_skipped_when_ram_low(self):
        """When available RAM is low, pin_memory() should be skipped."""
        keys = ["k0"]
        base = {k: torch.randn(257, 257) for k in keys}  # >65536 elements (66049)

        pin_memory_called = []

        def mock_pin(self, *args, **kwargs):
            pin_memory_called.append(True)
            return self  # Don't actually pin (no CUDA device needed)

        original_to = torch.Tensor.to

        def mock_to(self, *args, **kwargs):
            # Redirect non-cpu device .to() to cpu to avoid needing real CUDA
            if args and isinstance(args[0], str) and args[0] not in ("cpu",):
                return original_to(self, "cpu", *args[1:], **kwargs)
            return original_to(self, *args, **kwargs)

        # Mock low RAM: return 100 bytes (way below 3x threshold)
        with (
            patch("lib.gpu_ops.get_available_ram_bytes", return_value=100),
            patch.object(torch.Tensor, "pin_memory", mock_pin),
            patch.object(torch.Tensor, "to", mock_to),
        ):
            chunked_evaluation(
                keys,
                base,
                lambda k, b: b * 2,
                batch_size=1,
                device="cuda:0",  # Non-cpu to enter the pin_memory gate
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        assert len(pin_memory_called) == 0, "pin_memory should not be called under low RAM"

    # AC: @memory-management ac-9
    def test_pin_memory_used_when_ram_sufficient(self):
        """When RAM is sufficient and device != cpu, pin_memory() is called."""
        keys = ["k0"]
        base = {k: torch.randn(257, 257) for k in keys}  # >65536 elements (66049)

        pin_memory_called = []

        def mock_pin(self, *args, **kwargs):
            pin_memory_called.append(True)
            return self  # Return self (don't actually pin, no CUDA needed)

        original_to = torch.Tensor.to

        def mock_to(self, *args, **kwargs):
            # Redirect any device .to() to cpu to avoid needing real CUDA
            if args and isinstance(args[0], str) and args[0] not in ("cpu",):
                return original_to(self, "cpu", *args[1:], **kwargs)
            return original_to(self, *args, **kwargs)

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        with (
            patch("lib.gpu_ops.get_available_ram_bytes", return_value=100 * 1024**3),
            patch.object(torch.Tensor, "pin_memory", mock_pin),
            patch.object(torch.Tensor, "to", mock_to),
        ):
            chunked_evaluation(
                keys,
                base,
                eval_fn,
                batch_size=1,
                device="cuda:0",  # Non-cpu device to trigger pin_memory path
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        assert len(pin_memory_called) > 0, "pin_memory should be called with sufficient RAM"


# =============================================================================
# AC-10: Pre-flight RAM check
# =============================================================================


class TestPreflightRamCheck:
    """AC: @memory-management ac-10

    Given: an exit node begins GPU evaluation
    When: available system RAM is estimated
    Then: if MemAvailable is below the estimated peak working set, a RuntimeError
          is raised with a message naming the shortfall in MB
    """

    # AC: @memory-management ac-10
    def test_raises_when_ram_insufficient(self):
        """check_ram_preflight raises RuntimeError when RAM is too low."""
        from lib.gpu_ops import check_ram_preflight

        with patch("lib.gpu_ops.get_available_ram_bytes", return_value=100 * 1024**2):
            with pytest.raises(RuntimeError, match="Insufficient system RAM"):
                check_ram_preflight(
                    merged_state_bytes=2 * 1024**3,  # 2 GB base state
                    worst_chunk_bytes=1 * 1024**3,  # 1 GB chunk
                    save_model=False,
                )

    # AC: @memory-management ac-10
    def test_passes_when_ram_sufficient(self):
        """check_ram_preflight does not raise when RAM is sufficient."""
        from lib.gpu_ops import check_ram_preflight

        # 100 GB available, tiny base state
        with patch("lib.gpu_ops.get_available_ram_bytes", return_value=100 * 1024**3):
            check_ram_preflight(
                merged_state_bytes=1024,
                worst_chunk_bytes=1024,
                save_model=False,
            )

    # AC: @memory-management ac-10
    def test_error_message_includes_mb_values(self):
        """Error message should include available and needed MB."""
        from lib.gpu_ops import check_ram_preflight

        with patch("lib.gpu_ops.get_available_ram_bytes", return_value=500 * 1024**2):
            with pytest.raises(RuntimeError, match=r"\d+ MB available.*\d+ MB needed"):
                check_ram_preflight(
                    merged_state_bytes=2 * 1024**3,
                    worst_chunk_bytes=1 * 1024**3,
                    save_model=True,
                )

    # AC: @memory-management ac-10
    def test_save_model_increases_estimate(self):
        """save_model=True should increase the peak RAM estimate."""
        from lib.gpu_ops import estimate_peak_ram

        base = estimate_peak_ram(
            merged_state_bytes=1024**3,
            worst_chunk_bytes=4 * 1024**2,
            save_model=False,
        )
        with_save = estimate_peak_ram(
            merged_state_bytes=1024**3,
            worst_chunk_bytes=4 * 1024**2,
            save_model=True,
        )
        assert with_save > base


# =============================================================================
# Accurate RAM Preflight Estimation
# =============================================================================


class TestAccurateRamPreflight:
    """Tests for accurate RAM preflight estimation.

    Validates that the corrected formula does not include loader memory
    (already reflected in MemAvailable) and properly handles incremental
    recompute scenarios.
    """

    # AC: @accurate-ram-preflight ac-1
    def test_loader_memory_not_in_estimate(self):
        """Loader memory is already in MemAvailable, not added to estimate."""
        from lib.gpu_ops import estimate_peak_ram

        result = estimate_peak_ram(
            merged_state_bytes=1024**3,
            worst_chunk_bytes=4 * 1024**2,
            save_model=False,
            loader_bytes=500 * 1024**2,  # 500 MB of loaders
        )
        # Formula: merged_state_bytes + 2*worst_chunk_bytes (no loader term)
        expected = 1024**3 + 2 * 4 * 1024**2
        assert result == expected

    # AC: @accurate-ram-preflight ac-2
    def test_incremental_estimates_less_than_full(self):
        """Incremental recompute of M bytes estimates less than full N bytes."""
        from lib.gpu_ops import estimate_peak_ram

        full_bytes = 1024**3  # 1 GB full recompute
        incremental_bytes = 100 * 1024**2  # 100 MB incremental

        full_estimate = estimate_peak_ram(
            merged_state_bytes=full_bytes,
            worst_chunk_bytes=4 * 1024**2,
            save_model=True,
        )
        incremental_estimate = estimate_peak_ram(
            merged_state_bytes=incremental_bytes,
            worst_chunk_bytes=4 * 1024**2,
            save_model=True,
        )
        assert incremental_estimate < full_estimate

    # AC: @accurate-ram-preflight ac-3
    def test_loader_bytes_logged_but_not_in_estimate(self, caplog):
        """loader_bytes is logged at DEBUG but does not affect the estimate."""
        import logging

        from lib.gpu_ops import estimate_peak_ram

        with caplog.at_level(logging.DEBUG, logger="ecaj.gpu_ops"):
            without_loader = estimate_peak_ram(
                merged_state_bytes=1024**3,
                worst_chunk_bytes=4 * 1024**2,
                save_model=False,
                loader_bytes=0,
            )
        # Zero value is still logged
        assert "loader_bytes=0" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="ecaj.gpu_ops"):
            with_loader = estimate_peak_ram(
                merged_state_bytes=1024**3,
                worst_chunk_bytes=4 * 1024**2,
                save_model=False,
                loader_bytes=10**9,
            )
        assert with_loader == without_loader
        assert "loader_bytes=1000000000" in caplog.text

    # AC: @accurate-ram-preflight ac-4
    def test_save_overhead_is_5_percent(self):
        """Save overhead is exactly 5% of merged_state_bytes."""
        from lib.gpu_ops import estimate_peak_ram

        merged = 1000 * 1024**2  # 1000 MB
        no_save = estimate_peak_ram(
            merged_state_bytes=merged,
            worst_chunk_bytes=0,
            save_model=False,
        )
        with_save = estimate_peak_ram(
            merged_state_bytes=merged,
            worst_chunk_bytes=0,
            save_model=True,
        )
        assert with_save - no_save == int(merged * 0.05)

    # AC: @accurate-ram-preflight ac-5
    def test_no_save_overhead_when_false(self):
        """No save overhead is added when save_model=False."""
        from lib.gpu_ops import estimate_peak_ram

        merged = 1024**3
        chunk = 4 * 1024**2
        result = estimate_peak_ram(
            merged_state_bytes=merged,
            worst_chunk_bytes=chunk,
            save_model=False,
        )
        # Exact formula: merged + 2*chunk, no save term
        assert result == merged + 2 * chunk

    # AC: @accurate-ram-preflight ac-6
    def test_incremental_lte_full_recompute(self):
        """Incremental estimate is strictly <= full recompute estimate."""
        from lib.gpu_ops import estimate_peak_ram

        chunk = 4 * 1024**2
        for save_model in [True, False]:
            full = estimate_peak_ram(
                merged_state_bytes=1024**3,
                worst_chunk_bytes=chunk,
                save_model=save_model,
            )
            incremental = estimate_peak_ram(
                merged_state_bytes=100 * 1024**2,
                worst_chunk_bytes=chunk,
                save_model=save_model,
            )
            assert incremental <= full

    # AC: @accurate-ram-preflight ac-7
    def test_clip_exit_uses_corrected_formula(self):
        """CLIP exit node calls check_ram_preflight with same corrected signature."""
        from nodes.clip_exit import WIDENCLIPExitNode

        mock_clip = MagicMock()
        mock_patcher = MagicMock()
        state = {
            "clip_l.transformer.text_model.weight": torch.randn(4, 4, dtype=torch.float32),
        }
        mock_patcher.model_state_dict.return_value = state
        mock_clip.patcher = mock_patcher
        mock_clip.clone.return_value = MagicMock()

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="clip"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_loader.loaded_bytes = 100 * 1024**2
        mock_analyze = MagicMock(
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(None)): set(state.keys())},
            affected_keys=set(state.keys()),
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()

        sig = OpSignature(shape=(4, 4), ndim=2)
        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())
        preflight_mock = MagicMock()

        with (
            patch("nodes.clip_exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.clip_exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.clip_exit.compile_plan", return_value=dummy_plan),
            patch("nodes.clip_exit.compile_batch_groups", return_value={sig: list(state.keys())}),
            patch("nodes.clip_exit.chunked_evaluation"),
            patch("nodes.clip_exit.check_ram_preflight", preflight_mock),
        ):
            node = WIDENCLIPExitNode()
            node.execute(recipe)

        preflight_mock.assert_called_once()
        call_kwargs = preflight_mock.call_args.kwargs
        # AC-7: same corrected formula — no n_models, has loader_bytes
        assert "n_models" not in call_kwargs
        assert "loader_bytes" in call_kwargs
        assert call_kwargs["save_model"] is False


# =============================================================================
# AC-11: Broader OOM exception handling
# =============================================================================


class TestBroaderExceptionHandling:
    """AC: @memory-management ac-11

    Given: a non-CUDA memory exception occurs during chunked evaluation
    When: the exception is caught
    Then: gc.collect() and empty_cache() run, then a RuntimeError is re-raised
          with the original as __cause__
    """

    # AC: @memory-management ac-11
    def test_memory_error_caught_and_reraised(self):
        """MemoryError during eval_fn should trigger cleanup and re-raise."""
        import pytest

        keys = ["k0"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn_oom(batch_keys, batch_gpu):
            raise MemoryError("Cannot allocate memory")

        gc_calls = []
        original_gc = gc.collect

        def mock_gc():
            gc_calls.append(True)
            return original_gc()

        with patch("lib.gpu_ops.gc.collect", mock_gc):
            with pytest.raises(RuntimeError, match="System memory exhausted") as exc_info:
                chunked_evaluation(
                    keys, base, eval_fn_oom,
                    batch_size=1, device="cpu",
                    dtype=torch.float32, storage_dtype=torch.float32,
                )

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, MemoryError)
        assert len(gc_calls) >= 1

    # AC: @memory-management ac-11
    def test_runtime_error_oom_caught(self):
        """RuntimeError with 'not enough memory' should trigger cleanup."""
        import pytest

        keys = ["k0"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn_oom(batch_keys, batch_gpu):
            raise RuntimeError("DefaultCPUAllocator: not enough memory")

        with pytest.raises(RuntimeError, match="System memory exhausted") as exc_info:
            chunked_evaluation(
                keys, base, eval_fn_oom,
                batch_size=1, device="cpu",
                dtype=torch.float32, storage_dtype=torch.float32,
            )

        assert exc_info.value.__cause__ is not None

    # AC: @memory-management ac-11
    def test_non_oom_runtime_error_propagates(self):
        """RuntimeError without OOM message should propagate unchanged."""
        import pytest

        keys = ["k0"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn_bad(batch_keys, batch_gpu):
            raise RuntimeError("shape mismatch")

        with pytest.raises(RuntimeError, match="shape mismatch"):
            chunked_evaluation(
                keys, base, eval_fn_bad,
                batch_size=1, device="cpu",
                dtype=torch.float32, storage_dtype=torch.float32,
            )


# =============================================================================
# AC-12: Cache eviction under memory pressure
# =============================================================================


class TestCacheEvictionUnderPressure:
    """AC: @memory-management ac-12

    Given: the incremental cache holds tensors and MemAvailable drops below a threshold
    When: cache write is attempted
    Then: the cache entry is evicted instead of stored
    """

    def setup_method(self):
        _incremental_cache.clear()

    def teardown_method(self):
        _incremental_cache.clear()

    # AC: @memory-management ac-12
    def test_cache_evicted_when_ram_low(self, mock_model_patcher):
        """Cache should be evicted (not stored) when RAM is low at write time."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        # Return very low RAM so cache_bytes * 2 > avail
        with patch("nodes.exit.get_available_ram_bytes", return_value=1):
            _run_exit_node(recipe, mock_model_patcher, keys)

        assert len(_incremental_cache) == 0, "Cache should be evicted under low RAM"

    # AC: @memory-management ac-12
    def test_cache_stored_when_ram_sufficient(self, mock_model_patcher):
        """Cache should be stored normally when RAM is sufficient."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        # Return very high RAM so cache write succeeds
        with patch("nodes.exit.get_available_ram_bytes", return_value=100 * 1024**3):
            _run_exit_node(recipe, mock_model_patcher, keys)

        assert len(_incremental_cache) == 1, "Cache should have one entry with sufficient RAM"


# =============================================================================
# AC: @exit-node ac-9 — Pre-flight RAM check in exit node
# =============================================================================


class TestExitNodeRamPreflight:
    """AC: @exit-node ac-9

    Given: system MemAvailable is below the estimated peak working set
    When: execute() checks RAM before the GPU loop
    Then: a RuntimeError is raised naming the shortfall before any GPU work begins
    """

    # AC: @exit-node ac-9
    def test_exit_node_has_ram_preflight(self, mock_model_patcher):
        """Exit node raises RuntimeError before GPU work when RAM is insufficient."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)
        chunked_mock = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", chunked_mock),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("lib.gpu_ops.get_available_ram_bytes", return_value=100),
        ):
            node = WIDENExitNode()
            with pytest.raises(RuntimeError, match="Insufficient system RAM"):
                node.execute(recipe)

        # chunked_evaluation should NOT have been called (preflight aborted first)
        chunked_mock.assert_not_called()


# =============================================================================
# AC: @clip-exit-node ac-11 — Pre-flight RAM check in CLIP exit node
# =============================================================================


class TestClipExitNodeRamPreflight:
    """AC: @clip-exit-node ac-11

    Given: system MemAvailable is below the estimated peak working set
    When: execute() checks RAM before the GPU loop
    Then: a RuntimeError is raised naming the shortfall before any GPU work begins
    """

    # AC: @clip-exit-node ac-11
    def test_clip_exit_node_has_ram_preflight(self):
        """CLIP Exit node raises RuntimeError before GPU work when RAM is insufficient."""
        from nodes.clip_exit import WIDENCLIPExitNode

        # Build a CLIP-compatible mock
        mock_clip = MagicMock()
        mock_patcher = MagicMock()
        mock_patcher.model_state_dict.return_value = {
            "clip_l.transformer.text_model.weight": torch.randn(4, 4, dtype=torch.float32),
        }
        mock_clip.patcher = mock_patcher
        mock_clip.clone.return_value = MagicMock()

        keys = list(mock_patcher.model_state_dict().keys())

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="clip"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_analyze = MagicMock(
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(None)): set(keys)},
            affected_keys=set(keys),
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()

        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())
        sig = OpSignature(shape=(4, 4), ndim=2)
        chunked_mock = MagicMock()

        with (
            patch("nodes.clip_exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.clip_exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.clip_exit.compile_plan", return_value=dummy_plan),
            patch("nodes.clip_exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.clip_exit.chunked_evaluation", chunked_mock),
            patch("lib.gpu_ops.get_available_ram_bytes", return_value=100),
        ):
            node = WIDENCLIPExitNode()
            with pytest.raises(RuntimeError, match="Insufficient system RAM"):
                node.execute(recipe)

        chunked_mock.assert_not_called()


# =============================================================================
# Regression: preflight byte calculations use processed keys and per-sig batching
# =============================================================================


class TestExitNodePreflightByteCalculation:
    """Regression tests asserting exact preflight byte values passed to
    check_ram_preflight by the exit node.

    Ensures:
    - merged_state_bytes reflects only processed keys (not full base_state)
    - worst_chunk_bytes pairs each signature's shape with its own batch_size
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _incremental_cache.clear()
        yield
        _incremental_cache.clear()

    # AC: @exit-node ac-9
    def test_preflight_args_use_processed_keys_only(self, mock_model_patcher):
        """merged_state_bytes counts only keys in batch_groups, not full state."""
        state = mock_model_patcher.model_state_dict()
        all_keys = list(state.keys())
        # Use only the first key in batch_groups — the rest are unaffected
        processed_key = all_keys[0]
        unprocessed_keys = all_keys[1:]
        assert len(unprocessed_keys) > 0, "Need at least 2 keys for this test"

        sig = OpSignature(shape=(4, 4), ndim=2)
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, all_keys, recipe=recipe,
        )
        preflight_mock = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: [processed_key]}),
            patch("nodes.exit.chunked_evaluation"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight", preflight_mock),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        preflight_mock.assert_called_once()
        call_kwargs = preflight_mock.call_args
        t = state[processed_key]
        expected_merged_bytes = t.nelement() * t.element_size()
        assert call_kwargs.kwargs["merged_state_bytes"] == expected_merged_bytes

    # AC: @exit-node ac-9
    def test_preflight_worst_chunk_pairs_shape_with_own_batch_size(self, mock_model_patcher):
        """worst_chunk_bytes = max over sigs of (numel * batch_size_i * elem_size).

        Uses two signatures with skewed shapes. With deterministic batch sizes,
        we can assert the exact worst_chunk value and verify per-sig pairing.
        """
        state = mock_model_patcher.model_state_dict()
        all_keys = list(state.keys())

        # Two signatures: small shape gets large batch, large shape gets small batch
        sig_small = OpSignature(shape=(4, 4), ndim=2)       # numel=16
        sig_large = OpSignature(shape=(16, 16), ndim=2)     # numel=256
        batch_groups = {
            sig_small: [all_keys[0]],
            sig_large: [all_keys[1]] if len(all_keys) > 1 else [all_keys[0]],
        }

        # Deterministic batch sizes: small shape → batch 100, large shape → batch 2
        def fake_batch_size(shape, n_models, dtype, **kw):
            numel = 1
            for d in shape:
                numel *= d
            return 100 if numel <= 16 else 2

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, all_keys, recipe=recipe,
        )
        preflight_mock = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value=batch_groups),
            patch("nodes.exit.chunked_evaluation"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight", preflight_mock),
            patch("nodes.exit.compute_batch_size", side_effect=fake_batch_size),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        preflight_mock.assert_called_once()
        call_kwargs = preflight_mock.call_args

        element_size = torch.finfo(torch.float32).bits // 8  # 4
        # Per-sig: small = 4 * 16 * 100 = 6400, large = 4 * 256 * 2 = 2048
        # worst_chunk = max(6400, 2048) = 6400
        expected_worst = max(
            element_size * 16 * 100,   # sig_small
            element_size * 256 * 2,    # sig_large
        )
        assert expected_worst == 6400  # sanity check
        assert call_kwargs.kwargs["worst_chunk_bytes"] == expected_worst


class TestClipExitNodePreflightByteCalculation:
    """Regression tests for CLIP exit node preflight byte calculations."""

    # AC: @clip-exit-node ac-11
    def test_preflight_args_use_processed_keys_only(self):
        """merged_state_bytes counts only keys in batch_groups, not full state."""
        from nodes.clip_exit import WIDENCLIPExitNode

        mock_clip = MagicMock()
        mock_patcher = MagicMock()
        state = {
            "clip_l.transformer.text_model.weight": torch.randn(4, 4, dtype=torch.float32),
            "clip_l.transformer.text_model.bias": torch.randn(4, dtype=torch.float32),
        }
        mock_patcher.model_state_dict.return_value = state
        mock_clip.patcher = mock_patcher
        mock_clip.clone.return_value = MagicMock()

        all_keys = list(state.keys())
        processed_key = all_keys[0]

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="clip"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_analyze = MagicMock(
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(None)): set(all_keys)},
            affected_keys=set(all_keys),
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()

        sig = OpSignature(shape=(4, 4), ndim=2)
        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())
        preflight_mock = MagicMock()

        with (
            patch("nodes.clip_exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.clip_exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.clip_exit.compile_plan", return_value=dummy_plan),
            patch("nodes.clip_exit.compile_batch_groups", return_value={sig: [processed_key]}),
            patch("nodes.clip_exit.chunked_evaluation"),
            patch("nodes.clip_exit.check_ram_preflight", preflight_mock),
        ):
            node = WIDENCLIPExitNode()
            node.execute(recipe)

        preflight_mock.assert_called_once()
        call_kwargs = preflight_mock.call_args
        t = state[processed_key]
        expected_merged_bytes = t.nelement() * t.element_size()
        assert call_kwargs.kwargs["merged_state_bytes"] == expected_merged_bytes


# =============================================================================
# get_available_ram_bytes unit tests
# =============================================================================


class TestGetAvailableRamBytes:
    """Unit tests for get_available_ram_bytes()."""

    def test_returns_positive_integer(self):
        """Should return a positive integer on any platform."""
        from lib.gpu_ops import get_available_ram_bytes

        result = get_available_ram_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_fallback_on_missing_procfs(self):
        """Should return fallback when /proc/meminfo is unavailable."""
        from lib.gpu_ops import _FALLBACK_RAM_BYTES, get_available_ram_bytes

        with patch("builtins.open", side_effect=OSError("No such file")):
            # Reset warning flag so it doesn't affect output
            import lib.gpu_ops
            lib.gpu_ops._meminfo_warned = True  # Suppress warning
            result = get_available_ram_bytes()

        assert result == _FALLBACK_RAM_BYTES

    def test_parses_real_meminfo(self):
        """Should parse a realistic /proc/meminfo format."""
        from io import StringIO

        from lib.gpu_ops import get_available_ram_bytes

        fake_meminfo = (
            "MemTotal:       65536000 kB\n"
            "MemFree:         1000000 kB\n"
            "MemAvailable:   32000000 kB\n"
            "Buffers:          500000 kB\n"
        )

        with patch("builtins.open", return_value=StringIO(fake_meminfo)):
            result = get_available_ram_bytes()

        assert result == 32000000 * 1024  # 32 GB in bytes


# =============================================================================
# AC-13: base_state freed before save, re-acquired fresh
# =============================================================================


class TestBaseStateFreedBeforeSave:
    """AC: @memory-management ac-13

    Given: GPU evaluation completes and results are in merged_state
    When: save_model=True and save path is about to be written
    Then: base_state is freed before save; state dict re-acquired fresh for save only
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _incremental_cache.clear()
        yield
        _incremental_cache.clear()

    # AC: @memory-management ac-13
    def test_base_state_freed_before_save(self, mock_model_patcher):
        """model_state_dict called twice (setup + save) when save_model=True."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},)),
            backbone=None,
            t_factor=1.0,
        )

        # Track model_state_dict calls
        call_count = [0]
        original_msd = mock_model_patcher.model_state_dict

        def tracking_msd():
            call_count[0] += 1
            return original_msd()

        mock_model_patcher.model_state_dict = tracking_msd

        _run_exit_node(
            recipe, mock_model_patcher, keys,
            extra_patches={
                "nodes.exit.validate_model_name": "test.safetensors",
                "nodes.exit._resolve_checkpoints_path": "/tmp/test.safetensors",
                "nodes.exit.serialize_recipe": "{}",
                "nodes.exit.compute_recipe_hash": "hash",
                "nodes.exit.check_cache": None,
                "nodes.exit.build_metadata": {"__ecaj_version__": "1"},
                "nodes.exit.atomic_save": MagicMock(),
            },
            save_model=True,
            model_name="test",
        )

        # Called once at setup (line 445) and once for save re-acquisition
        assert call_count[0] == 2, (
            f"Expected 2 model_state_dict calls (setup + save), got {call_count[0]}"
        )
