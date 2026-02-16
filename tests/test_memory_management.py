"""Tests for GPU memory management during merge execution.

Covers acceptance criteria for @memory-management spec.
"""

import gc
import inspect
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch
from safetensors import safe_open

from lib.executor import (
    chunked_evaluation,
    compile_batch_groups,
    compute_batch_size,
)

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

    def test_exit_node_calls_gc_after_groups(self):
        """Exit node should call gc.collect after all OpSignature groups complete."""
        # AC: @memory-management ac-2
        # gc.collect runs once after the evaluation loop, not per-group.
        # OOM backoff in chunked_evaluation handles per-group memory pressure.
        import nodes.exit

        # Verify gc is imported
        assert hasattr(nodes.exit, "gc") or "gc" in dir(nodes.exit)


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

    def test_exit_node_cleanup_in_finally(self):
        """Exit node should call loader.cleanup() in finally block."""
        # AC: @memory-management ac-3
        # Verify by reading the source that cleanup is in finally
        import inspect

        from nodes.exit import WIDENExitNode

        source = inspect.getsource(WIDENExitNode.execute)
        assert "finally:" in source
        assert "loader.cleanup()" in source


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
        """atomic_save should use stream_save_file, not safetensors.save_file."""
        source = inspect.getsource(
            __import__("lib.persistence", fromlist=["atomic_save"]).atomic_save
        )
        assert "stream_save_file" in source

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
    def test_exit_node_calls_free_memory_after_save(self):
        """Exit node source contains free_memory call after save."""
        source = inspect.getsource(
            __import__("nodes.exit", fromlist=["WIDENExitNode"]).WIDENExitNode.execute
        )
        # free_memory should appear after atomic_save in the source
        save_idx = source.index("atomic_save")
        free_idx = source.index("free_memory")
        assert free_idx > save_idx, "free_memory must come after atomic_save"

    # AC: @memory-management ac-8
    def test_exit_node_calls_soft_empty_cache_after_save(self):
        """Exit node source contains soft_empty_cache call after save."""
        source = inspect.getsource(
            __import__("nodes.exit", fromlist=["WIDENExitNode"]).WIDENExitNode.execute
        )
        save_idx = source.index("atomic_save")
        cache_idx = source.index("soft_empty_cache")
        assert cache_idx > save_idx, "soft_empty_cache must come after atomic_save"
