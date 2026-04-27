"""Tests for lib/incremental_writer.py — incremental safetensors artifact writer.

Covers: @streaming-full-model-materialization ac-incomplete-write-not-reused.

Round-trip validation, error handling for incomplete/inconsistent artifacts,
and temp file cleanup on failure.
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from lib.incremental_writer import IncrementalSafetensorsWriter, TensorSpec

# =============================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Round-trip: complete artifacts are valid safetensors files
# =============================================================================


class TestIncrementalWriterRoundTrip:
    """Complete write cycles produce valid safetensors artifacts."""

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_single_tensor_round_trip(self, tmp_path):
        """A single tensor written incrementally round-trips through safe_open."""
        dest = tmp_path / "model.safetensors"
        manifest = {
            "weight": TensorSpec(shape=(4, 4), dtype=torch.float32),
        }
        tensor = torch.randn(4, 4, dtype=torch.float32)

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        try:
            writer.write_tensor("weight", tensor)
            writer.finalize()
        except BaseException:
            writer.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            loaded = f.get_tensor("weight")
            assert torch.equal(loaded, tensor)

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_multiple_keys_and_dtypes(self, tmp_path):
        """Multiple tensors with different dtypes all round-trip correctly."""
        dest = tmp_path / "model.safetensors"
        tensors = {
            "fp32_weight": torch.randn(8, 8, dtype=torch.float32),
            "bf16_weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "fp16_bias": torch.randn(16, dtype=torch.float16),
        }
        manifest = {
            k: TensorSpec(shape=tuple(t.shape), dtype=t.dtype)
            for k, t in tensors.items()
        }

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        try:
            for key, tensor in tensors.items():
                writer.write_tensor(key, tensor)
            writer.finalize()
        except BaseException:
            writer.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key, expected in tensors.items():
                loaded = f.get_tensor(key)
                assert loaded.dtype == expected.dtype, f"dtype mismatch for {key}"
                assert torch.equal(loaded, expected), f"value mismatch for {key}"

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_metadata_preserved(self, tmp_path):
        """Metadata dict is readable from the produced artifact."""
        dest = tmp_path / "model.safetensors"
        manifest = {"x": TensorSpec(shape=(2,), dtype=torch.float32)}
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "abc123",
        }

        writer = IncrementalSafetensorsWriter(
            manifest, str(dest), metadata=metadata
        )
        try:
            writer.write_tensor("x", torch.zeros(2, dtype=torch.float32))
            writer.finalize()
        except BaseException:
            writer.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            saved_meta = f.metadata()
            assert saved_meta["__ecaj_version__"] == "1"
            assert saved_meta["__ecaj_recipe_hash__"] == "abc123"

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_write_order_independent(self, tmp_path):
        """Tensors can be written in any order; the artifact is still valid."""
        dest = tmp_path / "model.safetensors"
        tensors = {
            "z_last": torch.randn(2, 2, dtype=torch.float32),
            "a_first": torch.randn(3, 3, dtype=torch.float32),
            "m_middle": torch.randn(4, dtype=torch.float32),
        }
        manifest = {
            k: TensorSpec(shape=tuple(t.shape), dtype=t.dtype)
            for k, t in tensors.items()
        }

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        try:
            # Write in reverse-sorted order (opposite of safetensors sort)
            for key in sorted(tensors.keys(), reverse=True):
                writer.write_tensor(key, tensors[key])
            writer.finalize()
        except BaseException:
            writer.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key, expected in tensors.items():
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, expected), f"mismatch for {key}"


# =============================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Rejection: incomplete or inconsistent artifacts are never published
# =============================================================================


class TestIncrementalWriterRejection:
    """Incomplete or inconsistent writes fail and leave no artifact."""

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_missing_key_finalize_fails(self, tmp_path):
        """Finalize with unwritten keys raises and leaves no artifact."""
        dest = tmp_path / "model.safetensors"
        manifest = {
            "written": TensorSpec(shape=(4,), dtype=torch.float32),
            "missing": TensorSpec(shape=(4,), dtype=torch.float32),
        }

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.write_tensor("written", torch.randn(4, dtype=torch.float32))

        with pytest.raises(RuntimeError, match="missing"):
            writer.finalize()

        assert not dest.exists(), "No artifact should exist after failed finalize"

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_duplicate_key_raises(self, tmp_path):
        """Writing the same key twice raises immediately."""
        dest = tmp_path / "model.safetensors"
        manifest = {"key": TensorSpec(shape=(4,), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.write_tensor("key", torch.randn(4, dtype=torch.float32))

        with pytest.raises(RuntimeError, match="[Dd]uplicate"):
            writer.write_tensor("key", torch.randn(4, dtype=torch.float32))

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_wrong_shape_raises(self, tmp_path):
        """Tensor with wrong shape raises immediately."""
        dest = tmp_path / "model.safetensors"
        manifest = {"key": TensorSpec(shape=(4, 4), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))

        with pytest.raises(RuntimeError, match="[Ss]hape"):
            writer.write_tensor("key", torch.randn(8, 8, dtype=torch.float32))

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_wrong_dtype_raises(self, tmp_path):
        """Tensor with wrong dtype raises immediately."""
        dest = tmp_path / "model.safetensors"
        manifest = {"key": TensorSpec(shape=(4, 4), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))

        with pytest.raises(RuntimeError, match="[Dd]type"):
            writer.write_tensor("key", torch.randn(4, 4, dtype=torch.float16))

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_unexpected_key_raises(self, tmp_path):
        """Writing a key not in the manifest raises immediately."""
        dest = tmp_path / "model.safetensors"
        manifest = {"expected": TensorSpec(shape=(4,), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))

        with pytest.raises(RuntimeError, match="[Uu]nexpected"):
            writer.write_tensor("surprise", torch.randn(4, dtype=torch.float32))


# =============================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Cleanup: failed or aborted writes leave no temp files
# =============================================================================


class TestIncrementalWriterCleanup:
    """Abort and failure paths clean up temporary files."""

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_abort_removes_temp_file(self, tmp_path):
        """Aborting after partial writes removes the temp file."""
        dest = tmp_path / "model.safetensors"
        manifest = {
            "a": TensorSpec(shape=(4,), dtype=torch.float32),
            "b": TensorSpec(shape=(4,), dtype=torch.float32),
        }

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.write_tensor("a", torch.randn(4, dtype=torch.float32))
        writer.abort()

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0, "Temp file should be cleaned up"

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_abort_safe_to_call_multiple_times(self, tmp_path):
        """Calling abort() multiple times does not raise."""
        dest = tmp_path / "model.safetensors"
        manifest = {"x": TensorSpec(shape=(2,), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.abort()
        writer.abort()  # Should not raise

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_failed_finalize_leaves_no_artifact(self, tmp_path):
        """When finalize fails (missing keys), no artifact or temp remains."""
        dest = tmp_path / "model.safetensors"
        manifest = {
            "a": TensorSpec(shape=(4,), dtype=torch.float32),
            "b": TensorSpec(shape=(4,), dtype=torch.float32),
        }

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.write_tensor("a", torch.randn(4, dtype=torch.float32))

        with pytest.raises(RuntimeError):
            writer.finalize()

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_write_after_finalize_raises(self, tmp_path):
        """Writing after finalize raises RuntimeError."""
        dest = tmp_path / "model.safetensors"
        manifest = {"x": TensorSpec(shape=(2,), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.write_tensor("x", torch.zeros(2, dtype=torch.float32))
        writer.finalize()

        with pytest.raises(RuntimeError, match="finalized"):
            writer.write_tensor("x", torch.zeros(2, dtype=torch.float32))

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_write_after_abort_raises(self, tmp_path):
        """Writing after abort raises RuntimeError."""
        dest = tmp_path / "model.safetensors"
        manifest = {"x": TensorSpec(shape=(2,), dtype=torch.float32)}

        writer = IncrementalSafetensorsWriter(manifest, str(dest))
        writer.abort()

        with pytest.raises(RuntimeError, match="aborted"):
            writer.write_tensor("x", torch.zeros(2, dtype=torch.float32))

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_context_manager_aborts_on_error(self, tmp_path):
        """Using as context manager auto-aborts on exception."""
        dest = tmp_path / "model.safetensors"
        manifest = {
            "a": TensorSpec(shape=(4,), dtype=torch.float32),
            "b": TensorSpec(shape=(4,), dtype=torch.float32),
        }

        with pytest.raises(ValueError, match="intentional"):
            with IncrementalSafetensorsWriter(manifest, str(dest)) as writer:
                writer.write_tensor("a", torch.randn(4, dtype=torch.float32))
                raise ValueError("intentional error")

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0
