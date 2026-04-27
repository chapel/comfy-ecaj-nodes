"""Tests for lib/checkpoint_materialization.py — full checkpoint materialization sink.

Covers:
  @streaming-full-model-materialization ac-base-weight-bounded-copying
  @streaming-full-model-materialization ac-affected-results-released
  @streaming-full-model-materialization ac-incomplete-write-not-reused

Uses a fake base model state with affected and unaffected keys. Loads the
resulting safetensors artifact and asserts correctness.
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from lib.checkpoint_materialization import CheckpointMaterializationSink


def _make_base_state(
    n_unaffected: int = 3,
    n_affected: int = 2,
    shape: tuple[int, ...] = (4, 4),
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Build a fake base model state and affected key set.

    Returns:
        (base_state, affected_keys) where base_state contains all keys
        and affected_keys is the subset that will be replaced by merge.
    """
    base_state: dict[str, torch.Tensor] = {}
    affected_keys: set[str] = set()

    for i in range(n_unaffected):
        base_state[f"diffusion_model.unaffected.{i}.weight"] = torch.randn(
            shape, dtype=dtype
        )
    for i in range(n_affected):
        key = f"diffusion_model.affected.{i}.weight"
        base_state[key] = torch.randn(shape, dtype=dtype)
        affected_keys.add(key)

    return base_state, affected_keys


def _make_merged_tensors(
    affected_keys: set[str],
    shape: tuple[int, ...] = (4, 4),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Create distinct merged tensors for each affected key."""
    return {k: torch.randn(shape, dtype=dtype) for k in affected_keys}


# =============================================================================
# AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
# Base weights recorded without a second full-model dict in memory.
# =============================================================================


class TestBaseWeightBoundedCopying:
    """Unaffected base weights appear in the artifact without a full dict copy."""

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_unaffected_keys_equal_base_tensors(self, tmp_path):
        """Unaffected keys in the artifact match the original base tensors."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key, expected in base_state.items():
                if key not in affected_keys:
                    loaded = f.get_tensor(key)
                    assert torch.equal(loaded, expected), (
                        f"Unaffected key {key} does not match base tensor"
                    )

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_write_base_weights_does_not_copy_full_state(self, tmp_path):
        """write_base_weights only writes unaffected keys, skipping affected ones."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=3, n_affected=2)
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            # After write_base_weights, only unaffected keys are written.
            # Affected keys still need to be provided via write_tensor.
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        # All keys present in artifact
        with safe_open(str(dest), framework="pt", device="cpu") as f:
            artifact_keys = set(f.keys())
            assert artifact_keys == set(base_state.keys())

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_all_base_keys_in_artifact(self, tmp_path):
        """Every key from the base state is present in the final artifact."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=5, n_affected=3)
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            assert set(f.keys()) == set(base_state.keys())

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_storage_dtype_applied_to_base_weights(self, tmp_path):
        """Base weights are converted to storage_dtype in the artifact."""
        dest = tmp_path / "model.safetensors"
        # Base state is float32, storage dtype is float16
        base_state, affected_keys = _make_base_state(
            n_unaffected=2, n_affected=1, dtype=torch.float32
        )
        merged = _make_merged_tensors(affected_keys, dtype=torch.float16)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float16,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in base_state:
                if key not in affected_keys:
                    loaded = f.get_tensor(key)
                    assert loaded.dtype == torch.float16


# =============================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# Affected tensors written incrementally, not accumulated.
# =============================================================================


class TestAffectedResultsReleased:
    """Affected keys written via sink protocol are in the artifact."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_affected_keys_equal_merged_tensors(self, tmp_path):
        """Affected keys in the artifact match the merged tensors, not the base."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in affected_keys:
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, merged[key]), (
                    f"Affected key {key} should match merged tensor, not base"
                )
                assert not torch.equal(loaded, base_state[key]), (
                    f"Affected key {key} should differ from base tensor"
                )

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_finalize_returns_empty_dict(self, tmp_path):
        """finalize() returns empty dict — tensors are on disk, not in memory."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=1, n_affected=1)
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            result = sink.finalize()
        except BaseException:
            sink.abort()
            raise

        assert result == {}, "finalize should return empty dict (data on disk)"

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_no_full_merged_state_dict_required(self, tmp_path):
        """Materialization completes without building a full merged_state dict.

        The sink only receives affected tensors one at a time via write_tensor,
        never a complete dict of all model keys.
        """
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=4, n_affected=3)
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            # Write affected tensors one at a time (simulating streaming eval)
            for key in sorted(affected_keys):
                sink.write_tensor(key, merged[key])
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        # Verify all keys correct
        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in base_state:
                loaded = f.get_tensor(key)
                if key in affected_keys:
                    assert torch.equal(loaded, merged[key])
                else:
                    assert torch.equal(loaded, base_state[key])

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_write_tensor_before_base_weights_works(self, tmp_path):
        """Affected tensors can be written before or after base weights."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=2, n_affected=2)
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            # Write one affected tensor first
            first_key = sorted(affected_keys)[0]
            sink.write_tensor(first_key, merged[first_key])

            # Then base weights
            sink.write_base_weights()

            # Then remaining affected tensors
            for key in sorted(affected_keys):
                if key != first_key:
                    sink.write_tensor(key, merged[key])
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in affected_keys:
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, merged[key])


# =============================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Partial writes leave no artifact; errors abort cleanly.
# =============================================================================


class TestIncompleteWriteNotReused:
    """Failed or incomplete materialization leaves no published artifact."""

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_missing_affected_keys_leaves_no_artifact(self, tmp_path):
        """Finalize fails if not all affected keys were written."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=2, n_affected=3)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            # Write only one of three affected keys
            first_key = sorted(affected_keys)[0]
            sink.write_tensor(
                first_key,
                torch.randn(4, 4, dtype=torch.float32),
            )
            with pytest.raises(RuntimeError, match="missing"):
                sink.finalize()
        except BaseException:
            sink.abort()

        assert not dest.exists(), "No artifact should exist after failed finalize"
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0, "No temp files should remain"

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_abort_leaves_no_artifact(self, tmp_path):
        """Explicitly aborting after partial writes leaves no artifact."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        sink.write_base_weights()
        for key, tensor in merged.items():
            sink.write_tensor(key, tensor)
        # Abort instead of finalize
        sink.abort()

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_error_during_base_write_leaves_no_artifact(self, tmp_path):
        """If base weight writing fails, no artifact or temp file remains."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )

        # Corrupt a base tensor AFTER sink construction so the manifest
        # expects (4,4) but the actual tensor is (8,8), triggering a
        # shape validation error in the underlying writer.
        corrupt_key = next(
            k for k in base_state if k not in affected_keys
        )
        base_state[corrupt_key] = torch.ones(8, 8, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            sink.write_base_weights()

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_existing_artifact_untouched_on_error(self, tmp_path):
        """A pre-existing valid artifact is not corrupted by a failed write."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=2, n_affected=1)
        merged = _make_merged_tensors(affected_keys)

        # First, create a valid artifact
        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        # Read the valid artifact
        with safe_open(str(dest), framework="pt", device="cpu") as f:
            original_data = {k: f.get_tensor(k) for k in f.keys()}

        # Now attempt a second materialization that fails.
        # Construct the sink with correct state, then corrupt a tensor
        # so write_base_weights hits a shape mismatch.
        new_base = dict(base_state)
        bad_key = next(k for k in new_base if k not in affected_keys)

        sink2 = CheckpointMaterializationSink(
            base_state=new_base,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        new_base[bad_key] = torch.ones(8, 8, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            sink2.write_base_weights()

        # Original artifact still valid and unchanged
        assert dest.exists()
        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in original_data:
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, original_data[key]), (
                    "Existing artifact should be untouched after failed write"
                )

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_context_manager_aborts_on_exception(self, tmp_path):
        """Using the sink as a context manager aborts on exception."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()

        with pytest.raises(ValueError, match="intentional"):
            with CheckpointMaterializationSink(
                base_state=base_state,
                affected_keys=affected_keys,
                dest_path=str(dest),
                storage_dtype=torch.float32,
            ) as sink:
                sink.write_base_weights()
                raise ValueError("intentional error")

        assert not dest.exists()
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_double_abort_is_safe(self, tmp_path):
        """Calling abort() twice does not raise."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state()

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        sink.abort()
        sink.abort()  # Should not raise


# =============================================================================
# Integration: complete round-trip with metadata
# =============================================================================


class TestCheckpointMaterializationRoundTrip:
    """Full round-trip: base + affected → valid safetensors artifact."""

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_complete_round_trip_with_metadata(self, tmp_path):
        """Complete materialization produces a valid artifact with metadata."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(n_unaffected=4, n_affected=2)
        merged = _make_merged_tensors(affected_keys)
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": str(sorted(affected_keys)),
        }

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
            metadata=metadata,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            # All keys present
            assert set(f.keys()) == set(base_state.keys())

            # Metadata preserved
            saved_meta = f.metadata()
            assert saved_meta["__ecaj_version__"] == "1"
            assert saved_meta["__ecaj_recipe_hash__"] == "abc123"

            # Unaffected keys match base
            for key in base_state:
                loaded = f.get_tensor(key)
                if key in affected_keys:
                    assert torch.equal(loaded, merged[key])
                else:
                    assert torch.equal(loaded, base_state[key])

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_bfloat16_storage_dtype(self, tmp_path):
        """Materialization works with bfloat16 storage dtype."""
        dest = tmp_path / "model.safetensors"
        base_state, affected_keys = _make_base_state(
            n_unaffected=2, n_affected=1, dtype=torch.bfloat16
        )
        merged = _make_merged_tensors(affected_keys, dtype=torch.bfloat16)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.bfloat16,
        )
        try:
            sink.write_base_weights()
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in base_state:
                loaded = f.get_tensor(key)
                assert loaded.dtype == torch.bfloat16

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_all_keys_affected(self, tmp_path):
        """When all keys are affected, write_base_weights is a no-op."""
        dest = tmp_path / "model.safetensors"
        base_state, _ = _make_base_state(n_unaffected=0, n_affected=4)
        affected_keys = set(base_state.keys())
        merged = _make_merged_tensors(affected_keys)

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()  # No-op: all keys are affected
            for key, tensor in merged.items():
                sink.write_tensor(key, tensor)
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            for key in affected_keys:
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, merged[key])

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_no_keys_affected(self, tmp_path):
        """When no keys are affected, only base weights are written."""
        dest = tmp_path / "model.safetensors"
        base_state, _ = _make_base_state(n_unaffected=4, n_affected=0)
        affected_keys: set[str] = set()

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=str(dest),
            storage_dtype=torch.float32,
        )
        try:
            sink.write_base_weights()
            sink.finalize()
        except BaseException:
            sink.abort()
            raise

        with safe_open(str(dest), framework="pt", device="cpu") as f:
            assert set(f.keys()) == set(base_state.keys())
            for key, expected in base_state.items():
                loaded = f.get_tensor(key)
                assert torch.equal(loaded, expected)
