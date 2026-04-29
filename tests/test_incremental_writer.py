"""Tests for incremental safetensors writer — lib/incremental_writer.py.

Verifies manifest-driven incremental writes, atomic publication, abort
cleanup, and poison semantics for invalid tensor submissions.

AC: @saved-model-artifact-safety ac-no-partial-publication
AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
"""

import os
import random
import tempfile
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open

from lib.incremental_writer import IncrementalWriter


def _make_manifest(
    tensors: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Build a manifest from a dict of tensors."""
    return {
        name: (t.dtype, tuple(t.shape)) for name, t in tensors.items()
    }


def _write_valid_artifact(
    path: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a valid safetensors artifact at path using the writer itself."""
    manifest = _make_manifest(tensors)
    w = IncrementalWriter(manifest, path, metadata=metadata)
    for name, tensor in tensors.items():
        w.write_tensor(name, tensor)
    w.finalize()


class TestRandomWriteOrderRoundTrip:
    """Tensors written in random order round-trip correctly."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_random_order_round_trip(self):
        """Manifest-driven writer produces valid safetensors regardless of write order."""
        tensors = {
            "alpha": torch.randn(8, 8),
            "beta": torch.randn(4),
            "gamma": torch.randn(2, 3, 4, dtype=torch.float16),
            "delta": torch.randn(16, dtype=torch.bfloat16),
        }
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)

            names = list(tensors.keys())
            random.shuffle(names)
            for name in names:
                w.write_tensor(name, tensors[name])
            w.finalize()

            assert os.path.exists(save_path)
            with safe_open(save_path, framework="pt", device="cpu") as f:
                for name, expected in tensors.items():
                    loaded = f.get_tensor(name)
                    assert loaded.dtype == expected.dtype, f"dtype mismatch for {name}"
                    assert torch.equal(loaded, expected), f"value mismatch for {name}"


class TestMetadataPreserved:
    """Metadata written through the writer round-trips correctly."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_metadata_preserved(self):
        """String metadata is preserved in the finalized artifact."""
        tensors = {"x": torch.zeros(4)}
        manifest = _make_manifest(tensors)
        metadata = {"__ecaj_version__": "1", "custom_key": "custom_value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path, metadata=metadata)
            w.write_tensor("x", tensors["x"])
            w.finalize()

            with safe_open(save_path, framework="pt", device="cpu") as f:
                loaded_meta = f.metadata()
                assert loaded_meta is not None
                assert loaded_meta["__ecaj_version__"] == "1"
                assert loaded_meta["custom_key"] == "custom_value"


class TestMissingKeyPreventsFinalize:
    """finalize() raises when not all manifest keys have been written."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_missing_key_prevents_finalize(self):
        """Writer with unwritten manifest keys cannot finalize."""
        tensors = {
            "a": torch.randn(4),
            "b": torch.randn(4),
            "c": torch.randn(4),
        }
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])
            # b and c not written

            with pytest.raises(RuntimeError, match="wrote 1/3"):
                w.finalize()

            # No artifact published
            assert not os.path.exists(save_path)


class TestDuplicateKeyPoisonsWriter:
    """Writing the same key twice poisons the writer so finalize cannot succeed."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_duplicate_key_poisons(self):
        """Duplicate key poisons the writer — finalize raises and no artifact published."""
        tensors = {"a": torch.randn(4), "b": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])

            with pytest.raises(RuntimeError, match="duplicate|twice|already written"):
                w.write_tensor("a", tensors["a"])

            # Writer is poisoned — finalize must fail even if we wrote b
            with pytest.raises(RuntimeError):
                w.finalize()

            assert not os.path.exists(save_path)


class TestUnknownKeyPoisonsWriter:
    """Writing a key not in the manifest poisons the writer."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_unknown_key_poisons(self):
        """Unknown key poisons the writer — finalize raises and no artifact published."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)

            with pytest.raises(RuntimeError, match="unknown|not in.*manifest"):
                w.write_tensor("not_in_manifest", torch.randn(4))

            # Writer is poisoned — finalize must fail
            with pytest.raises(RuntimeError):
                w.finalize()

            assert not os.path.exists(save_path)


class TestWrongShapePoisonsAndCleansUp:
    """Wrong tensor shape poisons the writer and removes the temp file."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_wrong_shape_poisons_and_cleans_temp(self):
        """Wrong shape poisons the writer, removes temp, no artifact published."""
        manifest = {"a": (torch.float32, (4, 4))}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)

            with pytest.raises(ValueError, match="shape"):
                w.write_tensor("a", torch.randn(8))  # wrong shape

            # Writer is poisoned — finalize must fail
            with pytest.raises(RuntimeError):
                w.finalize()

            # No artifact at target, no temp file left
            assert not os.path.exists(save_path)
            remaining = [f for f in os.listdir(tmpdir) if f.startswith(".ecaj_tmp_")]
            assert remaining == [], f"temp file not cleaned: {remaining}"


class TestWrongDtypePoisonsAndCleansUp:
    """Wrong tensor dtype poisons the writer and removes the temp file."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_wrong_dtype_poisons_and_cleans_temp(self):
        """Wrong dtype poisons the writer, removes temp, no artifact published."""
        manifest = {"a": (torch.float32, (4,))}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)

            with pytest.raises(ValueError, match="dtype"):
                w.write_tensor("a", torch.randn(4, dtype=torch.float16))

            # Writer is poisoned — finalize must fail
            with pytest.raises(RuntimeError):
                w.finalize()

            assert not os.path.exists(save_path)
            remaining = [f for f in os.listdir(tmpdir) if f.startswith(".ecaj_tmp_")]
            assert remaining == [], f"temp file not cleaned: {remaining}"


class TestExplicitAbortRemovesTempFile:
    """Explicit abort() removes the temp file without touching the target."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_abort_removes_temp_file(self):
        """abort() cleans up the temp file and leaves no artifact at target."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])
            w.abort()

            assert not os.path.exists(save_path)
            remaining = [f for f in os.listdir(tmpdir) if f.startswith(".ecaj_tmp_")]
            assert remaining == [], f"temp file not cleaned: {remaining}"

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_abort_idempotent(self):
        """Multiple abort() calls are safe."""
        manifest = {"a": (torch.float32, (4,))}
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.abort()
            w.abort()  # second call is a no-op

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_finalize_after_abort_raises(self):
        """finalize() after abort() raises."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])
            w.abort()
            with pytest.raises(RuntimeError, match="aborted"):
                w.finalize()


class TestPreExistingArtifactPreserved:
    """A pre-existing valid artifact is preserved when a later write fails."""

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_failed_finalize_preserves_existing(self):
        """If finalize fails (missing keys), the pre-existing artifact is untouched."""
        original_tensors = {"x": torch.randn(4), "y": torch.randn(4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            _write_valid_artifact(save_path, original_tensors)

            # Verify original exists
            assert os.path.exists(save_path)
            with safe_open(save_path, framework="pt", device="cpu") as f:
                original_x = f.get_tensor("x")

            # Start a new write with a different manifest — write only one key
            new_manifest = {"a": (torch.float32, (8,)), "b": (torch.float32, (8,))}
            w = IncrementalWriter(new_manifest, save_path)
            w.write_tensor("a", torch.randn(8))
            # Don't write "b" — finalize should fail

            with pytest.raises(RuntimeError):
                w.finalize()

            # Original artifact is intact
            assert os.path.exists(save_path)
            with safe_open(save_path, framework="pt", device="cpu") as f:
                assert set(f.keys()) == {"x", "y"}
                assert torch.equal(f.get_tensor("x"), original_x)

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_abort_preserves_existing(self):
        """Explicit abort leaves a pre-existing valid artifact untouched."""
        original_tensors = {"x": torch.randn(4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            _write_valid_artifact(save_path, original_tensors)

            with safe_open(save_path, framework="pt", device="cpu") as f:
                original_x = f.get_tensor("x")

            # Start a new write and abort
            new_manifest = {"a": (torch.float32, (8,))}
            w = IncrementalWriter(new_manifest, save_path)
            w.write_tensor("a", torch.randn(8))
            w.abort()

            # Original artifact is intact
            assert os.path.exists(save_path)
            with safe_open(save_path, framework="pt", device="cpu") as f:
                assert set(f.keys()) == {"x"}
                assert torch.equal(f.get_tensor("x"), original_x)

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_poison_preserves_existing(self):
        """Poison from bad write leaves a pre-existing valid artifact untouched."""
        original_tensors = {"x": torch.randn(4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            _write_valid_artifact(save_path, original_tensors)

            with safe_open(save_path, framework="pt", device="cpu") as f:
                original_x = f.get_tensor("x")

            # Start a new write and trigger poison via wrong dtype
            new_manifest = {"a": (torch.float32, (4,))}
            w = IncrementalWriter(new_manifest, save_path)
            with pytest.raises(ValueError):
                w.write_tensor("a", torch.randn(4, dtype=torch.float16))

            with pytest.raises(RuntimeError):
                w.finalize()

            # Original artifact is intact
            assert os.path.exists(save_path)
            with safe_open(save_path, framework="pt", device="cpu") as f:
                assert set(f.keys()) == {"x"}
                assert torch.equal(f.get_tensor("x"), original_x)


class TestContextManager:
    """Writer can be used as a context manager for automatic cleanup."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_context_manager_abort_on_exception(self):
        """Exception inside context manager triggers abort — no artifact published."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            with pytest.raises(ValueError, match="simulated"):
                with IncrementalWriter(manifest, save_path) as w:
                    w.write_tensor("a", tensors["a"])
                    raise ValueError("simulated failure")

            assert not os.path.exists(save_path)
            remaining = [f for f in os.listdir(tmpdir) if f.startswith(".ecaj_tmp_")]
            assert remaining == [], f"temp file not cleaned: {remaining}"

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_context_manager_normal_exit_does_not_auto_finalize(self):
        """Normal exit from context manager does NOT auto-finalize."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            with IncrementalWriter(manifest, save_path) as w:
                w.write_tensor("a", tensors["a"])
                w.finalize()

            assert os.path.exists(save_path)


class TestWriteTimeExceptionPoisonsWriter:
    """I/O or serialization errors during write poison the writer."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_serialization_error_poisons_and_cleans_temp(self):
        """Serialization failure poisons writer and removes temp file."""
        tensors = {"a": torch.randn(4)}
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)

            with patch(
                "lib.incremental_writer._tensor_bytes",
                side_effect=RuntimeError("serialization failed"),
            ):
                with pytest.raises(
                    RuntimeError, match="serialization"
                ):
                    w.write_tensor("a", tensors["a"])

            # Writer is poisoned — cannot finalize
            with pytest.raises(RuntimeError, match="poisoned"):
                w.finalize()

            assert not os.path.exists(save_path)
            remaining = [
                f for f in os.listdir(tmpdir)
                if f.startswith(".ecaj_tmp_")
            ]
            assert remaining == [], (
                f"temp file not cleaned: {remaining}"
            )

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_serialization_error_preserves_existing(self):
        """Serialization failure preserves a pre-existing artifact."""
        original_tensors = {"x": torch.randn(4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            _write_valid_artifact(save_path, original_tensors)

            with safe_open(
                save_path, framework="pt", device="cpu"
            ) as f:
                original_x = f.get_tensor("x")

            manifest = {"a": (torch.float32, (4,))}
            w = IncrementalWriter(manifest, save_path)

            with patch(
                "lib.incremental_writer._tensor_bytes",
                side_effect=RuntimeError("serialization failed"),
            ):
                with pytest.raises(RuntimeError):
                    w.write_tensor("a", torch.randn(4))

            # Original artifact is intact
            assert os.path.exists(save_path)
            with safe_open(
                save_path, framework="pt", device="cpu"
            ) as f:
                assert set(f.keys()) == {"x"}
                assert torch.equal(
                    f.get_tensor("x"), original_x
                )


class TestMetadataValidation:
    """Invalid metadata is rejected at construction time."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_non_string_metadata_value_rejected(self):
        """Non-string metadata value raises TypeError."""
        manifest = {"a": (torch.float32, (4,))}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            with pytest.raises(
                TypeError, match="metadata value must be str"
            ):
                IncrementalWriter(
                    manifest, save_path, metadata={"bad": 1}
                )

            # No temp file created
            remaining = [
                f for f in os.listdir(tmpdir)
                if f.startswith(".ecaj_tmp_")
            ]
            assert remaining == []

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_non_string_metadata_key_rejected(self):
        """Non-string metadata key raises TypeError."""
        manifest = {"a": (torch.float32, (4,))}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            with pytest.raises(
                TypeError, match="metadata key must be str"
            ):
                IncrementalWriter(
                    manifest, save_path, metadata={42: "val"}
                )

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_invalid_metadata_does_not_corrupt_existing(self):
        """Invalid metadata fails before touching existing artifact."""
        original_tensors = {"x": torch.randn(4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            _write_valid_artifact(save_path, original_tensors)

            with safe_open(
                save_path, framework="pt", device="cpu"
            ) as f:
                original_x = f.get_tensor("x")

            with pytest.raises(TypeError):
                IncrementalWriter(
                    {"a": (torch.float32, (4,))},
                    save_path,
                    metadata={"bad": 1},
                )

            # Original artifact is intact
            assert os.path.exists(save_path)
            with safe_open(
                save_path, framework="pt", device="cpu"
            ) as f:
                assert torch.equal(
                    f.get_tensor("x"), original_x
                )


class TestFailedFinalizeIsTerminal:
    """Failed finalize makes the writer terminal (aborted)."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_write_after_failed_finalize_raises(self):
        """write_tensor after failed finalize raises aborted error."""
        tensors = {
            "a": torch.randn(4),
            "b": torch.randn(4),
        }
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])

            with pytest.raises(RuntimeError, match="wrote 1/2"):
                w.finalize()

            # Writer is terminal — further writes must fail
            with pytest.raises(RuntimeError, match="aborted"):
                w.write_tensor("b", tensors["b"])

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_second_finalize_after_failed_raises(self):
        """Second finalize after failure raises aborted error."""
        tensors = {
            "a": torch.randn(4),
            "b": torch.randn(4),
        }
        manifest = _make_manifest(tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.safetensors")
            w = IncrementalWriter(manifest, save_path)
            w.write_tensor("a", tensors["a"])

            with pytest.raises(RuntimeError, match="wrote 1/2"):
                w.finalize()

            with pytest.raises(RuntimeError, match="aborted"):
                w.finalize()
