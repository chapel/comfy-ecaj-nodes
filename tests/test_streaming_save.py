"""Tests for streaming safetensors writer — lib/streaming_save.py.

Verifies format compatibility with safetensors.safe_open(), correct
handling of dtypes, metadata, views, and error cases.

AC: @memory-management ac-7
"""

import os
import tempfile

import pytest
import torch
from safetensors import safe_open

from lib.streaming_save import stream_save_file


class TestStreamingSaveRoundTrip:
    """Round-trip tests: write with stream_save_file, read with safe_open."""

    # AC: @memory-management ac-7
    def test_basic_f32(self):
        """F32 tensors round-trip correctly."""
        tensors = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                for name in tensors:
                    loaded = sf.get_tensor(name)
                    assert torch.equal(loaded, tensors[name]), f"Mismatch for {name}"
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_multiple_dtypes(self):
        """All common dtypes round-trip correctly."""
        tensors = {
            "f32": torch.randn(3, 3, dtype=torch.float32),
            "f16": torch.randn(3, 3, dtype=torch.float16),
            "bf16": torch.randn(3, 3, dtype=torch.bfloat16),
            "f64": torch.randn(3, 3, dtype=torch.float64),
            "i32": torch.randint(-100, 100, (3, 3), dtype=torch.int32),
            "i64": torch.randint(-100, 100, (3, 3), dtype=torch.int64),
            "i8": torch.randint(-128, 127, (3, 3), dtype=torch.int8),
            "i16": torch.randint(-100, 100, (3, 3), dtype=torch.int16),
            "u8": torch.randint(0, 255, (3, 3), dtype=torch.uint8),
            "bool": torch.tensor([[True, False, True], [False, True, False], [True, True, False]]),
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                for name, expected in tensors.items():
                    loaded = sf.get_tensor(name)
                    assert loaded.dtype == expected.dtype, f"Dtype mismatch for {name}"
                    assert torch.equal(loaded, expected), f"Value mismatch for {name}"
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_metadata_preserved(self):
        """String metadata round-trips correctly."""
        tensors = {"x": torch.zeros(2)}
        metadata = {"key1": "value1", "key2": "value2"}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path, metadata=metadata)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded_meta = sf.metadata()
                assert loaded_meta is not None
                assert loaded_meta["key1"] == "value1"
                assert loaded_meta["key2"] == "value2"
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_empty_tensor(self):
        """Zero-element tensor round-trips correctly."""
        tensors = {"empty": torch.zeros(0, dtype=torch.float32)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded = sf.get_tensor("empty")
                assert loaded.shape == (0,)
                assert loaded.dtype == torch.float32
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_scalar_tensor(self):
        """Scalar (0-dim) tensor round-trips correctly."""
        tensors = {"scalar": torch.tensor(3.14)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded = sf.get_tensor("scalar")
                assert loaded.shape == ()
                assert torch.equal(loaded, tensors["scalar"])
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_large_tensor(self):
        """Larger tensor (model-like) round-trips correctly."""
        tensors = {"big": torch.randn(512, 512)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded = sf.get_tensor("big")
                assert torch.equal(loaded, tensors["big"])
        finally:
            os.unlink(path)


class TestStreamingSaveViewSafety:
    """View safety: tensor views with storage_offset must round-trip correctly."""

    # AC: @memory-management ac-7
    def test_view_with_storage_offset(self):
        """Tensor view (storage_offset > 0) round-trips correctly.

        This is the critical fix: bytes(untyped_storage()) would include
        data before the view start, corrupting the file. .numpy().tobytes()
        respects view bounds.
        """
        base = torch.randn(10, 10)
        view = base[3:7]  # storage_offset > 0
        assert view.storage_offset() > 0, "Test setup: view must have offset"

        tensors = {"view": view}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded = sf.get_tensor("view")
                assert loaded.shape == view.shape
                assert torch.equal(loaded, view)
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_non_contiguous_tensor(self):
        """Non-contiguous tensor (e.g., transpose) round-trips correctly."""
        base = torch.randn(4, 8)
        transposed = base.t()
        assert not transposed.is_contiguous()

        tensors = {"transposed": transposed}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                loaded = sf.get_tensor("transposed")
                assert loaded.shape == transposed.shape
                assert torch.equal(loaded, transposed)
        finally:
            os.unlink(path)


class TestStreamingSaveErrorCases:
    """Error case handling."""

    # AC: @memory-management ac-7
    def test_rejects_shared_storage(self):
        """Tensors sharing the same data_ptr are rejected."""
        base = torch.randn(10)
        # Two aliases of the same tensor (same data_ptr)
        a = base[:]
        b = base[:]
        assert a.data_ptr() == b.data_ptr()
        tensors = {"a": a, "b": b}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Shared storage"):
                stream_save_file(tensors, path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    # AC: @memory-management ac-7
    def test_rejects_sparse_tensor(self):
        """Sparse tensors are rejected."""
        sparse = torch.randn(3, 3).to_sparse()
        tensors = {"sparse": sparse}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="Sparse"):
                stream_save_file(tensors, path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    # AC: @memory-management ac-7
    def test_empty_dict(self):
        """Empty tensor dict produces valid (header-only) file."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file({}, path)
            with safe_open(path, framework="pt", device="cpu") as sf:
                assert list(sf.keys()) == []
        finally:
            os.unlink(path)

    # AC: @memory-management ac-7
    def test_no_metadata(self):
        """None metadata produces valid file with no __metadata__."""
        tensors = {"x": torch.zeros(2)}
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            stream_save_file(tensors, path, metadata=None)
            with safe_open(path, framework="pt", device="cpu") as sf:
                meta = sf.metadata()
                # safe_open returns empty dict or None when no metadata
                assert meta is None or len(meta) == 0
        finally:
            os.unlink(path)
