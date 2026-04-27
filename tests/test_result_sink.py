"""Tests for merge result sink — lib/result_sink.py.

Covers the in-memory sink implementation that preserves current
merged_state dict behavior, plus the ABC contract for write_tensor,
finalize, and abort.

AC: @streaming-full-model-materialization ac-affected-results-released
"""

import pytest
import torch

from lib.result_sink import InMemorySink


class TestInMemorySinkWriteTensor:
    """write_tensor accumulates the exact keys and tensors passed to it."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_records_single_tensor(self):
        """A single write_tensor call is retrievable after finalize."""
        sink = InMemorySink()
        t = torch.randn(4, 4)
        sink.write_tensor("layer.weight", t)
        result = sink.finalize()
        assert "layer.weight" in result
        assert torch.equal(result["layer.weight"], t)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_records_multiple_tensors(self):
        """Multiple write_tensor calls accumulate all keys."""
        sink = InMemorySink()
        tensors = {
            "layer1.weight": torch.randn(4, 4),
            "layer2.weight": torch.randn(8, 8),
            "layer3.bias": torch.randn(16),
        }
        for key, tensor in tensors.items():
            sink.write_tensor(key, tensor)

        result = sink.finalize()
        assert set(result.keys()) == set(tensors.keys())
        for key, tensor in tensors.items():
            assert torch.equal(result[key], tensor)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_preserves_tensor_dtype(self):
        """write_tensor preserves the dtype of each tensor."""
        sink = InMemorySink()
        sink.write_tensor("f16", torch.randn(4, dtype=torch.float16))
        sink.write_tensor("bf16", torch.randn(4, dtype=torch.bfloat16))
        sink.write_tensor("f32", torch.randn(4, dtype=torch.float32))
        result = sink.finalize()
        assert result["f16"].dtype == torch.float16
        assert result["bf16"].dtype == torch.bfloat16
        assert result["f32"].dtype == torch.float32

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_later_write_overwrites_earlier(self):
        """Writing the same key twice keeps the latest tensor."""
        sink = InMemorySink()
        t1 = torch.ones(4)
        t2 = torch.zeros(4)
        sink.write_tensor("key", t1)
        sink.write_tensor("key", t2)
        result = sink.finalize()
        assert torch.equal(result["key"], t2)


class TestInMemorySinkFinalize:
    """finalize returns the collected dict and seals the sink."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_finalize_returns_dict(self):
        """finalize returns a plain dict of key -> tensor."""
        sink = InMemorySink()
        sink.write_tensor("a", torch.randn(2))
        result = sink.finalize()
        assert isinstance(result, dict)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_finalize_empty_sink(self):
        """finalize on an empty sink returns an empty dict."""
        sink = InMemorySink()
        result = sink.finalize()
        assert result == {}

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_write_after_finalize_raises(self):
        """write_tensor after finalize raises RuntimeError."""
        sink = InMemorySink()
        sink.finalize()
        with pytest.raises(RuntimeError):
            sink.write_tensor("key", torch.randn(2))

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_double_finalize_raises(self):
        """Calling finalize twice raises RuntimeError."""
        sink = InMemorySink()
        sink.finalize()
        with pytest.raises(RuntimeError):
            sink.finalize()


class TestInMemorySinkAbort:
    """abort releases temporary resources and is safe from error paths."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_abort_clears_state(self):
        """abort releases internal tensor references."""
        sink = InMemorySink()
        sink.write_tensor("a", torch.randn(4, 4))
        sink.write_tensor("b", torch.randn(4, 4))
        sink.abort()
        # After abort, finalize should raise (sink is sealed)
        with pytest.raises(RuntimeError):
            sink.finalize()

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_write_after_abort_raises(self):
        """write_tensor after abort raises RuntimeError."""
        sink = InMemorySink()
        sink.abort()
        with pytest.raises(RuntimeError):
            sink.write_tensor("key", torch.randn(2))

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_abort_after_finalize_is_safe(self):
        """abort after finalize does not raise — safe for finally blocks."""
        sink = InMemorySink()
        sink.write_tensor("a", torch.randn(2))
        sink.finalize()
        # Should not raise — abort is a cleanup operation
        sink.abort()

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_double_abort_is_safe(self):
        """Calling abort twice does not raise."""
        sink = InMemorySink()
        sink.write_tensor("a", torch.randn(2))
        sink.abort()
        sink.abort()  # Should not raise

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_abort_on_empty_sink(self):
        """abort on a fresh sink does not raise."""
        sink = InMemorySink()
        sink.abort()


class TestInMemorySinkContextManager:
    """InMemorySink works as a context manager for safe cleanup."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_context_manager_normal_exit(self):
        """Context manager calls finalize on normal exit."""
        with InMemorySink() as sink:
            sink.write_tensor("a", torch.randn(2))
            result = sink.finalize()
        assert "a" in result

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_context_manager_aborts_on_exception(self):
        """Context manager calls abort when an exception occurs."""
        sink = InMemorySink()
        with pytest.raises(ValueError, match="test error"):
            with sink:
                sink.write_tensor("a", torch.randn(2))
                raise ValueError("test error")
        # Sink should be aborted — write should fail
        with pytest.raises(RuntimeError):
            sink.write_tensor("b", torch.randn(2))
