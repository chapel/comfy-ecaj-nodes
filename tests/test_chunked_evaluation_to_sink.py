"""Tests for sink-writing chunked evaluation — lib/gpu_ops.py.

Covers the chunked_evaluation_to_sink function that writes evaluated
tensors directly to a MergeResultSink, plus behavioral parity with the
dict-returning chunked_evaluation wrapper.

AC: @streaming-full-model-materialization ac-affected-results-released
"""

import torch

from lib.gpu_ops import chunked_evaluation, chunked_evaluation_to_sink
from lib.result_sink import InMemorySink


class TestSinkWritingParity:
    """Dict-returning and sink-writing evaluation produce identical results."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_identical_results_simple_multiply(self):
        """Both paths produce the same tensors for a simple multiplication."""
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        dict_results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        sink_results = sink.finalize()

        assert set(dict_results.keys()) == set(sink_results.keys())
        for k in keys:
            assert torch.equal(dict_results[k], sink_results[k])

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_identical_results_dtype_conversion(self):
        """Both paths produce correct storage_dtype tensors."""
        keys = ["a", "b"]
        base = {k: torch.randn(8, dtype=torch.float16) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu + 1

        dict_results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
        )

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
            sink=sink,
        )
        sink_results = sink.finalize()

        for k in keys:
            assert dict_results[k].dtype == torch.float16
            assert sink_results[k].dtype == torch.float16
            assert torch.equal(dict_results[k], sink_results[k])

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_identical_results_single_key(self):
        """Single-key evaluation produces identical results."""
        keys = ["only"]
        base = {"only": torch.randn(3, 3)}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 0.5

        dict_results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        sink_results = sink.finalize()

        assert torch.equal(dict_results["only"], sink_results["only"])


class TestSinkWritingOomRetry:
    """OOM retry path writes exactly the completed keys to the sink."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_sink_receives_all_keys_after_oom_retry(self):
        """After an OOM retry, the sink has all keys with correct values."""
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(4, 4) for k in keys}

        call_count = [0]

        def eval_fn_with_oom(batch_keys, batch_gpu):
            call_count[0] += 1
            if call_count[0] == 1 and len(batch_keys) > 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu * 2

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn_with_oom,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        results = sink.finalize()

        assert len(results) == 4
        for k in keys:
            assert k in results
            assert torch.allclose(results[k], base[k] * 2, atol=1e-5)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_multiple_chunks_with_oom_on_first(self):
        """OOM on first chunk does not affect later chunks in the sink."""
        keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
        base = {k: torch.randn(4, 4) for k in keys}

        chunk_call_count = [0]

        def eval_fn(batch_keys, batch_gpu):
            chunk_call_count[0] += 1
            if chunk_call_count[0] == 1 and len(batch_keys) > 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu + 1

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        results = sink.finalize()

        assert len(results) == 6
        for k in keys:
            assert torch.allclose(results[k], base[k] + 1, atol=1e-5)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_oom_retry_parity_with_dict_path(self):
        """OOM retry produces identical results between dict and sink paths."""
        keys = ["a", "b", "c", "d"]
        base = {k: torch.randn(4, 4) for k in keys}

        def make_oom_fn():
            call_count = [0]

            def eval_fn(batch_keys, batch_gpu):
                call_count[0] += 1
                if call_count[0] == 1 and len(batch_keys) > 1:
                    raise torch.cuda.OutOfMemoryError("Simulated OOM")
                return batch_gpu * 3

            return eval_fn

        dict_results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=make_oom_fn(),
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=make_oom_fn(),
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        sink_results = sink.finalize()

        for k in keys:
            assert torch.equal(dict_results[k], sink_results[k])


class TestSinkWritingBehavior:
    """Sink-writing evaluation preserves all existing computation semantics."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_results_are_cpu_tensors(self):
        """All tensors written to sink are on CPU."""
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        results = sink.finalize()

        for t in results.values():
            assert t.device == torch.device("cpu")

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_results_match_storage_dtype(self):
        """Sink tensors match the requested storage dtype."""
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, dtype=torch.float16) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu

        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
            sink=sink,
        )
        results = sink.finalize()

        for t in results.values():
            assert t.dtype == torch.float16

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_keys_written_incrementally(self):
        """Keys are written to the sink as chunks complete, not all at once."""
        keys = ["k0", "k1", "k2", "k3"]
        base = {k: torch.randn(2, 2) for k in keys}

        write_order: list[str] = []

        class TrackingSink(InMemorySink):
            def write_tensor(self, key: str, tensor: torch.Tensor) -> None:
                write_order.append(key)
                super().write_tensor(key, tensor)

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu

        sink = TrackingSink()
        chunked_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        sink.finalize()

        # Keys should be written in order, one at a time
        assert write_order == ["k0", "k1", "k2", "k3"]

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_empty_keys_writes_nothing(self):
        """Empty key list writes nothing to the sink."""
        sink = InMemorySink()
        chunked_evaluation_to_sink(
            keys=[],
            base_tensors={},
            eval_fn=lambda k, b: b,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )
        results = sink.finalize()
        assert results == {}


class TestSinkWritingErrorHandling:
    """Error handling preserves existing behavior — sink is NOT auto-aborted."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_memory_error_propagates(self):
        """MemoryError during evaluation propagates as RuntimeError."""
        keys = ["k0"]
        base = {"k0": torch.randn(4, 4)}

        def eval_fn(batch_keys, batch_gpu):
            raise MemoryError("Simulated system OOM")

        import pytest

        sink = InMemorySink()
        with pytest.raises(RuntimeError, match="System memory exhausted"):
            chunked_evaluation_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=eval_fn,
                batch_size=1,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_runtime_error_memory_propagates(self):
        """RuntimeError with 'not enough memory' propagates as RuntimeError."""
        keys = ["k0"]
        base = {"k0": torch.randn(4, 4)}

        def eval_fn(batch_keys, batch_gpu):
            raise RuntimeError("not enough memory to allocate")

        import pytest

        sink = InMemorySink()
        with pytest.raises(RuntimeError, match="System memory exhausted"):
            chunked_evaluation_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=eval_fn,
                batch_size=1,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )


class TestDictReturningWrapper:
    """chunked_evaluation still works identically after refactoring."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dict_wrapper_basic(self):
        """Dict-returning wrapper produces correct results."""
        keys = ["a", "b", "c"]
        base = {k: torch.randn(4, 4) for k in keys}

        def eval_fn(batch_keys, batch_gpu):
            return batch_gpu * 2

        results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        assert len(results) == 3
        for k in keys:
            assert torch.allclose(results[k], base[k] * 2, atol=1e-5)
            assert results[k].device == torch.device("cpu")

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dict_wrapper_oom_retry(self):
        """Dict-returning wrapper handles OOM retry correctly."""
        keys = ["k0", "k1"]
        base = {k: torch.randn(4, 4) for k in keys}

        call_count = [0]

        def eval_fn(batch_keys, batch_gpu):
            call_count[0] += 1
            if call_count[0] == 1 and len(batch_keys) > 1:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch_gpu + 1

        results = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        assert len(results) == 2
        for k in keys:
            assert torch.allclose(results[k], base[k] + 1, atol=1e-5)
