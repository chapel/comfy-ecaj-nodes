"""Tests for the chunked evaluation-to-sink path.

Validates that the sink-based evaluation core in gpu_ops correctly streams
results to a ResultSink incrementally, preserves OOM retry behavior, matches
the dict-returning path, and propagates errors.

AC coverage for:
  @streaming-full-model-materialization ac-affected-results-released
  @streaming-full-model-materialization ac-direct-artifact-handoff
  @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
  @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
"""

from unittest.mock import patch

import pytest
import torch

from lib.gpu_ops import (
    chunked_evaluation,
    evaluate_affected_group,
    evaluate_to_sink,
    streaming_evaluation_to_sink,
)
from lib.result_sink import DictResultSink, WriteFnSink

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _identity_eval_fn(keys, base_batch):
    """Return base_batch unchanged (identity merge)."""
    return base_batch


def _addone_eval_fn(keys, base_batch):
    """Add 1.0 to every element (simple deterministic transform)."""
    return base_batch + 1.0


def _make_base_tensors(keys, shape=(4, 4), dtype=torch.float32):
    """Create base tensors dict for testing."""
    return {k: torch.randn(shape, dtype=dtype) for k in keys}


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# Sink receives keys incrementally before later groups complete
# ===========================================================================


class TestSinkReceivesKeysIncrementally:
    """The sink receives keys one at a time as each chunk completes,
    not all at once after the full evaluation.
    """

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_evaluate_to_sink_delivers_incrementally(self):
        """evaluate_to_sink delivers each key to the sink as it completes,
        not in a batch after all keys are done.
        """
        keys = ["a", "b", "c", "d"]
        base = _make_base_tensors(keys)
        receive_order = []

        class TrackingSink:
            def receive(self, key, tensor):
                receive_order.append(key)

        # batch_size=2 means two chunks: [a, b] then [c, d]
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=TrackingSink(),
        )

        assert receive_order == ["a", "b", "c", "d"]

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_streaming_evaluation_to_sink_delivers_incrementally(self):
        """streaming_evaluation_to_sink delivers each key via write_fn
        as it completes.
        """
        keys = ["x", "y", "z"]
        base = _make_base_tensors(keys)
        receive_order = []

        streaming_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            write_fn=lambda k, t: receive_order.append(k),
        )

        assert receive_order == ["x", "y", "z"]

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_sink_receives_before_later_chunks(self):
        """With batch_size=1, the sink receives each key before the next
        key's evaluation begins.
        """
        keys = ["first", "second", "third"]
        base = _make_base_tensors(keys, shape=(2,))
        events = []

        eval_count = [0]
        original_fn = _identity_eval_fn

        def tracking_eval_fn(k, b):
            eval_count[0] += 1
            events.append(f"eval:{eval_count[0]}")
            return original_fn(k, b)

        class EventSink:
            def receive(self, key, tensor):
                events.append(f"receive:{key}")

        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=tracking_eval_fn,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=EventSink(),
        )

        # Each receive must happen after its eval but before the next eval
        assert events == [
            "eval:1",
            "receive:first",
            "eval:2",
            "receive:second",
            "eval:3",
            "receive:third",
        ]


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# Sink output equals dict-returning output
# ===========================================================================


class TestSinkOutputEqualsDictReturn:
    """Sink-based evaluation produces identical results to dict-returning
    evaluation for the same inputs.
    """

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dict_sink_matches_chunked_evaluation(self):
        """DictResultSink via evaluate_to_sink matches chunked_evaluation."""
        keys = ["w1", "w2", "w3", "w4"]
        base = _make_base_tensors(keys)

        dict_result = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )

        assert set(sink.results.keys()) == set(dict_result.keys())
        for k in keys:
            assert torch.equal(sink.results[k], dict_result[k]), f"Mismatch for key {k}"

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dict_sink_matches_evaluate_affected_group(self):
        """DictResultSink via evaluate_to_sink matches evaluate_affected_group."""
        keys = ["a", "b"]
        base = _make_base_tensors(keys, shape=(8,))

        group_result = evaluate_affected_group(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )

        for k in keys:
            assert torch.equal(sink.results[k], group_result[k])

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_write_fn_sink_matches_streaming_evaluation(self):
        """WriteFnSink via evaluate_to_sink matches streaming_evaluation_to_sink."""
        keys = ["p", "q", "r"]
        base = _make_base_tensors(keys, shape=(3, 3))

        streaming_results = {}
        streaming_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            write_fn=lambda k, t: streaming_results.__setitem__(k, t),
        )

        sink_results = {}
        sink = WriteFnSink(lambda k, t: sink_results.__setitem__(k, t))
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )

        for k in keys:
            assert torch.equal(sink_results[k], streaming_results[k])


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# Dtype conversion matches existing patch path
# ===========================================================================


class TestDtypeConversion:
    """Storage dtype conversion in the sink path matches the dict path."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_storage_dtype_conversion(self):
        """Sink receives tensors in storage_dtype, not compute dtype."""
        keys = ["w"]
        # Base tensor is float16
        base = {"w": torch.randn(4, dtype=torch.float16)}

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,  # compute in fp32
            storage_dtype=torch.float16,  # store in fp16
            sink=sink,
        )

        assert sink.results["w"].dtype == torch.float16

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_none_storage_dtype_uses_base_dtype(self):
        """When storage_dtype is None, the base tensor's own dtype is used."""
        keys = ["w"]
        base = {"w": torch.randn(4, dtype=torch.bfloat16)}

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=None,
            sink=sink,
        )

        assert sink.results["w"].dtype == torch.bfloat16

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dtype_matches_dict_path(self):
        """Sink dtype matches chunked_evaluation dtype for same inputs."""
        keys = ["a", "b"]
        base = _make_base_tensors(keys, dtype=torch.float16)

        dict_result = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
        )

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float16,
            sink=sink,
        )

        for k in keys:
            assert sink.results[k].dtype == dict_result[k].dtype


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# OOM retry still writes all expected keys
# ===========================================================================


class TestOOMRetry:
    """OOM retry in the sink path writes all expected keys."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_oom_retry_writes_all_keys_to_sink(self):
        """After OOM on a batch, retry at batch_size=1 still delivers
        all keys to the sink.
        """
        keys = ["a", "b", "c"]
        base = _make_base_tensors(keys, shape=(4,))

        call_count = [0]

        def oom_then_succeed(k, b):
            call_count[0] += 1
            # First call (the batch) raises OOM, single-key retries succeed
            if call_count[0] == 1 and len(k) > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return b

        sink = DictResultSink()
        with patch("lib.gpu_ops.torch.cuda.is_available", return_value=False):
            evaluate_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=oom_then_succeed,
                batch_size=3,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )

        assert set(sink.results.keys()) == set(keys)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_oom_retry_matches_dict_path(self):
        """OOM retry produces same results via sink as via dict path."""
        keys = ["x", "y"]
        base = _make_base_tensors(keys, shape=(2,))

        call_count_dict = [0]
        call_count_sink = [0]

        def oom_then_succeed_dict(k, b):
            call_count_dict[0] += 1
            if call_count_dict[0] == 1 and len(k) > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return b + 1.0

        def oom_then_succeed_sink(k, b):
            call_count_sink[0] += 1
            if call_count_sink[0] == 1 and len(k) > 1:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return b + 1.0

        with patch("lib.gpu_ops.torch.cuda.is_available", return_value=False):
            dict_result = chunked_evaluation(
                keys=keys,
                base_tensors=base,
                eval_fn=oom_then_succeed_dict,
                batch_size=2,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
            )

        sink = DictResultSink()
        with patch("lib.gpu_ops.torch.cuda.is_available", return_value=False):
            evaluate_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=oom_then_succeed_sink,
                batch_size=2,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )

        for k in keys:
            assert torch.equal(sink.results[k], dict_result[k])


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# Sink outputs are CPU tensors
# ===========================================================================


class TestSinkOutputsAreCPU:
    """Sink receives CPU tensors regardless of computation device."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_sink_receives_cpu_tensors(self):
        """All tensors received by the sink are on CPU."""
        keys = ["a", "b", "c"]
        base = _make_base_tensors(keys)

        sink = DictResultSink()
        evaluate_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=sink,
        )

        for k in keys:
            assert sink.results[k].device.type == "cpu"


# ===========================================================================
# AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
# Errors propagate and prevent sink finalization
# ===========================================================================


class TestErrorPropagation:
    """Errors during evaluation propagate and do not leave partial state."""

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_eval_fn_error_propagates_through_sink(self):
        """An error in eval_fn propagates through evaluate_to_sink."""
        keys = ["a"]
        base = _make_base_tensors(keys)

        def failing_eval(k, b):
            raise RuntimeError("evaluation failed")

        sink = DictResultSink()
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluate_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=failing_eval,
                batch_size=4,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )

        # Sink should have no results (error happened before receive)
        assert len(sink.results) == 0

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_sink_receive_error_propagates(self):
        """An error in the sink's receive method propagates to the caller."""
        keys = ["a"]
        base = _make_base_tensors(keys)

        class FailingSink:
            def receive(self, key, tensor):
                raise RuntimeError("sink write failed")

        with pytest.raises(RuntimeError, match="sink write failed"):
            evaluate_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=_identity_eval_fn,
                batch_size=4,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=FailingSink(),
            )

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_partial_error_delivers_completed_keys_only(self):
        """If the second chunk fails, the first chunk's keys were already
        delivered to the sink (they can't be "un-delivered").
        """
        keys = ["a", "b", "c", "d"]
        base = _make_base_tensors(keys, shape=(2,))

        call_count = [0]

        def fail_on_second_chunk(k, b):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second chunk failed")
            return b

        sink = DictResultSink()
        with pytest.raises(RuntimeError, match="second chunk failed"):
            evaluate_to_sink(
                keys=keys,
                base_tensors=base,
                eval_fn=fail_on_second_chunk,
                batch_size=2,
                device="cpu",
                dtype=torch.float32,
                storage_dtype=torch.float32,
                sink=sink,
            )

        # First chunk's keys were delivered before the error
        assert set(sink.results.keys()) == {"a", "b"}


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# Dict-returning wrapper backward compatibility
# ===========================================================================


class TestDictReturningWrapperBackwardCompat:
    """chunked_evaluation and evaluate_affected_group still return dicts
    and are backward-compatible.
    """

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_chunked_evaluation_returns_dict(self):
        """chunked_evaluation returns a dict of key -> CPU tensor."""
        keys = ["a", "b"]
        base = _make_base_tensors(keys)

        result = chunked_evaluation(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(keys)
        for k in keys:
            expected = base[k] + 1.0
            assert torch.allclose(result[k], expected, atol=1e-6)

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_evaluate_affected_group_returns_dict(self):
        """evaluate_affected_group returns a dict of key -> CPU tensor."""
        keys = ["x"]
        base = _make_base_tensors(keys, shape=(8,))

        result = evaluate_affected_group(
            keys=keys,
            base_tensors=base,
            eval_fn=_addone_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
        )

        assert isinstance(result, dict)
        assert "x" in result
        assert result["x"].device.type == "cpu"

    # AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
    def test_streaming_evaluation_to_sink_returns_none(self):
        """streaming_evaluation_to_sink returns None (write_fn receives tensors)."""
        keys = ["a"]
        base = _make_base_tensors(keys)
        received = {}

        result = streaming_evaluation_to_sink(
            keys=keys,
            base_tensors=base,
            eval_fn=_identity_eval_fn,
            batch_size=4,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            write_fn=lambda k, t: received.__setitem__(k, t),
        )

        assert result is None
        assert "a" in received
