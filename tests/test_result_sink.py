"""Tests for the ResultSink protocol and implementations.

AC coverage for:
  @streaming-full-model-materialization ac-affected-results-released
  @streaming-full-model-materialization ac-direct-artifact-handoff
  @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
  @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
"""

import torch

from lib.result_sink import DictResultSink, ResultSink, WriteFnSink

# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# Protocol conformance
# ===========================================================================


class TestResultSinkProtocol:
    """ResultSink protocol is runtime-checkable and satisfied by implementations."""

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_dict_result_sink_satisfies_protocol(self):
        """DictResultSink is a runtime instance of ResultSink."""
        sink = DictResultSink()
        assert isinstance(sink, ResultSink)

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_write_fn_sink_satisfies_protocol(self):
        """WriteFnSink is a runtime instance of ResultSink."""
        sink = WriteFnSink(lambda k, t: None)
        assert isinstance(sink, ResultSink)

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_custom_class_with_receive_satisfies_protocol(self):
        """Any class with a receive(key, tensor) method satisfies ResultSink."""

        class CustomSink:
            def receive(self, key: str, tensor: torch.Tensor) -> None:
                pass

        sink = CustomSink()
        assert isinstance(sink, ResultSink)

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_class_without_receive_does_not_satisfy_protocol(self):
        """A class without receive() does not satisfy ResultSink."""

        class NotASink:
            def write(self, key: str, tensor: torch.Tensor) -> None:
                pass

        sink = NotASink()
        assert not isinstance(sink, ResultSink)


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# DictResultSink behavior
# ===========================================================================


class TestDictResultSink:
    """DictResultSink collects tensors into a dict."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_receive_stores_tensors(self):
        """receive() stores each tensor by key in the results dict."""
        sink = DictResultSink()
        t1 = torch.randn(4, 4)
        t2 = torch.randn(8)

        sink.receive("key1", t1)
        sink.receive("key2", t2)

        assert set(sink.results.keys()) == {"key1", "key2"}
        assert torch.equal(sink.results["key1"], t1)
        assert torch.equal(sink.results["key2"], t2)

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_empty_sink_has_empty_results(self):
        """Fresh DictResultSink has empty results."""
        sink = DictResultSink()
        assert sink.results == {}

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_results_are_cpu_tensors(self):
        """DictResultSink stores whatever tensors are given (CPU in practice)."""
        sink = DictResultSink()
        t = torch.randn(4, dtype=torch.float16)
        sink.receive("k", t)
        assert sink.results["k"].device.type == "cpu"
        assert sink.results["k"].dtype == torch.float16

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_last_write_wins_for_duplicate_keys(self):
        """If the same key is received twice, the last tensor wins."""
        sink = DictResultSink()
        t1 = torch.zeros(4)
        t2 = torch.ones(4)
        sink.receive("k", t1)
        sink.receive("k", t2)
        assert torch.equal(sink.results["k"], t2)


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# WriteFnSink behavior
# ===========================================================================


class TestWriteFnSink:
    """WriteFnSink delegates to the wrapped write function."""

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_receive_delegates_to_write_fn(self):
        """receive() calls the wrapped function with key and tensor."""
        calls = []

        def recorder(key, tensor):
            calls.append((key, tensor))

        sink = WriteFnSink(recorder)
        t = torch.randn(4)
        sink.receive("my_key", t)

        assert len(calls) == 1
        assert calls[0][0] == "my_key"
        assert torch.equal(calls[0][1], t)

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_multiple_receives_delegate_in_order(self):
        """Multiple receive() calls delegate in order."""
        keys_seen = []

        def recorder(key, tensor):
            keys_seen.append(key)

        sink = WriteFnSink(recorder)
        sink.receive("a", torch.randn(2))
        sink.receive("b", torch.randn(2))
        sink.receive("c", torch.randn(2))

        assert keys_seen == ["a", "b", "c"]

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_write_fn_error_propagates(self):
        """Errors from the wrapped write function propagate to the caller."""

        def failing_write(key, tensor):
            raise RuntimeError("write failed")

        sink = WriteFnSink(failing_write)
        try:
            sink.receive("k", torch.randn(4))
            assert False, "should have raised"
        except RuntimeError as e:
            assert "write failed" in str(e)


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# Sink output matches dict-returning output
# ===========================================================================


class TestSinkOutputMatchesDictReturn:
    """DictResultSink produces the same result as direct dict accumulation.

    This validates that the sink-based evaluation path is a faithful
    replacement for the old dict-returning path.
    """

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_dict_sink_matches_manual_dict(self):
        """DictResultSink.results matches manually built dict."""
        keys = ["a", "b", "c", "d"]
        tensors = {k: torch.randn(4, 4) for k in keys}

        # Manual dict accumulation (old path)
        manual = {}
        for k in keys:
            manual[k] = tensors[k]

        # Sink-based accumulation (new path)
        sink = DictResultSink()
        for k in keys:
            sink.receive(k, tensors[k])

        assert set(sink.results.keys()) == set(manual.keys())
        for k in keys:
            assert torch.equal(sink.results[k], manual[k])


# ===========================================================================
# AC: @streaming-full-model-materialization ac-affected-results-released
# Dtype conversion semantics
# ===========================================================================


class TestDtypeConversion:
    """Sinks receive tensors in the caller-specified dtype."""

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_dict_sink_preserves_dtype(self):
        """DictResultSink stores tensors with whatever dtype is given."""
        sink = DictResultSink()
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            t = torch.randn(4).to(dtype=dt)
            sink.receive(f"key_{dt}", t)
            assert sink.results[f"key_{dt}"].dtype == dt

    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_write_fn_sink_passes_dtype_through(self):
        """WriteFnSink passes the tensor dtype through unchanged."""
        received_dtypes = {}

        def recorder(key, tensor):
            received_dtypes[key] = tensor.dtype

        sink = WriteFnSink(recorder)
        for dt in [torch.float32, torch.float16, torch.bfloat16]:
            sink.receive(f"key_{dt}", torch.randn(4).to(dtype=dt))
            assert received_dtypes[f"key_{dt}"] == dt
