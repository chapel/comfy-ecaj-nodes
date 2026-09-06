"""CPU-injected CUDA OOM regressions through the real evaluation wrappers."""

import weakref

import pytest
import torch

from lib.gpu_ops import chunked_evaluation, evaluate_to_sink
from lib.result_sink import WriteFnSink


def _run(entry, keys, base, evaluate, batch_size=2, storage_dtype=None, receive=None):
    args = (keys, base, evaluate, batch_size, "cpu", torch.float32, storage_dtype)
    if entry == "dict":
        return chunked_evaluation(*args)
    results = {}

    def collect(key, tensor):
        assert key not in results, "successful sink writes must not be repeated"
        if receive is not None:
            receive(key, tensor)
        results[key] = tensor

    evaluate_to_sink(*args, sink=WriteFnSink(collect))
    return results


# AC: @batched-executor ac-4
# AC: @batched-executor ac-5
# AC: @batched-executor ac-6
@pytest.mark.parametrize("entry", ["dict", "sink"])
@pytest.mark.parametrize("storage_dtype", [None, torch.bfloat16])
def test_failed_attempt_tensors_die_before_retry(entry, storage_dtype, monkeypatch):
    keys = ["a", "b", "c", "d"]
    base = {
        key: torch.tensor([index, index + 0.5], dtype=dtype)
        for index, (key, dtype) in enumerate(
            zip(keys, [torch.float16, torch.bfloat16, torch.float16, torch.bfloat16])
        )
    }
    failed_refs = {}
    calls = []
    delivered = []
    real_to = torch.Tensor.to

    def checked_to(tensor, *args, **kwargs):
        # Check before retry input conversion allocates, not only after the
        # retry has overwritten its base_gpu local and entered eval_fn.
        if failed_refs:
            assert all(ref() is None for ref in failed_refs.values()), (
                "failed attempt tensors are still live before retry conversion"
            )
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", checked_to)

    def evaluate(batch_keys, batch):
        calls.append(list(batch_keys))
        assert batch.dtype == torch.float32
        if len(calls) == 1:
            temporary = torch.ones(3)
            failed_refs.update(base=weakref.ref(batch), temporary=weakref.ref(temporary))
            raise torch.cuda.OutOfMemoryError("injected batch capacity failure")
        assert {name: ref() is None for name, ref in failed_refs.items()} == {
            "base": True, "temporary": True,
        }, "failed attempt tensors are still live at retry entry"
        if entry == "sink" and batch_keys == ["b"]:
            assert delivered == ["a"]
        return batch * 2 + 0.25

    results = _run(
        entry, keys, base, evaluate, storage_dtype=storage_dtype,
        receive=lambda key, tensor: delivered.append(key),
    )
    assert calls == [["a", "b"], ["a"], ["b"], ["c", "d"]]
    assert list(results) == keys
    for key in keys:
        expected_dtype = base[key].dtype if storage_dtype is None else storage_dtype
        assert results[key].device.type == "cpu"
        assert results[key].dtype == expected_dtype
        torch.testing.assert_close(results[key], (base[key].float() * 2 + 0.25).to(expected_dtype))


# AC: @batched-executor ac-4
@pytest.mark.parametrize("entry", ["dict", "sink"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_irreducible_single_key_oom_propagates_without_repeating(entry, batch_size):
    calls = []

    def evaluate(keys, batch):
        calls.append(list(keys))
        raise torch.cuda.OutOfMemoryError("single key cannot fit")

    keys = ["a", "b"]
    with pytest.raises(torch.cuda.OutOfMemoryError, match="single key cannot fit"):
        _run(entry, keys, {key: torch.ones(2) for key in keys}, evaluate, batch_size)
    assert calls == ([["a"]] if batch_size == 1 else [["a", "b"], ["a"]])


# AC: @batched-executor ac-4
@pytest.mark.parametrize("entry", ["dict", "sink"])
@pytest.mark.parametrize("during_retry", [False, True])
def test_non_oom_errors_are_not_retried_or_replaced(entry, during_retry):
    calls = []
    error = ValueError("invalid evaluation")

    def evaluate(keys, batch):
        calls.append(list(keys))
        if during_retry and len(keys) > 1:
            raise torch.cuda.OutOfMemoryError("batch too large")
        raise error

    keys = ["a", "b"]
    with pytest.raises(ValueError) as caught:
        _run(entry, keys, {key: torch.ones(2) for key in keys}, evaluate)
    assert caught.value is error
    assert calls == ([["a", "b"], ["a"]] if during_retry else [["a", "b"]])


# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
def test_sink_oom_does_not_replay_successful_writes():
    calls = []
    delivered = []
    error = torch.cuda.OutOfMemoryError("sink failed")

    def evaluate(keys, batch):
        calls.append(list(keys))
        return batch + 1

    def receive(key, tensor):
        if key == "b":
            raise error
        delivered.append(key)

    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        _run("sink", ["a", "b"], {key: torch.ones(2) for key in ["a", "b"]},
             evaluate, receive=receive)
    assert caught.value is error
    assert calls == [["a", "b"]]
    assert delivered == ["a"]
