"""Fresh lazy Mapping inputs: storage lifetimes and one fetch per attempt."""

import weakref
from collections import Counter
from collections.abc import Mapping

import pytest
import torch

from lib import gpu_ops
from lib.result_sink import WriteFnSink

ENTRIES = [
    "chunked_evaluation",
    "evaluate_affected_group",
    "evaluate_to_sink",
    "streaming_evaluation_to_sink",
]


def owner(tensor):
    while tensor._base is not None:
        tensor = tensor._base
    return tensor


class LazyInputs(Mapping):
    """Never stores tensors; each fetch allocates a fresh storage-owning base."""

    dtypes = dict(zip("abcd", [torch.float16, torch.bfloat16, torch.float32, torch.float64]))

    def __init__(self):
        self.refs = []
        self.fetches = []
        self.liveness = []

    def __iter__(self):
        return iter(self.dtypes)

    def __len__(self):
        return len(self.dtypes)

    def __getitem__(self, key):
        dtype = self.dtypes[key]
        self.fetches.append(key)
        self.liveness.append((key, sum(ref() is not None for ref in self.refs)))
        tensor = torch.full((4,), ord(key) - ord("a") + 1, dtype=dtype)
        self.refs.append(weakref.ref(owner(tensor)))
        return tensor.view(2, 2)


def run(entry, source, evaluate, storage_dtype=None, receive=None):
    args = (list(source), source, evaluate, 2, "cpu", torch.float32, storage_dtype)
    results = {}

    def collect(key, tensor):
        assert key not in results, "sink replay"
        if receive is not None:
            receive(key, tensor)
        results[key] = tensor

    if entry == "evaluate_to_sink":
        gpu_ops.evaluate_to_sink(*args, sink=WriteFnSink(collect))
    elif entry == "streaming_evaluation_to_sink":
        gpu_ops.streaming_evaluation_to_sink(*args, write_fn=collect)
    else:
        return getattr(gpu_ops, entry)(*args)
    return results


# AC: @memory-management ac-1
# AC: @batched-executor ac-4, ac-5, ac-6
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("storage_dtype", [None, torch.float32])
def test_lazy_storage_dead_at_next_attempt_fetch(entry, retry, storage_dtype):
    source = LazyInputs()
    failed_refs = []
    calls = []

    def evaluate(keys, batch):
        calls.append(keys)
        assert batch.device.type == "cpu"
        assert batch.dtype == torch.float32
        if retry and len(calls) == 1:
            temporary = torch.ones(3)
            failed_refs.extend([weakref.ref(owner(batch)), weakref.ref(temporary)])
            raise torch.cuda.OutOfMemoryError("CPU-injected batch OOM")
        assert all(ref() is None for ref in failed_refs)
        return batch + 0.5

    result = run(entry, source, evaluate, storage_dtype)
    expected = [("a", 0), ("b", 1)]
    if retry:
        expected += [("a", 0), ("b", 0)]
    expected += [("c", 0), ("d", 1)]
    assert source.liveness == expected
    assert all(ref() is None for ref in source.refs)
    assert calls == ([["a", "b"], ["a"], ["b"], ["c", "d"]] if retry else [["a", "b"], ["c", "d"]])
    for key, value in result.items():
        expected_dtype = source.dtypes[key] if storage_dtype is None else storage_dtype
        assert value.dtype == expected_dtype
        assert value.device.type == "cpu"
        torch.testing.assert_close(
            value, torch.full((2, 2), ord(key) - ord("a") + 1.5, dtype=expected_dtype)
        )
    assert not torch.cuda.is_initialized()


# AC: @batched-executor ac-4, ac-6
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("retry", [False, True])
def test_dtype_metadata_does_not_rematerialize_inputs(entry, retry):
    source = LazyInputs()

    def evaluate(keys, batch):
        if retry and keys == ["a", "b"]:
            raise torch.cuda.OutOfMemoryError("CPU-injected batch OOM")
        return batch + 1

    results = run(entry, source, evaluate)
    assert Counter(source.fetches) == Counter(
        {"a": 2 if retry else 1, "b": 2 if retry else 1, "c": 1, "d": 1}
    )
    assert {key: value.dtype for key, value in results.items()} == source.dtypes


# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
@pytest.mark.parametrize("entry", ENTRIES[2:])
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("error_type", [torch.cuda.OutOfMemoryError, RuntimeError, MemoryError])
def test_lazy_sink_failure_never_replays(entry, retry, error_type):
    source = LazyInputs()
    calls = []
    received = []
    error = error_type("sink failed")

    def evaluate(keys, batch):
        calls.append(keys)
        if retry and len(keys) == 2:
            raise torch.cuda.OutOfMemoryError("CPU-injected batch OOM")
        return batch + 1

    def receive(key, tensor):
        received.append(key)
        if key == "b":
            raise error

    with pytest.raises(error_type) as caught:
        run(entry, source, evaluate, receive=receive)
    assert caught.value is error
    assert received == ["a", "b"]
    assert calls == ([["a", "b"], ["a"], ["b"]] if retry else [["a", "b"]])
    assert "c" not in source.fetches


# AC: @memory-management ac-11
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("message", ["cannot allocate memory", "can't allocate memory"])
def test_host_allocation_diagnostics_keep_original_cause(retry, message, monkeypatch):
    calls = []
    cleanup = []
    error = RuntimeError(message)
    monkeypatch.setattr(gpu_ops.gc, "collect", lambda: cleanup.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleanup.append("cache"))

    def evaluate(keys, batch):
        calls.append(keys)
        if retry and len(keys) == 2:
            raise torch.cuda.OutOfMemoryError("CPU-injected batch OOM")
        raise error

    with pytest.raises(RuntimeError, match="System memory exhausted") as caught:
        run("chunked_evaluation", LazyInputs(), evaluate)
    assert caught.value.__cause__ is error
    assert cleanup == ["gc", "cache"] * (2 if retry else 1)
    assert calls == ([["a", "b"], ["a"]] if retry else [["a", "b"]])
