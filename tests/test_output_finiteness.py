"""Completed evaluator outputs must be safe before sink delivery."""

# AC: @batched-executor ac-finite-computed-output

import pytest
import torch
from safetensors.torch import load_file, save_file

from lib import gpu_ops
from lib.incremental_writer import IncrementalWriter
from lib.result_sink import DictResultSink
from lib.widen import WIDEN


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
)
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_float8_storage_checks_without_requiring_isfinite_kernel(dtype, retry, overflow):
    base = {k: torch.ones(2, 2) for k in ("a", "bad.weight")}

    def evaluate(keys, batch):
        if retry and len(keys) > 1:
            raise torch.cuda.OutOfMemoryError("batch")
        return batch * (1e10 if overflow else 2)

    received = {}
    # Native cast semantics differ by Torch version (e4m3fn may saturate).
    # This boundary rejects nonfinite output, not otherwise finite rounding.
    native_cast = torch.full((2, 2), 1e10 if overflow else 2).to(dtype)
    if not torch.isfinite(native_cast.float()).all():
        with pytest.raises(ValueError, match="Non-finite.*storage cast"):
            run("streaming", base, evaluate, received.__setitem__, dtype)
        assert not received
    else:
        run("streaming", base, evaluate, received.__setitem__, dtype)
        assert list(received) == list(base)
        for tensor in received.values():
            assert tensor.dtype == dtype
            assert torch.equal(tensor.float(), native_cast.float())


WRAPPERS = ("sink", "dict", "streaming")


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("bad", [False, True])
def test_validation_is_bounded_for_noncontiguous_output(monkeypatch, retry, bad, storage_dtype):
    # Large enough to require several validation windows, not a model payload.
    shape = (257, 513)
    base = {k: torch.zeros(shape) for k in ("a", "bad.weight")}
    seen = []
    original = torch.isfinite

    def bounded(tensor):
        seen.append((tensor.numel(), tensor.dtype))
        assert tensor.numel() <= 65536
        return original(tensor)

    monkeypatch.setattr(torch, "isfinite", bounded)
    original_float = torch.Tensor.float
    casts = []

    def bounded_float(tensor, *args, **kwargs):
        casts.append(tensor.numel())
        assert tensor.numel() <= 65536
        return original_float(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", bounded_float)

    def evaluate(keys, batch):
        if retry and len(keys) > 1:
            raise torch.cuda.OutOfMemoryError("batch")
        result = torch.ones(len(keys), shape[1], shape[0]).transpose(1, 2)
        assert not result.is_contiguous()
        if bad and "bad.weight" in keys:
            result[keys.index("bad.weight"), -1, -1] = -float("inf")
        return result

    received = {}
    if bad:
        with pytest.raises(ValueError, match="bad.weight"):
            run("streaming", base, evaluate, received.__setitem__, storage_dtype)
        assert list(received) == ["a"]
    else:
        run("streaming", base, evaluate, received.__setitem__, storage_dtype)
        assert list(received) == list(base)
        assert torch.equal(received["bad.weight"].to(torch.float32), torch.ones(shape))
    assert len(seen) > 2
    if storage_dtype == torch.float8_e4m3fn:
        assert len(casts) > 1
        assert {dtype for _, dtype in seen} == {torch.float32}
    else:
        assert not casts
        assert {dtype for _, dtype in seen} == {torch.float32, torch.float16}


@pytest.mark.parametrize("shape", [(), (0,), (1, 0, 2)])
def test_empty_and_scalar_outputs(shape):
    base = {"a": torch.ones(shape)}
    result = run("dict", base, lambda keys, batch: batch + 2, None)
    assert torch.equal(result["a"], base["a"] + 2)


def test_unrequested_base_passthrough_is_not_scanned():
    class Base(dict):
        def __getitem__(self, key):
            assert key == "affected"
            return super().__getitem__(key)

    base = Base(affected=torch.ones(2), untouched=torch.tensor([float("nan")]))
    result = gpu_ops.chunked_evaluation(
        ["affected"],
        base,
        lambda keys, batch: batch + 2,
        1,
        "cpu",
        torch.float32,
        None,
    )
    assert list(result) == ["affected"]
    assert torch.equal(result["affected"], torch.full((2,), 3.0))


def run(wrapper, base, evaluate, receive, storage_dtype=None):
    args = (list(base), base, evaluate, 2, "cpu", torch.float32, storage_dtype)
    if wrapper == "dict":
        return gpu_ops.chunked_evaluation(*args)
    if wrapper == "streaming":
        return gpu_ops.streaming_evaluation_to_sink(*args, receive)
    sink = type("Sink", (), {"receive": staticmethod(receive)})()
    return gpu_ops.evaluate_to_sink(*args, sink)


def evaluator(kind, retry, calls):
    def evaluate(keys, batch):
        calls.append(tuple(keys))
        if retry and len(keys) > 1:
            raise torch.cuda.OutOfMemoryError("injected batch capacity")
        result = batch + 2
        if "bad.weight" in keys:
            i = keys.index("bad.weight")
            if kind == "cast":
                result[i] = 70000.0  # finite fp32, outside fp16 range
            elif kind == "nan":
                result[i] = (batch[i] - batch[i]) / (batch[i] - batch[i])
            else:
                factors = torch.full((2, 2), 1e20)
                applied = gpu_ops.apply_lora_batch_gpu(
                    ["bad.weight"],
                    batch[i : i + 1],
                    [gpu_ops.DeltaSpec("standard", 0, up=factors, down=factors)],
                    "cpu",
                    torch.float32,
                )
                result[i] = WIDEN().filter_delta_batched(applied, batch[i : i + 1])[0]
        return result

    return evaluate


# AC: @batched-executor ac-4, ac-5, ac-6
@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("kind", ["lora", "nan", "cast"])
def test_nonfinite_result_never_reaches_sink(wrapper, retry, kind, monkeypatch):
    base = {k: torch.ones(2, 2) for k in ("good.weight", "bad.weight", "later.weight")}
    received = []
    original = DictResultSink.receive

    def record(self, key, tensor):
        received.append((key, tensor.clone()))
        original(self, key, tensor)

    monkeypatch.setattr(DictResultSink, "receive", record)
    calls = []
    with pytest.raises(ValueError, match="Non-finite.*bad.weight") as error:
        run(
            wrapper,
            base,
            evaluator(kind, retry, calls),
            lambda k, t: received.append((k, t.clone())),
            torch.float16,
        )
    assert "torch.float16" in str(error.value) or "computed" in str(error.value)
    assert "Check" in str(error.value)
    assert [key for key, _ in received] == ["good.weight"]
    assert torch.equal(received[0][1], torch.full((2, 2), 3, dtype=torch.float16))
    assert calls == (
        [("good.weight", "bad.weight"), ("good.weight",), ("bad.weight",)]
        if retry
        else [("good.weight", "bad.weight")]
    )


# AC: @batched-executor ac-4, ac-6
@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("retry", [False, True])
def test_finite_mixed_storage_preserves_math(wrapper, retry):
    base = {
        k: torch.tensor([[1.25, -2.5], [0, 8]], dtype=dt)
        for k, dt in zip(("a", "b", "c"), (torch.float16, torch.bfloat16, torch.float32))
    }
    received = {}
    calls = []
    result = run(wrapper, base, evaluator("finite", retry, calls), received.__setitem__)
    result = result if wrapper == "dict" else received
    for key, tensor in base.items():
        assert result[key].dtype == tensor.dtype
        assert torch.equal(result[key], tensor + 2)
    assert calls[-1] == ("c",)


# AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
# AC: @saved-model-artifact-safety ac-no-partial-publication
@pytest.mark.parametrize("retry", [False, True])
def test_caller_abort_preserves_real_artifact(tmp_path, retry):
    target = tmp_path / "existing.safetensors"
    save_file({"old": torch.tensor([17.0])}, str(target))
    old_bytes = target.read_bytes()
    base = {k: torch.ones(2, 2) for k in ("good.weight", "bad.weight")}
    manifest = {k: (torch.float16, (2, 2)) for k in base}
    received = []
    with pytest.raises(ValueError, match="bad.weight"):
        with IncrementalWriter(manifest, str(target)) as writer:

            def receive(key, tensor):
                received.append(key)
                writer.write_tensor(key, tensor)

            run("streaming", base, evaluator("cast", retry, []), receive, torch.float16)
            writer.finalize()
    assert received == ["good.weight"]
    assert target.read_bytes() == old_bytes
    assert load_file(str(target))["old"].item() == 17
    assert list(tmp_path.iterdir()) == [target]


# AC: @widen-core ac-8, ac-9
@pytest.mark.parametrize("method", ["filter", "merge"])
def test_ordinary_widen_error_keeps_finite_fallback(method, monkeypatch, caplog):
    widen = WIDEN()

    def fail(*args):
        raise RuntimeError("ordinary algorithm error")

    monkeypatch.setattr(widen, "_disentangle_batched", fail)
    base = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    def evaluate(keys, batch):
        if method == "filter":
            return widen.filter_delta_batched(batch + 2, batch)
        return widen.merge_weights_batched([batch + 1, batch + 3], batch)

    result = run("dict", base, evaluate, None)
    assert torch.equal(result["weight"], base["weight"] + 2)
    assert "ordinary algorithm error" in caplog.text
