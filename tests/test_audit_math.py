"""Independent numerical and resource regressions from LM-06/10/11 and M3."""

import math

import pytest
import torch

from lib.gpu_ops import chunked_evaluation, evaluate_to_sink
from lib.numerical_config import NumericalConfig
from lib.ranking import RankingMechanism
from lib.sparsity import entmax, sparsemax
from lib.widen import WIDEN, WIDENConfig


def entmax15_reference(x, dim=-1):
    """Closed-form active-set solution of sum(max(x/2 - tau, 0)^2)=1.

    Sort candidate supports, solve their quadratic, then check support. This
    uses no bisection and no production sparsity/ranking helpers.
    """
    rows = x.double().movedim(dim, -1)
    result = torch.empty_like(rows)
    for row, out in zip(rows.reshape(-1, rows.shape[-1]), result.reshape(-1, rows.shape[-1])):
        z = row / 2
        ordered = z.sort(descending=True).values
        for k in range(1, z.numel() + 1):
            s = ordered[:k].sum()
            discriminant = s * s - k * (ordered[:k].square().sum() - 1)
            if discriminant < 0:
                continue
            tau = (s - discriminant.sqrt()) / k
            if ordered[k - 1] >= tau and (k == z.numel() or ordered[k] <= tau):
                out.copy_((z - tau).clamp_min(0).square())
                break
        else:
            raise AssertionError("No feasible simplex support")
    return result.movedim(-1, dim).to(x.dtype)


@pytest.mark.parametrize(
    "values", [[0.0, 1.0], [0.0, 0.0], [7.0], [-20.0, 0.0, 3.0], [1.0, 2.0, 2.0, 3.0]]
)
def test_entmax_matches_independent_active_set(values):
    x = torch.tensor(values, dtype=torch.float64)
    expected = entmax15_reference(x)
    torch.testing.assert_close(entmax(x), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(entmax(x + 100), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(entmax(x.flip(0)).flip(0), expected, rtol=1e-10, atol=1e-10)
    assert entmax(x).sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", [1.0, 1.1, 1.5, 1.9, 2.0])
def test_entmax_ties_and_singleton(alpha):
    torch.testing.assert_close(
        entmax(torch.zeros(2, 4), alpha=alpha, dim=0), torch.full((2, 4), 0.5)
    )
    torch.testing.assert_close(entmax(torch.tensor([3.0]), alpha=alpha), torch.ones(1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5},
        {"alpha": 3.0},
        {"alpha": math.nan},
        {"n_iter": 0},
        {"n_iter": -1},
        {"n_iter": 1.5},
    ],
)
def test_entmax_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        entmax(torch.tensor([0.0, 1.0]), **kwargs)


@pytest.mark.parametrize("dim", [-1, 0])
def test_sparsemax_vector(dim):
    torch.testing.assert_close(
        sparsemax(torch.tensor([0.0, 0.5]), dim=dim), torch.tensor([0.25, 0.75])
    )


# AC: @widen-core ac-3
@pytest.mark.parametrize("strategy", ["zscore", "minmax"])
def test_mixed_scale_ranks_match_independent_keys(strategy):
    x = torch.tensor([[1e-8, 2e-8, 5e-8], [1.0, 2.0, 5.0]])
    ranker = RankingMechanism(strategy)
    expected = torch.stack([ranker.rank_weights(row) for row in x])
    torch.testing.assert_close(ranker.rank_weights_batched(x), expected, rtol=1e-6, atol=1e-7)


# AC: @widen-core ac-3
@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_singleton_zscore_is_neutral(shape):
    ranker = RankingMechanism("zscore")
    x = torch.full(shape, 7.0)
    torch.testing.assert_close(ranker.rank_weights(x), torch.full_like(x, 0.5))
    torch.testing.assert_close(
        ranker.rank_weights_batched(x.unsqueeze(0)), torch.full_like(x.unsqueeze(0), 0.5)
    )


# AC: @widen-core ac-2
@pytest.mark.parametrize("sparsity", ["softmax", "entmax"])
def test_single_column_zscore_merge_is_finite_average(sparsity):
    merger = WIDEN(WIDENConfig(ranking_strategy="zscore", sparsity_method=sparsity, t_factor=2))
    a, b = torch.tensor([[1.0], [3.0]]), torch.tensor([[5.0], [7.0]])
    torch.testing.assert_close(merger.merge_weights([a, b], torch.zeros_like(a)), (a + b) / 2)


# AC: @widen-core ac-3
@pytest.mark.parametrize("strategy", ["percentile", "minmax", "zscore"])
@pytest.mark.parametrize("shape", [(8,), (2, 4), (2, 2, 2), (2, 2, 1, 2)])
@pytest.mark.parametrize("operation", ["filter", "merge"])
def test_actual_chunk_oom_retry_matches_unsplit_math(strategy, shape, operation):
    generator = torch.Generator().manual_seed(2026)
    scale = torch.tensor([1e-5, 1.0]).reshape(2, *([1] * len(shape)))
    base = torch.randn(2, *shape, generator=generator) * scale
    targets = [torch.randn(2, *shape, generator=generator) * scale for _ in range(3)]
    keys = ["small", "large"]
    merger = WIDEN(WIDENConfig(t_factor=1.25, ranking_strategy=strategy))
    calls = []

    def evaluate(selected, batch):
        indices = [keys.index(k) for k in selected]
        if operation == "filter":
            return merger.filter_delta_batched(targets[0][indices], batch)
        return merger.merge_weights_batched([t[indices] for t in targets], batch)

    def oom_evaluate(selected, batch):
        calls.append(tuple(selected))
        if len(selected) > 1:
            raise torch.cuda.OutOfMemoryError("CPU-injected batch OOM")
        return evaluate(selected, batch)

    kwargs = dict(
        keys=keys,
        base_tensors=dict(zip(keys, base)),
        batch_size=2,
        device="cpu",
        dtype=torch.float32,
        storage_dtype=torch.float32,
    )
    normal = chunked_evaluation(eval_fn=evaluate, **kwargs)
    retried = chunked_evaluation(eval_fn=oom_evaluate, **kwargs)
    assert calls == [("small", "large"), ("small",), ("large",)]
    for key in keys:
        torch.testing.assert_close(normal[key], retried[key], rtol=1e-6, atol=1e-12)
    assert not torch.cuda.is_initialized()


# AC: @widen-core ac-8
# AC: @widen-core ac-9
@pytest.mark.parametrize("operation", ["filter", "merge"])
@pytest.mark.parametrize(
    "error",
    [
        MemoryError("RAM"),
        RuntimeError("DefaultCPUAllocator: not enough memory"),
        RuntimeError("out of memory"),
        RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 4096 bytes"
        ),
        RuntimeError("DefaultCPUAllocator: allocation failed (Cannot allocate memory)"),
        torch.cuda.OutOfMemoryError("CUDA"),
    ],
)
def test_resource_errors_propagate_unchanged(monkeypatch, operation, error):
    merger = WIDEN()
    x = torch.ones(1, 2, 3)

    def fail(_):
        raise error

    monkeypatch.setattr(merger, "_disentangle_batched", fail)
    with pytest.raises(type(error)) as caught:
        if operation == "filter":
            merger.filter_delta_batched(x, torch.zeros_like(x))
        else:
            merger.merge_weights_batched([x, x * 2], torch.zeros_like(x))
    assert caught.value is error


# AC: @widen-core ac-8
# AC: @widen-core ac-9
@pytest.mark.parametrize("operation", ["filter", "merge"])
def test_non_resource_fallback_remains(monkeypatch, operation, caplog):
    merger = WIDEN()
    x = torch.ones(1, 2, 3)

    def fail(_):
        raise RuntimeError("numerical kernel failed")

    monkeypatch.setattr(merger, "_disentangle_batched", fail)
    if operation == "filter":
        actual = merger.filter_delta_batched(x, torch.zeros_like(x))
        expected = x
    else:
        actual = merger.merge_weights_batched([x, 2 * x], torch.zeros_like(x))
        expected = 1.5 * x
    torch.testing.assert_close(actual, expected)
    assert "numerical kernel failed" in caplog.text


# AC: @memory-management ac-11
@pytest.mark.parametrize(
    "error", [MemoryError("RAM"), RuntimeError("DefaultCPUAllocator: not enough memory")]
)
def test_resource_error_reaches_outer_sink_without_publication(monkeypatch, error):
    merger = WIDEN()
    delivered = []

    class Sink:
        def receive(self, key, value):
            delivered.append((key, value))

    def fail(_):
        raise error

    monkeypatch.setattr(merger, "_disentangle_batched", fail)
    with pytest.raises(RuntimeError, match="System memory exhausted") as caught:
        evaluate_to_sink(
            keys=["a", "b"],
            base_tensors={k: torch.zeros(2, 3) for k in ["a", "b"]},
            eval_fn=lambda keys, batch: merger.filter_delta_batched(batch + 1, batch),
            batch_size=2,
            device="cpu",
            dtype=torch.float32,
            storage_dtype=torch.float32,
            sink=Sink(),
        )
    assert caught.value.__cause__ is error
    assert delivered == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ranking_strategy": "typo"},
        {"calibration_mode": "typo"},
        {"sparsity_method": "typo"},
        {"t_factor": math.nan},
        {"t_factor": math.inf},
        {"s_calibration": math.nan},
        {"s_calibration": math.inf},
        {"s_calibration": -1.0},
        {"dtype": torch.int32},
    ],
)
def test_invalid_widen_config_fails_before_numerical_fallback(kwargs):
    with pytest.raises(ValueError):
        WIDEN(WIDENConfig(**kwargs))


# AC: @widen-core ac-2
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 2, 3), (2, 2, 1, 3)])
def test_entmax_identical_branches_preserve_weights(shape):
    w = torch.arange(1.0, math.prod(shape) + 1).reshape(shape)
    merger = WIDEN(WIDENConfig(t_factor=2.0, sparsity_method="entmax"))
    torch.testing.assert_close(merger.merge_weights([w, w.clone()], torch.zeros_like(w)), w)
    torch.testing.assert_close(merger.merge_weights([w], torch.zeros_like(w)), w)


# AC: @widen-core ac-3
def test_default_filter_mixed_scale_matches_independent_equation():
    # Zero backbone gives constant direction divergence; percentile ranks tie
    # stably in column order. At t=2 ranks [.5, 1] yield mask [1/3, 2/3].
    a = 1e-4 * torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.stack([a, 1e4 * a])
    expected = weights * torch.tensor([1 / 3, 2 / 3])
    actual = WIDEN(WIDENConfig(t_factor=2.0)).filter_delta_batched(
        weights, torch.zeros_like(weights)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("zero", [True, False])
@pytest.mark.parametrize("keepdim", [True, False])
def test_global_safe_norm_honors_shape(zero, keepdim):
    x = torch.zeros(2, 3) if zero else torch.arange(6.0).reshape(2, 3)
    expected = torch.norm(x, keepdim=keepdim)
    torch.testing.assert_close(NumericalConfig().safe_norm(x, keepdim=keepdim), expected)
