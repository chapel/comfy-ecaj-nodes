"""MR-01/MR-02 regressions; independent high-precision inference oracle."""

import warnings
from decimal import Decimal, localcontext

import pytest
import torch

from lib.sparsity import Entmax, entmax
from lib.widen import WIDEN, WIDENConfig


def _decimal_entmax(row, alpha):
    # Reviewer oracle retained: scalar root in *unscaled* threshold coordinates,
    # independent of production's tensor arithmetic and threshold bracket.
    with localcontext() as ctx:
        ctx.prec = 70
        a = Decimal(str(alpha)) - 1
        x = [Decimal(str(v)) for v in row]
        if a == 0:
            values = [(v - max(x)).exp() for v in x]
            return [float(v / sum(values)) for v in values]
        lo, hi = min(x) - 1 / a, max(x)
        for _ in range(260):
            mid = (lo + hi) / 2
            mass = sum(max(a * (v - mid), Decimal(0)) ** (1 / a) for v in x)
            if mass > 1:
                lo = mid
            else:
                hi = mid
        threshold = (lo + hi) / 2
        return [float(max(a * (v - threshold), Decimal(0)) ** (1 / a)) for v in x]


# AC: @widen-core ac-3
# MR-01: singleton flat policy is explicit in filter source, not a dedicated AC.
@pytest.mark.parametrize("shape", [(1,), (2, 1), (2, 1, 1), (2, 1, 1, 1)])
@pytest.mark.parametrize("ranking", ["percentile", "minmax", "zscore", "soft", "softrank"])
@pytest.mark.parametrize("t_factor", [0.5, 2.0, 10.0])
def test_singleton_filter_passthrough(shape, ranking, t_factor, caplog):
    merger = WIDEN(WIDENConfig(t_factor=t_factor, ranking_strategy=ranking))
    backbone = torch.arange(1, 1 + 3 * torch.Size(shape).numel()).float().reshape(3, *shape)
    weights = backbone + torch.tensor([4.0, -2.0, 8.0]).reshape(3, *([1] * len(shape)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batched = merger.filter_delta_batched(weights, backbone)
        individual = torch.stack([merger.filter_delta(w, b) for w, b in zip(weights, backbone)])
    torch.testing.assert_close(batched, weights, rtol=0, atol=0)
    torch.testing.assert_close(individual, weights, rtol=0, atol=0)
    assert not caplog.records  # no caught warning / numerical fallback hiding failure
    assert not torch.cuda.is_initialized()


# AC: @widen-core ac-3
@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 2, 1), (1, 2, 1, 1)])
def test_nonsingleton_filter_retains_sample_variance(shape):
    merger = WIDEN(WIDENConfig(t_factor=2.0, ranking_strategy="percentile"))
    backbone = torch.ones(shape)
    weights = backbone + torch.tensor([1.0, 1.00015]).reshape(shape)
    delta = weights - backbone
    eps = merger.numerical_config.get_adaptive_epsilon(delta)
    # Population variance would classify this as flat, sample variance does not.
    assert delta.var(correction=0) < eps < delta.var(correction=1)
    # Both ranks are [1/2, 1]; threshold=2*(3/4), masks=[1/3, 2/3].
    expected = backbone + delta * torch.tensor([1 / 3, 2 / 3]).reshape(shape)
    torch.testing.assert_close(merger.filter_delta(weights, backbone), expected)
    torch.testing.assert_close(
        merger.filter_delta_batched(weights.unsqueeze(0), backbone.unsqueeze(0)),
        expected.unsqueeze(0),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "alpha",
    [1.0, 1.000000000001, 1.00000001, 1.000001, 1.0001, 1.01, 1.099, 1.1, 1.5, 1.99999999, 2.0],
)
@pytest.mark.parametrize("dim", [0, -1])
def test_entmax_unequal_logits_decimal_endpoints(dtype, alpha, dim):
    rows = [[0.0, 1.0, -2.0], [0.0, 0.25, -0.5]]
    x = torch.tensor(rows, dtype=dtype)
    # Exact alpha=2 delegates to sparsemax, whose existing low-precision
    # output is FP32. Do not expand MR-02 into changing that endpoint API.
    output_dtype = (
        torch.float32 if alpha == 2 and dtype in (torch.float16, torch.bfloat16) else dtype
    )
    expected = torch.tensor([_decimal_entmax(row, alpha) for row in rows], dtype=output_dtype)
    if dim == 0:
        x, expected = x.T, expected.T
    actual = entmax(x, alpha=alpha, dim=dim)
    # Lower precision retains output dtype, but computes the sensitive branch
    # at least in FP32. FP64 gets a substantially tighter independent check.
    tolerance = {
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 2e-6,
        torch.float64: 2e-10,
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == output_dtype
    assert actual.device == x.device
    torch.testing.assert_close(Entmax(alpha=alpha, dim=dim)(x), actual, rtol=0, atol=0)
    if alpha == 1.5:
        torch.testing.assert_close(entmax(x, dim=dim), actual, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
