"""Frozen pre-memory-refactor math and real CPU tensor lifetime regressions."""

import weakref

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from lib.sparsity import Entmax, Sparsemax
from lib.widen import WIDEN, WIDENConfig


def _prechange_merge(weights, backbone, config):
    """Frozen ca35bab6 equations; never call WIDEN or its math helpers.

    Deliberately retains all directions/deltas as the prechange implementation
    did. The unchanged standalone sparsity operators are shared for alternates.
    """

    def disentangle(w):
        flat = w.reshape(w.shape[0], w.shape[1], -1)
        scale = flat.abs().amax(dim=1, keepdim=True)
        nonzero = scale > 0
        magnitude = torch.zeros_like(scale)
        if nonzero.any():
            scaled = torch.where(nonzero, flat.float() / scale.float(), 0.0)
            norm = torch.norm(scaled, p=2, dim=1, keepdim=True)
            magnitude = torch.where(nonzero, scale.float() * norm, 0.0).to(w.dtype)
        direction = torch.where(
            magnitude > 64 * torch.finfo(w.dtype).tiny,
            flat / magnitude,
            torch.zeros_like(flat),
        )
        return magnitude.reshape(w.shape[0], 1, *w.shape[2:]), direction.reshape_as(w)

    def rank(values):
        dims = tuple(range(1, values.ndim))
        eps = torch.maximum(
            torch.tensor(1e-12, dtype=values.dtype), 1e-8 * values.float().abs().mean()
        )
        if config.ranking_strategy == "zscore":
            return torch.sigmoid(
                (values - values.mean(dim=dims, keepdim=True))
                / (values.std(dim=dims, keepdim=True) + eps)
            )
        if config.ranking_strategy == "minmax":
            low = values.amin(dim=dims, keepdim=True)
            return (values - low) / (values.amax(dim=dims, keepdim=True) - low + eps)
        flat = values.view(values.shape[0], -1)
        indices = torch.argsort(flat, dim=1, stable=True)
        ranks = torch.linspace(1 / flat.shape[1], 1, flat.shape[1], dtype=flat.dtype)
        return (
            torch.empty_like(flat)
            .scatter_(1, indices, ranks.unsqueeze(0).expand_as(flat))
            .view_as(values)
        )

    def sparsity(values):
        if config.sparsity_method == "sparsemax":
            return Sparsemax(dim=0)(values)
        if config.sparsity_method == "entmax":
            return Entmax(alpha=1.5, dim=0)(values)
        return torch.softmax(values, dim=0)

    def scores(ranked):
        dims = tuple(range(1, ranked[0].ndim))
        mask = torch.stack([r > config.t_factor * r.mean(dim=dims, keepdim=True) for r in ranked])
        result = sparsity(torch.stack(ranked))
        if config.calibration_mode == "overwrite":
            result = torch.where(mask, torch.ones_like(result) * config.s_calibration, result)
        elif config.calibration_mode == "multiplicative":
            result = result * torch.where(mask, config.s_calibration, torch.ones_like(result))
        else:
            return result
        eps = 1e-6 if config.dtype in (torch.float16, torch.bfloat16) else 1e-12
        return result / (result.sum(dim=0, keepdim=True) + eps)

    if config.t_factor <= 0:
        return backbone
    if backbone.ndim == 2:
        deltas = [w - backbone for w in weights]
        s = sparsity(torch.stack([rank(d.abs()) for d in deltas]))
    else:
        mb, db = disentangle(backbone)
        components = [disentangle(w) for w in weights]
        deltas = [w - backbone for w in weights]
        dm = [(m - mb).abs() for m, _ in components]
        dd = [
            (
                1
                - F.cosine_similarity(
                    d.reshape(d.shape[0], d.shape[1], -1),
                    db.reshape(db.shape[0], db.shape[1], -1),
                    dim=1,
                )
            ).reshape_as(mb)
            for _, d in components
        ]
        sm, sd = scores([rank(v) for v in dm]), scores([rank(v) for v in dd])
        s = (sm + sd) / 2
    merged = s[0] * deltas[0]
    for n in range(1, len(weights)):
        merged += s[n] * deltas[n]
    merged += backbone
    return merged


def _inputs(shape, case):
    generator = torch.Generator().manual_seed(417)
    backbone = torch.randn(shape, generator=generator)
    weights = [torch.randn(shape, generator=generator) for _ in range(9)]
    if case == "zero":
        backbone.zero_()
        for w in weights:
            w.zero_()
    elif case == "tiny":
        backbone *= 1e-38
        for w in weights:
            w *= 1e-38
    elif case == "equal":
        weights = [backbone.clone() for _ in weights]
    elif case == "tied":
        backbone.fill_(1)
        for i, w in enumerate(weights):
            w.fill_((i - 4) / 4)
    elif case == "mixed":
        backbone[..., 0] = 0
        backbone[..., 1] *= 1e-30
        for i, w in enumerate(weights):
            w[..., 0] = 0 if i % 2 else 1e-38
            w[..., 1] *= 1e-30
        weights[0] = backbone.clone()
        weights[1] = -backbone
        weights[2] = weights[3].clone()
    return weights, backbone


CONFIGS = [
    WIDENConfig(),
    WIDENConfig(t_factor=1.25),
    WIDENConfig(t_factor=1.25, calibration_mode="multiplicative", s_calibration=0.3),
    WIDENConfig(t_factor=1.25, ranking_strategy="zscore"),
    WIDENConfig(t_factor=1.25, ranking_strategy="minmax"),
    WIDENConfig(t_factor=1.25, ranking_strategy="soft"),
    WIDENConfig(t_factor=1.25, sparsity_method="sparsemax"),
    WIDENConfig(t_factor=1.25, sparsity_method="entmax"),
    WIDENConfig(t_factor=1.25, dtype=torch.bfloat16),
    WIDENConfig(t_factor=0),
    WIDENConfig(t_factor=-1),
]


# AC: @widen-core ac-2
# AC: @widen-core ac-3
# AC: @widen-core ac-5
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("param_shape", [(7,), (5, 8), (5, 4, 3), (5, 3, 2, 4)])
@pytest.mark.parametrize("case", ["random", "zero", "tiny", "equal", "tied", "mixed"])
@pytest.mark.parametrize("config", CONFIGS)
def test_nine_branch_frozen_prechange_parity(batch, param_shape, case, config):
    weights, backbone = _inputs((batch, *param_shape), case)
    originals = [w.clone() for w in [backbone, *weights]]
    expected = _prechange_merge(weights, backbone, config)
    actual = WIDEN(config).merge_weights_batched(weights, backbone)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == backbone.dtype
    for tensor, original in zip([backbone, *weights], originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def _storage_owner(tensor):
    while tensor._base is not None:
        tensor = tensor._base
    return tensor


class _DeltaLifetimes(TorchDispatchMode):
    """Observe actual branch-minus-backbone ATen operations without retaining outputs."""

    def __init__(self, weights, backbone):
        super().__init__()
        self.weights = weights
        self.backbone = backbone
        self.refs = []
        self.live_before_subtract = []
        self.branch_order = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        is_delta = (
            func == torch.ops.aten.sub.Tensor
            and args[1] is self.backbone
            and any(args[0] is w for w in self.weights)
        )
        if is_delta:
            self.live_before_subtract.append(sum(ref() is not None for ref in self.refs))
            self.branch_order.append(next(i for i, w in enumerate(self.weights) if args[0] is w))
        result = func(*args, **(kwargs or {}))
        if is_delta:
            self.refs.append(weakref.ref(_storage_owner(result)))
        return result


# AC: @widen-core ac-2
@pytest.mark.parametrize("shape", [(2, 5, 8), (2, 5, 4, 3), (2, 5, 3, 2, 4)])
def test_direction_storage_released_before_next_branch_and_ranking(monkeypatch, shape):
    weights, backbone = _inputs(shape, "mixed")
    merger = WIDEN(WIDENConfig(t_factor=1.25))
    disentangle = merger._disentangle_batched
    rank = merger.ranker.rank_weights_batched
    refs, before_disentangle, before_rank = [], [], []

    def observe_disentangle(w):
        before_disentangle.append(sum(ref() is not None for ref in refs))
        magnitude, direction = disentangle(w)
        refs.append(weakref.ref(_storage_owner(direction)))
        return magnitude, direction

    def observe_rank(values):
        before_rank.append(sum(ref() is not None for ref in refs))
        return rank(values)

    monkeypatch.setattr(merger, "_disentangle_batched", observe_disentangle)
    monkeypatch.setattr(merger.ranker, "rank_weights_batched", observe_rank)
    actual = merger.merge_weights_batched(weights, backbone)
    torch.testing.assert_close(
        actual, _prechange_merge(weights, backbone, merger.config), rtol=0, atol=0
    )
    assert before_disentangle == [0] + [1] * 9, "retain only the backbone direction"
    assert before_rank == [0] * 18, "release backbone and last branch before ranking"
    assert all(ref() is None for ref in refs)


# AC: @widen-core ac-2
@pytest.mark.parametrize("shape", [(2, 5, 8), (2, 5, 4, 3), (2, 5, 3, 2, 4)])
def test_delta_storage_released_before_next_subtraction(shape):
    weights, backbone = _inputs(shape, "random")
    config = WIDENConfig(t_factor=1.25)
    with _DeltaLifetimes(weights, backbone) as observed:
        actual = WIDEN(config).merge_weights_batched(weights, backbone)
    torch.testing.assert_close(actual, _prechange_merge(weights, backbone, config), rtol=0, atol=0)
    assert observed.branch_order == list(range(9))
    assert observed.live_before_subtract == [0] * 9, "do not retain branch deltas"
    assert all(ref() is None for ref in observed.refs)


# AC: @widen-core ac-9
@pytest.mark.parametrize("oom", [False, True])
def test_merge_failure_contract_unchanged(monkeypatch, caplog, oom):
    weights, backbone = _inputs((2, 5, 8), "random")
    merger = WIDEN()
    error = torch.cuda.OutOfMemoryError("injected") if oom else RuntimeError("injected")

    def fail(*args):
        raise error

    monkeypatch.setattr(merger.divergence_calc, "compute_direction_divergence_batched", fail)
    if oom:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="injected"):
            merger.merge_weights_batched(weights, backbone)
        assert "averaging fallback" not in caplog.text
    else:
        actual = merger.merge_weights_batched(weights, backbone)
        torch.testing.assert_close(actual, torch.stack(weights).mean(dim=0), rtol=0, atol=0)
        assert "averaging fallback" in caplog.text
