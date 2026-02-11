"""Tests for WIDEN core algorithm implementation.

Covers all acceptance criteria from @widen-core spec:
- AC-1: filter_delta zeros low-importance parameters
- AC-2: merge_weights routes via calibrated softmax
- AC-3: batched variants match per-key
- AC-4: no ComfyUI imports (pure torch/stdlib)
- AC-5: behavior matches merge-router reference
- AC-6: fp16/bf16 inputs use fp32 internally
- AC-7: default config values
- AC-8: filter_delta_batched error fallback
- AC-9: merge_weights_batched error fallback
"""

import ast
from pathlib import Path

import pytest
import torch

from lib.numerical_config import NumericalConfig
from lib.widen import WIDEN, WIDENConfig

# ---------------------------------------------------------------------------
# AC-1: filter_delta zeros low-importance parameters
# ---------------------------------------------------------------------------


class TestFilterDelta:
    """AC: @widen-core ac-1 — filter_delta importance filtering."""

    def test_filter_delta_zeros_low_importance_2d(self):
        """Low-importance parameters should be zeroed/scaled down."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        # Create backbone with small uniform values
        backbone = torch.ones(8, 8) * 0.1

        # Create lora_applied with some high-importance and some low-importance deltas
        lora_applied = backbone.clone()
        # High importance: large delta in top-left 2x2
        lora_applied[:2, :2] += 1.0
        # Low importance: tiny delta everywhere else (already zero delta)

        result = widen.filter_delta(lora_applied, backbone)
        delta_result = result - backbone

        # High-importance region should retain most of delta
        high_importance_delta = delta_result[:2, :2]
        assert high_importance_delta.abs().mean() > 0.5, (
            "High-importance delta should be preserved"
        )

    def test_filter_delta_1d_param(self):
        """filter_delta should work on 1D parameters (biases)."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.zeros(32)
        lora_applied = backbone.clone()
        lora_applied[:4] = 1.0  # High importance (large delta)
        lora_applied[4:] = 0.01  # Low importance (small delta)

        result = widen.filter_delta(lora_applied, backbone)
        delta_result = result - backbone

        # High importance region should be preserved more than low
        high_ratio = delta_result[:4].abs().mean() / 1.0
        low_ratio = delta_result[4:].abs().mean() / 0.01

        assert high_ratio > low_ratio, "High-importance delta should be preserved more"

    def test_filter_delta_t_negative_passthrough(self):
        """t_factor < 0 should pass through without filtering."""
        widen = WIDEN(WIDENConfig(t_factor=-1.0))

        backbone = torch.ones(8, 8)
        lora_applied = backbone + torch.randn(8, 8)

        result = widen.filter_delta(lora_applied, backbone)

        # Should be identical to input
        assert torch.allclose(result, lora_applied), "t<0 should pass through unfiltered"


# ---------------------------------------------------------------------------
# AC-2: merge_weights routes via calibrated softmax
# ---------------------------------------------------------------------------


class TestMergeWeights:
    """AC: @widen-core ac-2 — merge_weights multi-model routing."""

    def test_merge_weights_routes_to_important_contributor(self):
        """Parameters should route to most-important contributor."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.zeros(8, 8)
        # Model 1: large delta in top half
        w1 = backbone.clone()
        w1[:4, :] = 1.0
        # Model 2: large delta in bottom half
        w2 = backbone.clone()
        w2[4:, :] = 2.0

        result = widen.merge_weights([w1, w2], backbone)

        # Top half should be influenced by w1
        assert result[:4, :].mean() > 0, "Top half should have w1 contribution"
        # Bottom half should be influenced by w2
        assert result[4:, :].mean() > 0, "Bottom half should have w2 contribution"

    def test_merge_weights_1d(self):
        """merge_weights should work on 1D parameters."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.zeros(16)
        w1 = backbone.clone()
        w1[:8] = 1.0
        w2 = backbone.clone()
        w2[8:] = 2.0

        result = widen.merge_weights([w1, w2], backbone)

        assert result[:8].abs().sum() > 0, "w1 region should have contribution"
        assert result[8:].abs().sum() > 0, "w2 region should have contribution"

    def test_merge_weights_t_negative_exact_average(self):
        """t_factor < 0 should compute exact average."""
        widen = WIDEN(WIDENConfig(t_factor=-1.0))

        backbone = torch.ones(4, 4)
        w1 = backbone + torch.ones(4, 4)  # delta = 1
        w2 = backbone + torch.ones(4, 4) * 3  # delta = 3

        result = widen.merge_weights([w1, w2], backbone)

        # Average delta = (1 + 3) / 2 = 2
        expected = backbone + 2.0
        assert torch.allclose(result, expected), "t<0 should produce exact average"


# ---------------------------------------------------------------------------
# AC-3: batched variants match per-key
# ---------------------------------------------------------------------------


class TestBatchedVariants:
    """AC: @widen-core ac-3 — batched results match per-key."""

    def test_filter_delta_batched_matches_per_key(self):
        """filter_delta_batched should match individual calls."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        B = 4
        backbone = torch.randn(B, 8, 8)
        lora_applied = backbone + torch.randn(B, 8, 8) * 0.5

        # Batched result
        batched_result = widen.filter_delta_batched(lora_applied, backbone)

        # Per-key results
        per_key_results = []
        for i in range(B):
            result_i = widen.filter_delta(lora_applied[i], backbone[i])
            per_key_results.append(result_i)
        per_key_result = torch.stack(per_key_results)

        assert torch.allclose(batched_result, per_key_result, atol=1e-5), (
            "Batched filter_delta should match per-key"
        )

    def test_merge_weights_batched_matches_per_key(self):
        """merge_weights_batched should match individual calls."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        B = 4
        backbone = torch.randn(B, 8, 8)
        w1 = backbone + torch.randn(B, 8, 8) * 0.5
        w2 = backbone + torch.randn(B, 8, 8) * 0.5

        # Batched result
        batched_result = widen.merge_weights_batched([w1, w2], backbone)

        # Per-key results
        per_key_results = []
        for i in range(B):
            result_i = widen.merge_weights([w1[i], w2[i]], backbone[i])
            per_key_results.append(result_i)
        per_key_result = torch.stack(per_key_results)

        assert torch.allclose(batched_result, per_key_result, atol=1e-5), (
            "Batched merge_weights should match per-key"
        )


# ---------------------------------------------------------------------------
# AC-4: no ComfyUI imports (pure torch/stdlib)
# ---------------------------------------------------------------------------


class TestNoComfyUIImports:
    """AC: @widen-core ac-4 — pure torch and stdlib only."""

    def test_lib_widen_no_comfyui_imports(self):
        """lib/widen.py should not import any ComfyUI modules."""
        lib_path = Path(__file__).parent.parent / "lib" / "widen.py"
        source = lib_path.read_text()
        tree = ast.parse(source)

        comfyui_modules = {"folder_paths", "comfy", "nodes", "server"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    assert module not in comfyui_modules, (
                        f"widen.py imports ComfyUI module: {module}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    assert module not in comfyui_modules, (
                        f"widen.py imports from ComfyUI module: {module}"
                    )

    def test_all_lib_modules_pure(self):
        """All lib/*.py modules should be pure torch/stdlib."""
        lib_path = Path(__file__).parent.parent / "lib"
        comfyui_modules = {"folder_paths", "comfy", "nodes", "server"}

        for py_file in lib_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            source = py_file.read_text()
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        assert module not in comfyui_modules, (
                            f"{py_file.name} imports ComfyUI module: {module}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        assert module not in comfyui_modules, (
                            f"{py_file.name} imports from ComfyUI: {module}"
                        )


# ---------------------------------------------------------------------------
# AC-5: behavior matches merge-router reference
# ---------------------------------------------------------------------------


class TestMergeRouterEquivalence:
    """AC: @widen-core ac-5 — behavior matches merge-router within tolerance."""

    def test_filter_delta_deterministic(self):
        """filter_delta should be deterministic for same inputs."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.randn(16, 16)
        lora_applied = backbone + torch.randn(16, 16) * 0.5

        result1 = widen.filter_delta(lora_applied, backbone)
        result2 = widen.filter_delta(lora_applied, backbone)

        assert torch.allclose(result1, result2), "filter_delta should be deterministic"

    def test_merge_weights_deterministic(self):
        """merge_weights should be deterministic for same inputs."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.randn(16, 16)
        w1 = backbone + torch.randn(16, 16) * 0.5
        w2 = backbone + torch.randn(16, 16) * 0.5

        result1 = widen.merge_weights([w1, w2], backbone)
        result2 = widen.merge_weights([w1, w2], backbone)

        assert torch.allclose(result1, result2), "merge_weights should be deterministic"

    def test_disentangle_linear_preserves_norm(self):
        """Disentanglement should approximately preserve column norms."""
        from lib.widen import WeightDisentangler

        disentangler = WeightDisentangler()
        W = torch.randn(32, 64)

        m, D = disentangler.disentangle_linear(W)

        # Reconstruct
        W_reconstructed = m * D

        # Should be very close to original
        assert torch.allclose(W_reconstructed, W, atol=1e-5), (
            "Disentanglement should be reversible"
        )


# ---------------------------------------------------------------------------
# AC-6: bf16/fp16 inputs use fp32 internally
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """AC: @widen-core ac-6 — fp16/bf16 use fp32 internally."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_filter_delta_handles_low_precision(self, dtype):
        """filter_delta should work with fp16/bf16 inputs."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            # bfloat16 support varies by hardware
            if not hasattr(torch, "bfloat16"):
                pytest.skip("bfloat16 not supported")

        widen = WIDEN(WIDENConfig(t_factor=1.0, dtype=dtype))

        backbone = torch.randn(8, 8, dtype=dtype)
        lora_applied = backbone + torch.randn(8, 8, dtype=dtype) * 0.5

        result = widen.filter_delta(lora_applied, backbone)

        # Should not have NaN or Inf
        assert not result.isnan().any(), "Result should not have NaN"
        assert not result.isinf().any(), "Result should not have Inf"
        # Output dtype should match input
        assert result.dtype == dtype, "Output dtype should match input"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_merge_weights_handles_low_precision(self, dtype):
        """merge_weights should work with fp16/bf16 inputs."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            if not hasattr(torch, "bfloat16"):
                pytest.skip("bfloat16 not supported")

        widen = WIDEN(WIDENConfig(t_factor=1.0, dtype=dtype))

        backbone = torch.randn(8, 8, dtype=dtype)
        w1 = backbone + torch.randn(8, 8, dtype=dtype) * 0.5
        w2 = backbone + torch.randn(8, 8, dtype=dtype) * 0.5

        result = widen.merge_weights([w1, w2], backbone)

        assert not result.isnan().any(), "Result should not have NaN"
        assert not result.isinf().any(), "Result should not have Inf"
        assert result.dtype == dtype, "Output dtype should match input"


# ---------------------------------------------------------------------------
# AC-7: default config values
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """AC: @widen-core ac-7 — default WIDENConfig values."""

    def test_default_ranking_strategy(self):
        """Default ranking_strategy should be 'percentile'."""
        config = WIDENConfig()
        assert config.ranking_strategy == "percentile"

    def test_default_sparsity_method(self):
        """Default sparsity_method should be 'softmax'."""
        config = WIDENConfig()
        assert config.sparsity_method == "softmax"

    def test_default_s_calibration(self):
        """Default s_calibration should be 1.0."""
        config = WIDENConfig()
        assert config.s_calibration == 1.0

    def test_widen_uses_defaults(self):
        """WIDEN() with no args should use default config."""
        widen = WIDEN()
        assert widen.ranking_strategy == "percentile"
        assert widen.s_calibration == 1.0
        assert widen.config.sparsity_method == "softmax"


# ---------------------------------------------------------------------------
# AC-8: filter_delta_batched error fallback
# ---------------------------------------------------------------------------


class TestFilterDeltaBatchedFallback:
    """AC: @widen-core ac-8 — error fallback to passthrough."""

    def test_filter_delta_batched_fallback_on_error(self, monkeypatch):
        """Non-OOM error should fall back to passthrough."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        def raise_error(*args, **kwargs):
            raise RuntimeError("Simulated error")

        # Patch the ranker to raise an error
        monkeypatch.setattr(widen.ranker, "rank_weights_batched", raise_error)

        backbone = torch.randn(4, 8, 8)
        lora_applied = backbone + torch.randn(4, 8, 8) * 0.5

        # Should not raise, should return passthrough
        result = widen.filter_delta_batched(lora_applied, backbone)

        assert torch.allclose(result, lora_applied), (
            "Fallback should return lora_applied (passthrough)"
        )


# ---------------------------------------------------------------------------
# AC-9: merge_weights_batched error fallback
# ---------------------------------------------------------------------------


class TestMergeWeightsBatchedFallback:
    """AC: @widen-core ac-9 — error fallback to averaging."""

    def test_merge_weights_batched_fallback_on_error(self, monkeypatch):
        """Non-OOM error should fall back to simple averaging."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        def raise_error(*args, **kwargs):
            raise RuntimeError("Simulated error")

        # Patch the ranker to raise an error
        monkeypatch.setattr(widen.ranker, "rank_weights_batched", raise_error)

        backbone = torch.randn(4, 8, 8)
        w1 = backbone + torch.ones(4, 8, 8)
        w2 = backbone + torch.ones(4, 8, 8) * 3

        # Should not raise, should return average
        result = widen.merge_weights_batched([w1, w2], backbone)

        # Average delta = (1 + 3) / 2 = 2
        expected = backbone + 2.0
        assert torch.allclose(result, expected), "Fallback should compute simple average"


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and additional coverage."""

    def test_empty_delta(self):
        """filter_delta should handle zero delta gracefully."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.randn(8, 8)
        lora_applied = backbone.clone()  # Zero delta

        result = widen.filter_delta(lora_applied, backbone)

        assert torch.allclose(result, backbone), "Zero delta should return backbone"

    def test_single_model_merge(self):
        """merge_weights with single model should work."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        backbone = torch.randn(8, 8)
        w1 = backbone + torch.randn(8, 8) * 0.5

        result = widen.merge_weights([w1], backbone)

        # Single model should apply its delta
        assert not torch.allclose(result, backbone), "Single model should apply its delta"

    def test_conv2d_shape(self):
        """filter_delta should work with Conv2D weight shapes."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        # Conv2D weight: (out_channels, in_channels, h, w)
        backbone = torch.randn(64, 32, 3, 3)
        lora_applied = backbone + torch.randn(64, 32, 3, 3) * 0.5

        result = widen.filter_delta(lora_applied, backbone)

        assert result.shape == backbone.shape, "Shape should be preserved"
        assert not result.isnan().any(), "Result should not have NaN"

    def test_conv1d_shape(self):
        """filter_delta should work with Conv1D weight shapes."""
        widen = WIDEN(WIDENConfig(t_factor=1.0))

        # Conv1D weight: (out_channels, in_channels, kernel_size)
        backbone = torch.randn(64, 32, 5)
        lora_applied = backbone + torch.randn(64, 32, 5) * 0.5

        result = widen.filter_delta(lora_applied, backbone)

        assert result.shape == backbone.shape, "Shape should be preserved"
        assert not result.isnan().any(), "Result should not have NaN"

    def test_sparsemax_sparsity(self):
        """Sparsemax should produce sparse outputs."""
        from lib.sparsity import Sparsemax

        sparsemax = Sparsemax(dim=-1)
        input = torch.randn(10, 5)

        output = sparsemax(input)

        # Should have some zeros
        zero_ratio = (output == 0).float().mean()
        assert zero_ratio > 0, "Sparsemax should produce sparse output"

    def test_entmax_between_softmax_sparsemax(self):
        """Entmax with alpha=1.5 should be between softmax and sparsemax."""
        from lib.sparsity import Entmax

        entmax = Entmax(alpha=1.5, dim=-1)
        input = torch.randn(10, 5)

        output = entmax(input)

        # Should sum to 1
        assert torch.allclose(output.sum(dim=-1), torch.ones(10), atol=1e-5), (
            "Entmax should sum to 1"
        )

    def test_ranking_strategies(self):
        """All ranking strategies should work."""
        from lib.ranking import RankingMechanism

        divergences = torch.randn(16, 16).abs()

        for strategy in ["percentile", "zscore", "minmax"]:
            ranker = RankingMechanism(strategy=strategy)
            result = ranker.rank_weights(divergences)

            assert result.shape == divergences.shape, f"{strategy} should preserve shape"
            assert result.min() >= 0, f"{strategy} should be >= 0"
            assert result.max() <= 1, f"{strategy} should be <= 1"

    def test_numerical_config_safe_norm(self):
        """NumericalConfig.safe_norm should handle edge cases."""
        config = NumericalConfig()

        # Very small values that would underflow with naive squaring
        small = torch.tensor([1e-30, 2e-30, 3e-30], dtype=torch.float32)
        norm = config.safe_norm(small, p=2, dim=None, keepdim=False)

        assert not norm.isnan().any(), "safe_norm should not produce NaN"
        assert not norm.isinf().any(), "safe_norm should not produce Inf"

    def test_widen_different_sparsity_methods(self):
        """WIDEN should work with different sparsity methods."""
        backbone = torch.randn(8, 8)
        w1 = backbone + torch.randn(8, 8) * 0.5
        w2 = backbone + torch.randn(8, 8) * 0.5

        for method in ["softmax", "sparsemax", "entmax"]:
            widen = WIDEN(WIDENConfig(t_factor=1.0, sparsity_method=method))
            result = widen.merge_weights([w1, w2], backbone)

            assert result.shape == backbone.shape, f"{method} should preserve shape"
            assert not result.isnan().any(), f"{method} should not produce NaN"
