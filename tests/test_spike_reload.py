"""Spike: Validate saved-artifact reload before full materialization work.

Gate evidence for @full-saved-model-output ac-return-loaded-model and
ac-cache-reuses-artifact. Final product coverage remains with later tasks.

Tests prove:
1. A saved WIDEN artifact can be loaded via comfy.sd.load_diffusion_model
   and the returned MODEL has equivalent affected weights (parity).
2. On cache hit, the saved artifact is loaded (GPU merge skipped).
3. The returned loaded model reports ComfyUI memory mode compatibility.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from safetensors import safe_open

from lib.persistence import atomic_save, build_metadata
from lib.spike_reload import (
    detect_memory_mode,
    load_saved_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Representative SDXL-like keys (same as conftest.py)
_SDXL_KEYS = (
    "diffusion_model.input_blocks.0.0.weight",
    "diffusion_model.input_blocks.1.0.weight",
    "diffusion_model.middle_block.0.weight",
    "diffusion_model.output_blocks.0.0.weight",
)


def _make_base_state(keys=_SDXL_KEYS, dtype=torch.float32):
    """Create a representative base model state dict."""
    return {k: torch.randn(4, 4, dtype=dtype) for k in keys}


def _make_merged_state(base_state, affected_keys):
    """Create merged state by modifying affected keys."""
    merged = {}
    for k in affected_keys:
        merged[k] = base_state[k] + torch.ones_like(base_state[k])
    return merged


def _save_artifact(path, base_state, merged_state, recipe_hash="test_hash"):
    """Save a representative merged artifact (base + merged overlay)."""
    save_state = dict(base_state)
    for k, v in merged_state.items():
        save_state[k] = v

    affected_keys = sorted(merged_state.keys())
    metadata = build_metadata(
        serialized='{"test": true}',
        recipe_hash=recipe_hash,
        affected_keys=affected_keys,
    )
    atomic_save(save_state, path, metadata)


def _mock_comfy_sd(monkeypatch, load_fn):
    """Wire a mock comfy.sd module with load_diffusion_model into sys.modules.

    The implementation uses sys.modules.get("comfy.sd") first, so we only
    need to set the sys.modules entry. monkeypatch.setitem handles cleanup.
    """
    comfy_sd = ModuleType("comfy.sd")
    comfy_sd.load_diffusion_model = load_fn
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)


def _make_mock_loader(keys=_SDXL_KEYS):
    """Return a (mock_load_fn, calls_list) where mock_load_fn reads the
    saved artifact and returns a MockModelPatcher with those tensors."""
    from tests.conftest import MockModelPatcher

    calls = []

    def mock_load(path, model_options={}):
        calls.append(path)
        patcher = MockModelPatcher(keys=keys)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                patcher._state_dict[key] = f.get_tensor(key)
        return patcher

    return mock_load, calls


# ---------------------------------------------------------------------------
# AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out (gate)
# AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory (gate)
#
# Spike tests for saved-artifact reload viability. These are gate evidence,
# not final AC coverage.
# ---------------------------------------------------------------------------


class TestLoadSavedModel:
    """Prove a WIDEN saved artifact can be loaded via ComfyUI's diffusion
    model loading path and returns a usable ModelPatcher."""

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory (gate)
    def test_load_returns_model_patcher_with_state_dict(
        self, tmp_path, monkeypatch
    ):
        """load_saved_model calls comfy.sd.load_diffusion_model and returns
        an object with model_state_dict() method."""
        base_state = _make_base_state()
        merged_state = _make_merged_state(
            base_state, [_SDXL_KEYS[0], _SDXL_KEYS[1]]
        )
        artifact_path = str(tmp_path / "merged.safetensors")
        _save_artifact(artifact_path, base_state, merged_state)

        mock_load, calls = _make_mock_loader()
        _mock_comfy_sd(monkeypatch, mock_load)

        result = load_saved_model(artifact_path)

        assert calls == [artifact_path], "load_diffusion_model should be called with artifact path"
        assert hasattr(result, "model_state_dict"), "Result should have model_state_dict method"

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory (gate)
    def test_loaded_model_has_device_attributes(self, tmp_path, monkeypatch):
        """The returned model from load_saved_model should have
        load_device and offload_device attributes for ComfyUI memory management."""
        base_state = _make_base_state()
        artifact_path = str(tmp_path / "merged.safetensors")
        _save_artifact(artifact_path, base_state, {})

        class FakeModelPatcher:
            def __init__(self):
                self.load_device = torch.device("cpu")
                self.offload_device = torch.device("cpu")

            def model_state_dict(self):
                return dict(base_state)

        _mock_comfy_sd(
            monkeypatch,
            lambda path, model_options={}: FakeModelPatcher(),
        )

        result = load_saved_model(artifact_path)
        assert hasattr(result, "load_device"), "Should have load_device for ComfyUI memory management"
        assert hasattr(result, "offload_device"), "Should have offload_device for ComfyUI memory management"


class TestWeightParity:
    """Prove affected weights from a loaded saved artifact match the
    current in-memory merge output."""

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory (gate)
    def test_affected_weights_match_after_reload(self, tmp_path, monkeypatch):
        """Affected weights from loaded artifact should be equivalent to
        the in-memory merged state that was saved."""
        base_state = _make_base_state()
        affected_keys = [_SDXL_KEYS[0], _SDXL_KEYS[2]]
        merged_state = _make_merged_state(base_state, affected_keys)
        artifact_path = str(tmp_path / "merged.safetensors")
        _save_artifact(artifact_path, base_state, merged_state)

        mock_load, _ = _make_mock_loader()
        _mock_comfy_sd(monkeypatch, mock_load)

        loaded_patcher = load_saved_model(artifact_path)
        loaded_state = loaded_patcher.model_state_dict()

        # Parity: affected weights should match the merged state exactly
        for key in affected_keys:
            assert torch.allclose(
                loaded_state[key], merged_state[key]
            ), f"Affected weight {key} should match merged state"

        # Unaffected weights should match the base state
        unaffected_keys = [k for k in _SDXL_KEYS if k not in affected_keys]
        for key in unaffected_keys:
            assert torch.allclose(
                loaded_state[key], base_state[key]
            ), f"Unaffected weight {key} should match base state"

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory (gate)
    def test_parity_with_different_storage_dtypes(self, tmp_path, monkeypatch):
        """Parity should hold when the artifact uses bfloat16 storage dtype."""
        base_state = _make_base_state(dtype=torch.bfloat16)
        affected_keys = [_SDXL_KEYS[0]]
        merged_state = {}
        for k in affected_keys:
            merged_state[k] = base_state[k] + torch.ones(4, 4, dtype=torch.bfloat16)
        artifact_path = str(tmp_path / "merged_bf16.safetensors")
        _save_artifact(artifact_path, base_state, merged_state)

        mock_load, _ = _make_mock_loader()
        _mock_comfy_sd(monkeypatch, mock_load)

        loaded_patcher = load_saved_model(artifact_path)
        loaded_state = loaded_patcher.model_state_dict()

        for key in affected_keys:
            assert loaded_state[key].dtype == torch.bfloat16
            assert torch.allclose(loaded_state[key], merged_state[key])


class TestCacheHitReload:
    """Prove the cache-hit shape: when a saved artifact matches the current
    recipe identity, the GPU merge can be skipped and the MODEL loaded
    from the artifact."""

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out (gate)
    def test_cache_hit_loads_artifact_skips_gpu(self, tmp_path, monkeypatch):
        """On cache hit, load_saved_model is called (GPU eval skipped)."""
        base_state = _make_base_state()
        affected_keys = [_SDXL_KEYS[0]]
        merged_state = _make_merged_state(base_state, affected_keys)
        artifact_path = str(tmp_path / "cached.safetensors")
        _save_artifact(
            artifact_path, base_state, merged_state, recipe_hash="match_hash"
        )

        from lib.persistence import check_cache

        # Verify cache hit works
        cached_metadata = check_cache(artifact_path, "match_hash")
        assert cached_metadata is not None, "Should be a cache hit"

        # Now prove load_saved_model can load the artifact
        mock_load, load_calls = _make_mock_loader()
        _mock_comfy_sd(monkeypatch, mock_load)

        # Simulate the cache-hit flow: check_cache → load_saved_model
        loaded = load_saved_model(artifact_path)
        assert load_calls == [artifact_path]

        # Verify parity on the loaded model
        loaded_state = loaded.model_state_dict()
        for key in affected_keys:
            assert torch.allclose(loaded_state[key], merged_state[key])

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out (gate)
    def test_cache_miss_does_not_load(self, tmp_path):
        """On cache miss, check_cache returns None — no artifact load."""
        base_state = _make_base_state()
        artifact_path = str(tmp_path / "cached.safetensors")
        _save_artifact(
            artifact_path, base_state, {}, recipe_hash="original_hash"
        )

        from lib.persistence import check_cache

        result = check_cache(artifact_path, "different_hash")
        assert result is None, "Should be a cache miss"


class TestMemoryModeObservation:
    """Observe ComfyUI memory management mode and record for gate evidence."""

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior (gate)
    def test_detect_memory_mode_returns_dict(self, monkeypatch):
        """detect_memory_mode should return a dict with mode information."""
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        result = detect_memory_mode()
        assert isinstance(result, dict)
        assert "dynamic_vram_active" in result
        assert "mode_name" in result

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior (gate)
    def test_dynamic_vram_active(self, monkeypatch):
        """When DISABLE_SMART_MEMORY is False, dynamic VRAM should be active."""
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        result = detect_memory_mode()
        assert result["dynamic_vram_active"] is True
        assert result["mode_name"] == "dynamic_vram"

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior (gate)
    def test_dynamic_vram_disabled(self, monkeypatch):
        """When DISABLE_SMART_MEMORY is True, dynamic VRAM should be inactive."""
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = True
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        result = detect_memory_mode()
        assert result["dynamic_vram_active"] is False
        assert result["mode_name"] == "static"

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior (gate)
    def test_comfy_unavailable_fallback(self, monkeypatch):
        """When comfy.model_management has no DISABLE_SMART_MEMORY attr,
        should return unavailable mode info."""
        # The autouse fixture in conftest already stubs comfy.model_management
        # as a bare module with no DISABLE_SMART_MEMORY. That's exactly the
        # "unavailable" case: the module exists but lacks the attribute.
        mm = ModuleType("comfy.model_management")
        # No DISABLE_SMART_MEMORY attribute set
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        result = detect_memory_mode()
        assert result["dynamic_vram_active"] is None
        assert result["mode_name"] == "unavailable"
        assert "reason" in result


class TestSavedArtifactKeyFormat:
    """Verify the saved artifact's key format is compatible with
    ComfyUI's diffusion model loading path."""

    def test_saved_keys_have_diffusion_model_prefix(self, tmp_path):
        """Saved artifact should have diffusion_model.* keys that
        comfy.sd.load_diffusion_model_state_dict can strip and recognize."""
        base_state = _make_base_state()
        artifact_path = str(tmp_path / "artifact.safetensors")
        _save_artifact(artifact_path, base_state, {})

        with safe_open(artifact_path, framework="pt", device="cpu") as f:
            saved_keys = list(f.keys())

        for key in saved_keys:
            assert key.startswith("diffusion_model."), (
                f"Key {key} missing diffusion_model. prefix — "
                "ComfyUI's load_diffusion_model_state_dict expects this prefix "
                "to detect and strip it."
            )

    def test_saved_artifact_roundtrip_via_safetensors(self, tmp_path):
        """Saved artifact can be read back with identical tensor values."""
        base_state = _make_base_state()
        affected_keys = list(_SDXL_KEYS[:2])
        merged_state = _make_merged_state(base_state, affected_keys)
        artifact_path = str(tmp_path / "roundtrip.safetensors")
        _save_artifact(artifact_path, base_state, merged_state)

        with safe_open(artifact_path, framework="pt", device="cpu") as f:
            for key in _SDXL_KEYS:
                loaded = f.get_tensor(key)
                if key in merged_state:
                    assert torch.allclose(loaded, merged_state[key])
                else:
                    assert torch.allclose(loaded, base_state[key])

    def test_saved_artifact_has_ecaj_metadata(self, tmp_path):
        """Saved artifact should have ecaj metadata for cache checks."""
        base_state = _make_base_state()
        artifact_path = str(tmp_path / "meta.safetensors")
        _save_artifact(artifact_path, base_state, {}, recipe_hash="rh123")

        with safe_open(artifact_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()

        assert metadata["__ecaj_version__"] == "1"
        assert metadata["__ecaj_recipe_hash__"] == "rh123"
