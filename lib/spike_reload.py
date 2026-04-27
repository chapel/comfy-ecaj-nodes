"""Spike: saved-artifact reload validation.

Proves whether a WIDEN merged diffusion artifact can be loaded through
ComfyUI's normal model loading path and returned as a MODEL output with
equivalent downstream behavior.

Gate evidence for:
- @full-saved-model-output ac-return-loaded-model
- @full-saved-model-output ac-cache-reuses-artifact
- @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

This module is intentionally isolated as spike/PoC code. It will be removed
or replaced by production implementations in later tasks.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger("ecaj.spike_reload")


def load_saved_model(artifact_path: str) -> object:
    """Load a saved WIDEN artifact via ComfyUI's diffusion model loader.

    Uses comfy.sd.load_diffusion_model — the same public API that ComfyUI's
    built-in "Load Diffusion Model" node uses. This returns a ModelPatcher
    with load_device and offload_device set, making it a normal Comfy-managed
    model that participates in ComfyUI's memory lifecycle.

    Args:
        artifact_path: Path to a safetensors file containing the complete
            merged diffusion model (base weights + merged affected weights).

    Returns:
        A ComfyUI ModelPatcher wrapping the loaded model.

    Raises:
        RuntimeError: If ComfyUI cannot detect the model type from the
            saved artifact's state dict keys.
        ImportError: If comfy.sd is not available (no ComfyUI environment).
    """
    # Use sys.modules lookup to support both real ComfyUI and test mocks.
    # In test environments, comfy.sd is injected via sys.modules without
    # a real package structure.
    comfy_sd = sys.modules.get("comfy.sd")
    if comfy_sd is None:
        comfy_sd = importlib.import_module("comfy.sd")

    model_patcher = comfy_sd.load_diffusion_model(artifact_path)
    logger.info(
        "Loaded saved artifact as ComfyUI MODEL: %s "
        "(load_device=%s, offload_device=%s)",
        artifact_path,
        getattr(model_patcher, "load_device", "unknown"),
        getattr(model_patcher, "offload_device", "unknown"),
    )
    return model_patcher


def detect_memory_mode() -> dict:
    """Detect the active ComfyUI memory management mode.

    Checks whether ComfyUI's Dynamic VRAM (smart memory) is active.
    When Dynamic VRAM is active, ComfyUI manages model weight loading
    and offloading automatically — models returned via load_diffusion_model
    participate in this lifecycle through their load_device/offload_device.

    Returns:
        Dict with keys:
        - dynamic_vram_active: True if Dynamic VRAM is active, False if
          static mode, None if ComfyUI is unavailable.
        - mode_name: "dynamic_vram", "static", or "unavailable".
        - reason: Explanation string (present when unavailable).
    """
    try:
        mm = sys.modules.get("comfy.model_management")
        if mm is None:
            mm = importlib.import_module("comfy.model_management")

        disable_smart = getattr(mm, "DISABLE_SMART_MEMORY", None)
        if disable_smart is None:
            return {
                "dynamic_vram_active": None,
                "mode_name": "unavailable",
                "reason": "DISABLE_SMART_MEMORY attribute not found on comfy.model_management",
            }

        if disable_smart:
            return {
                "dynamic_vram_active": False,
                "mode_name": "static",
            }
        else:
            return {
                "dynamic_vram_active": True,
                "mode_name": "dynamic_vram",
            }

    except ImportError:
        return {
            "dynamic_vram_active": None,
            "mode_name": "unavailable",
            "reason": "comfy.model_management not available (no ComfyUI environment)",
        }
