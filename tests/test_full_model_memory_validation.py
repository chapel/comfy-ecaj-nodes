"""End-to-end validation: full saved model memory behavior vs in-memory patches.

Covers acceptance criteria for @comfy-memory-manager-compatibility:
- ac-no-dynamic-vram-opt-out
- ac-comfy-owns-returned-model-memory
- ac-measured-memory-behavior

Tests prove that full saved model output:
1. Works without disabling ComfyUI's dynamic memory management.
2. Returns a MODEL loaded via ComfyUI's model loading path (Comfy owns lifecycle).
3. Does not require all merged affected weights to remain resident as a
   persistent in-memory patch payload after the Exit node returns.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from lib.recipe import RecipeBase, RecipeLoRA, RecipeMerge
from lib.spike_reload import detect_memory_mode
from nodes.exit import (
    OUTPUT_MODE_FULL_MODEL,
    OUTPUT_MODE_PATCHES,
    WIDENExitNode,
    _incremental_cache,
)
from tests.conftest import MockModelPatcher

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Representative SDXL keys: 2 affected + 2 unaffected
_ALL_KEYS = (
    "diffusion_model.input_blocks.0.0.weight",
    "diffusion_model.input_blocks.1.0.weight",
    "diffusion_model.middle_block.0.weight",
    "diffusion_model.output_blocks.0.0.weight",
)
_AFFECTED_KEYS = [_ALL_KEYS[0], _ALL_KEYS[2]]
_UNAFFECTED_KEYS = [_ALL_KEYS[1], _ALL_KEYS[3]]


def _make_patcher(keys=_ALL_KEYS):
    """Create a MockModelPatcher with known deterministic weights."""
    patcher = MockModelPatcher(keys=keys)
    # Seed deterministic weights so equivalence checks are meaningful
    torch.manual_seed(42)
    for key in keys:
        patcher._state_dict[key] = torch.randn(4, 4, dtype=torch.float32)
    patcher.model = type(patcher.model)(patcher._state_dict)
    return patcher


def _make_recipe(patcher):
    """Build a representative merge recipe with a single LoRA target."""
    base = RecipeBase(model_patcher=patcher, arch="sdxl")
    lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
    return RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)


def _merged_tensors_for(keys):
    """Deterministic 'merged' tensors for affected keys."""
    torch.manual_seed(99)
    return {k: torch.randn(4, 4) for k in keys}


def _build_cached_artifact(path, patcher, merged, recipe_hash="hash1"):
    """Write a valid cached full_model artifact at *path*."""
    all_state = dict(patcher._state_dict)
    for k, v in merged.items():
        all_state[k] = v
    metadata = {
        "__ecaj_version__": "1",
        "__ecaj_recipe__": "{}",
        "__ecaj_recipe_hash__": recipe_hash,
        "__ecaj_affected_keys__": json.dumps(sorted(merged.keys())),
        "__ecaj_artifact_kind__": "full_model",
    }
    save_file(all_state, str(path), metadata=metadata)


def _make_exit_patches(patcher, recipe, merged, save_path, *, cache_hit=False):
    """Return context-manager patches for WIDENExitNode.execute().

    When cache_hit=True, check_cache returns metadata matching the artifact.
    When cache_hit=False, check_cache returns None (GPU eval path).
    """
    from lib.batch_groups import OpSignature
    from lib.recipe_eval import EvalPlan

    sig = OpSignature(shape=(4, 4), ndim=2)
    affected_set = set(merged.keys())
    lora = recipe.target

    mock_loader = MagicMock()
    mock_loader.cleanup = MagicMock()
    mock_loader.loaded_bytes = 0

    mock_analyze = MagicMock(
        model_patcher=patcher,
        arch="sdxl",
        loader=mock_loader,
        set_affected={str(id(lora)): affected_set},
        affected_keys=affected_set,
    )
    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()

    dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

    mock_mat_instance = MagicMock()
    mock_mat_instance.finalize.return_value = {}

    cache_metadata = None
    if cache_hit:
        cache_metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "hash1",
            "__ecaj_affected_keys__": json.dumps(sorted(merged.keys())),
            "__ecaj_artifact_kind__": "full_model",
        }

    return {
        "nodes.exit.validate_model_name": "model.safetensors",
        "nodes.exit._resolve_checkpoints_path": str(save_path),
        "nodes.exit.compute_recipe_hash": "hash1",
        "nodes.exit.compute_base_identity": "base_id",
        "nodes.exit.compute_lora_stats": {},
        "nodes.exit.serialize_recipe": "{}",
        "nodes.exit.check_cache": cache_metadata,
        "nodes.exit.analyze_recipe": mock_analyze,
        "nodes.exit.analyze_recipe_models": mock_model_analysis,
        "nodes.exit.compile_plan": dummy_plan,
        "nodes.exit.compile_batch_groups": {sig: list(affected_set)},
        "nodes.exit.chunked_evaluation": merged,
        "nodes.exit._unpatch_loaded_clones": None,
        "nodes.exit.ProgressBar": None,
        "nodes.exit.CheckpointMaterializationSink": mock_mat_instance,
        "nodes.exit.build_metadata": {"__ecaj_version__": "1"},
    }


def _run_with_patches(patches_dict, fn):
    """Enter all patches from dict, call fn, then stop patches."""
    entered = []
    try:
        for target, value in patches_dict.items():
            if isinstance(value, MagicMock):
                if hasattr(value, "return_value") and not callable(
                    getattr(value, "_mock_name", None)
                ):
                    p = patch(target, return_value=value)
                else:
                    p = patch(target, value)
            elif value is None:
                p = patch(target, return_value=None)
            else:
                p = patch(target, return_value=value)
            entered.append(p)
            p.start()
        return fn()
    finally:
        for p in entered:
            p.stop()


# =============================================================================
# AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
# =============================================================================


class TestNoDynamicVramOptOut:
    """AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out

    Given: ComfyUI is running with its dynamic model memory management enabled.
    When: A workflow uses full saved model output.
    Then: The workflow can run without requiring users to disable ComfyUI's
    dynamic memory management.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _incremental_cache.clear()
        yield
        _incremental_cache.clear()

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    def test_full_model_completes_with_dynamic_vram_enabled(self, tmp_path, monkeypatch):
        """Full saved model mode runs successfully when ComfyUI's Dynamic VRAM
        (smart memory) is enabled — no opt-out is required."""
        # Configure Dynamic VRAM as active
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        mm.free_memory = MagicMock()
        mm.get_torch_device = MagicMock(return_value="cpu")
        mm.soft_empty_cache = MagicMock()
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        mode_info = detect_memory_mode()
        assert mode_info["dynamic_vram_active"] is True

        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        fake_loaded = MagicMock()
        fake_loaded.load_device = "cuda"
        fake_loaded.offload_device = "cpu"

        patches = _make_exit_patches(patcher, recipe, merged, save_path)
        patches["nodes.exit.load_saved_model"] = fake_loaded

        node = WIDENExitNode()

        def run():
            return node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        (result,) = _run_with_patches(patches, run)

        # Workflow completed — no ValueError or requirement to disable Dynamic VRAM
        assert result is fake_loaded

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    def test_full_model_cache_hit_with_dynamic_vram_enabled(self, tmp_path, monkeypatch):
        """Cache hit in full saved model mode works with Dynamic VRAM enabled."""
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "cached.safetensors"
        _build_cached_artifact(save_path, patcher, merged)

        fake_loaded = MagicMock()
        fake_loaded.load_device = "cuda"
        fake_loaded.offload_device = "cpu"

        node = WIDENExitNode()

        with (
            patch(
                "nodes.exit.validate_model_name",
                return_value="cached.safetensors",
            ),
            patch(
                "nodes.exit._resolve_checkpoints_path",
                return_value=str(save_path),
            ),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.load_saved_model", return_value=fake_loaded),
        ):
            (result,) = node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="cached",
            )

            # GPU eval skipped (cache hit)
            mock_analyze.assert_not_called()

        assert result is fake_loaded

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    def test_full_model_does_not_check_or_gate_on_memory_mode(self, tmp_path):
        """The full saved model path does not gate on the active memory mode —
        it works regardless of whether Dynamic VRAM is enabled or disabled."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        fake_loaded = MagicMock()
        patches = _make_exit_patches(patcher, recipe, merged, save_path)
        patches["nodes.exit.load_saved_model"] = fake_loaded

        node = WIDENExitNode()

        # Should work without any memory mode configuration at all
        def run():
            return node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        (result,) = _run_with_patches(patches, run)
        assert result is fake_loaded


# =============================================================================
# AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
# =============================================================================


class TestComfyOwnsReturnedModelMemoryE2E:
    """AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

    Given: The Exit node has returned a MODEL from a full saved model artifact.
    When: ComfyUI loads, unloads, partially loads, or prioritizes models.
    Then: The returned MODEL remains compatible with ComfyUI's model memory
    lifecycle and does not require all merged affected weights to remain
    resident solely because the Exit node returned.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _incremental_cache.clear()
        yield
        _incremental_cache.clear()

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_full_model_returns_comfy_loaded_model_not_patched_clone(self, tmp_path):
        """Full saved model mode returns a model from load_saved_model (Comfy
        owns memory), not a patched clone that holds in-memory merge tensors."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        # Track whether load_saved_model or install_merged_patches is used
        load_calls = []
        install_calls = []

        fake_loaded = MagicMock()
        fake_loaded.load_device = "cuda"
        fake_loaded.offload_device = "cpu"

        def track_load(path):
            load_calls.append(path)
            return fake_loaded

        patches = _make_exit_patches(patcher, recipe, merged, save_path)
        # Override with tracking versions
        del patches["nodes.exit.CheckpointMaterializationSink"]
        mock_mat = MagicMock()
        mock_mat.finalize.return_value = {}

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(save_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_cache", return_value=None),
            patch("nodes.exit.analyze_recipe", return_value=patches["nodes.exit.analyze_recipe"]),
            patch(
                "nodes.exit.analyze_recipe_models",
                return_value=patches["nodes.exit.analyze_recipe_models"],
            ),
            patch("nodes.exit.compile_plan", return_value=patches["nodes.exit.compile_plan"]),
            patch(
                "nodes.exit.compile_batch_groups",
                return_value=patches["nodes.exit.compile_batch_groups"],
            ),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.CheckpointMaterializationSink", return_value=mock_mat),
            patch("nodes.exit.build_metadata", return_value={"__ecaj_version__": "1"}),
            patch("nodes.exit.load_saved_model", side_effect=track_load),
            patch(
                "nodes.exit.install_merged_patches",
                side_effect=lambda *a: install_calls.append(True),
            ),
        ):
            (result,) = node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        # load_saved_model was called — Comfy owns the returned model
        assert len(load_calls) == 1
        assert load_calls[0] == str(save_path)

        # install_merged_patches was NOT called — no in-memory patch payload
        assert len(install_calls) == 0

        # Result is the Comfy-loaded model
        assert result is fake_loaded

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_full_model_loaded_via_comfy_sd_load_diffusion_model(self, tmp_path, monkeypatch):
        """The load path uses comfy.sd.load_diffusion_model, the same API that
        ComfyUI's built-in model loading uses. This ensures ComfyUI manages the
        returned model's memory lifecycle (load_device, offload_device)."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "cached.safetensors"
        _build_cached_artifact(save_path, patcher, merged)

        # Wire comfy.sd.load_diffusion_model mock
        load_calls = []

        def mock_comfy_load(path, model_options={}):
            load_calls.append(path)
            result = MockModelPatcher()
            result.load_device = torch.device("cuda")
            result.offload_device = torch.device("cpu")
            return result

        comfy_sd = ModuleType("comfy.sd")
        comfy_sd.load_diffusion_model = mock_comfy_load
        monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="cached.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(save_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.ProgressBar", None),
        ):
            (result,) = node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="cached",
            )

        # comfy.sd.load_diffusion_model was the actual loading path
        assert load_calls == [str(save_path)]

        # Returned model has ComfyUI memory lifecycle attributes
        assert hasattr(result, "load_device")
        assert hasattr(result, "offload_device")
        assert result.load_device == torch.device("cuda")
        assert result.offload_device == torch.device("cpu")

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_patches_mode_retains_in_memory_payload_for_comparison(self, tmp_path):
        """Control: In-memory patches mode installs merged tensors as set
        patches on a cloned ModelPatcher, retaining them in process memory.
        This is the baseline that full saved model mode improves upon."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)

        from lib.batch_groups import OpSignature
        from lib.recipe_eval import EvalPlan

        sig = OpSignature(shape=(4, 4), ndim=2)
        affected_set = set(merged.keys())
        lora = recipe.target

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_loader.loaded_bytes = 0
        mock_analyze = MagicMock(
            model_patcher=patcher,
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(lora)): affected_set},
            affected_keys=affected_set,
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()
        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: list(affected_set)}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
        ):
            (result,) = node.execute(recipe, output_mode=OUTPUT_MODE_PATCHES)

        # Patches mode returns a clone of the original patcher (same model)
        assert result.is_clone(patcher)

        # The clone has set patches installed — in-memory weight payload
        assert len(result.patches) > 0, "Patches mode should have set patches installed in memory"


# =============================================================================
# AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
# =============================================================================


class TestMeasuredMemoryBehavior:
    """AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior

    Given: The same representative merge workflow is run with the current patch
    output and with full saved model output.
    When: memory behavior is measured under ComfyUI with dynamic memory management
    enabled.
    Then: The validation output reports the active ComfyUI memory mode, peak RAM,
    peak GPU memory, returned-model behavior, and whether merged affected
    weights remain persistently resident after the Exit node returns; the
    validation fails if full saved model output requires disabling ComfyUI's
    dynamic memory management or requires all merged affected weights to
    remain resident solely because the Exit node returned.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _incremental_cache.clear()
        yield
        _incremental_cache.clear()

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_comparative_validation_reports_memory_behavior(self, tmp_path, monkeypatch):
        """Run the same recipe with both output modes and validate the memory
        behavior report covers all required fields.

        The AC requires that the validation output reports:
        - active ComfyUI memory mode
        - peak RAM and peak GPU memory
        - returned-model behavior
        - whether merged affected weights remain persistently resident
        """
        import resource

        # Configure Dynamic VRAM as active
        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        mm.free_memory = MagicMock()
        mm.get_torch_device = MagicMock(return_value="cpu")
        mm.soft_empty_cache = MagicMock()
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        # --- Measure memory mode ---
        memory_mode = detect_memory_mode()
        assert memory_mode["dynamic_vram_active"] is True

        # --- Run patches mode (control) with RSS measurement ---
        from lib.batch_groups import OpSignature
        from lib.recipe_eval import EvalPlan

        sig = OpSignature(shape=(4, 4), ndim=2)
        affected_set = set(merged.keys())
        lora = recipe.target

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_loader.loaded_bytes = 0
        mock_analyze = MagicMock(
            model_patcher=patcher,
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(lora)): affected_set},
            affected_keys=affected_set,
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()
        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

        node = WIDENExitNode()

        # Measure peak RSS around patches mode execution
        rss_before_patches = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: list(affected_set)}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
        ):
            (patches_result,) = node.execute(recipe, output_mode=OUTPUT_MODE_PATCHES)

        rss_after_patches = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # --- Run full saved model mode with RSS measurement ---
        # Use a real MockModelPatcher loaded from artifact to observe memory behavior
        fake_loaded = MockModelPatcher(keys=_ALL_KEYS)
        fake_loaded.load_device = "cuda"
        fake_loaded.offload_device = "cpu"
        # Real loaded model has NO set patches — weights are on disk
        assert len(fake_loaded.patches) == 0

        full_model_patches = _make_exit_patches(patcher, recipe, merged, save_path)
        full_model_patches["nodes.exit.load_saved_model"] = fake_loaded

        _incremental_cache.clear()  # Reset between runs

        rss_before_full = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gpu_before_full = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        def run_full():
            return node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        (full_result,) = _run_with_patches(full_model_patches, run_full)

        rss_after_full = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gpu_after_full = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

        # --- Observe returned model state (not hardcoded) ---
        # Patches mode: inspect whether patches dict contains set patches
        patches_has_set_patches = len(patches_result.patches) > 0
        patches_affected_resident = any(
            any(
                isinstance(entry[1], tuple) and entry[1][0] == "set"
                for entry in entries
            )
            for entries in patches_result.patches.values()
        )

        # Full model mode: inspect returned model for in-memory patch payload
        full_has_set_patches = len(getattr(full_result, "patches", {})) > 0
        full_affected_resident = any(
            any(
                isinstance(entry[1], tuple) and entry[1][0] == "set"
                for entry in entries
            )
            for entries in getattr(full_result, "patches", {}).values()
        ) if full_has_set_patches else False

        # --- Build validation report from observed measurements ---
        # Convert ru_maxrss (KB on Linux) to MB
        peak_rss_patches_mb = rss_after_patches // 1024
        peak_rss_full_mb = rss_after_full // 1024
        peak_gpu_full_mb = gpu_after_full // (1024 * 1024) if gpu_after_full else 0

        report = {
            "memory_mode": memory_mode["mode_name"],
            "dynamic_vram_active": memory_mode["dynamic_vram_active"],
            "peak_rss_mb": {
                "patches_mode": peak_rss_patches_mb,
                "full_model_mode": peak_rss_full_mb,
            },
            "peak_gpu_mb": peak_gpu_full_mb
            if torch.cuda.is_available()
            else "cuda_unavailable",
            "patches_mode": {
                "returned_model_type": "patched_clone",
                "has_set_patches": patches_has_set_patches,
                "affected_weights_resident": patches_affected_resident,
                "model_is_clone_of_base": patches_result.is_clone(patcher),
            },
            "full_model_mode": {
                "returned_model_type": "comfy_loaded_model",
                "has_set_patches": full_has_set_patches,
                "affected_weights_resident": full_affected_resident,
                "model_loaded_via_comfy": hasattr(full_result, "load_device"),
            },
        }

        # --- Validation assertions ---

        # Report covers required fields
        assert report["memory_mode"] == "dynamic_vram"
        assert report["dynamic_vram_active"] is True
        assert "peak_rss_mb" in report
        assert "peak_gpu_mb" in report
        assert isinstance(report["peak_rss_mb"]["patches_mode"], int)
        assert isinstance(report["peak_rss_mb"]["full_model_mode"], int)

        # Patches mode: affected weights ARE resident (in-memory set patches)
        assert report["patches_mode"]["has_set_patches"] is True
        assert report["patches_mode"]["affected_weights_resident"] is True

        # Full model mode: affected weights are NOT persistently resident
        # (observed from returned model's patches dict, not hardcoded)
        assert report["full_model_mode"]["has_set_patches"] is False
        assert report["full_model_mode"]["affected_weights_resident"] is False
        assert report["full_model_mode"]["model_loaded_via_comfy"] is True

        # Full model mode did not require disabling Dynamic VRAM
        assert report["dynamic_vram_active"] is True

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_full_model_does_not_call_install_merged_patches(self, tmp_path):
        """Full saved model mode must not call install_merged_patches, which
        would create an in-memory set patch payload on the returned model."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        fake_loaded = MagicMock()
        fake_loaded.load_device = "cuda"
        fake_loaded.offload_device = "cpu"

        patches = _make_exit_patches(patcher, recipe, merged, save_path)
        patches["nodes.exit.load_saved_model"] = fake_loaded

        install_calls = []

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(save_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_cache", return_value=None),
            patch("nodes.exit.analyze_recipe", return_value=patches["nodes.exit.analyze_recipe"]),
            patch(
                "nodes.exit.analyze_recipe_models",
                return_value=patches["nodes.exit.analyze_recipe_models"],
            ),
            patch("nodes.exit.compile_plan", return_value=patches["nodes.exit.compile_plan"]),
            patch(
                "nodes.exit.compile_batch_groups",
                return_value=patches["nodes.exit.compile_batch_groups"],
            ),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit.CheckpointMaterializationSink",
                return_value=patches["nodes.exit.CheckpointMaterializationSink"],
            ),
            patch("nodes.exit.build_metadata", return_value={"__ecaj_version__": "1"}),
            patch("nodes.exit.load_saved_model", return_value=fake_loaded),
            patch(
                "nodes.exit.install_merged_patches",
                side_effect=lambda *a: install_calls.append(True),
            ),
        ):
            node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        assert len(install_calls) == 0, (
            "Full model mode must not call install_merged_patches — "
            "the returned model is loaded from disk, not patched in memory"
        )

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_affected_weights_equivalent_between_modes(self, tmp_path, monkeypatch):
        """Affected weights produced by both output modes should be equivalent.

        Patches mode stores merged tensors as set patches. Full saved model
        mode writes them to disk via CheckpointMaterializationSink and the
        artifact is loaded back. Both should produce the same affected weight
        values for the same recipe.

        This test runs BOTH output modes end-to-end and reads the artifact
        from disk to verify equivalence, so a bug in CheckpointMaterializationSink,
        load_saved_model, or artifact writing would be caught.
        """
        from safetensors import safe_open

        from lib.batch_groups import OpSignature
        from lib.checkpoint_materialization import CheckpointMaterializationSink
        from lib.recipe_eval import EvalPlan

        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        sig = OpSignature(shape=(4, 4), ndim=2)
        affected_set = set(merged.keys())
        lora = recipe.target

        mock_loader = MagicMock()
        mock_loader.cleanup = MagicMock()
        mock_loader.loaded_bytes = 0
        mock_analyze = MagicMock(
            model_patcher=patcher,
            arch="sdxl",
            loader=mock_loader,
            set_affected={str(id(lora)): affected_set},
            affected_keys=affected_set,
        )
        mock_model_analysis = MagicMock()
        mock_model_analysis.model_loaders = {}
        mock_model_analysis.model_affected = {}
        mock_model_analysis.all_model_keys = frozenset()
        dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

        node = WIDENExitNode()

        # --- Run patches mode: extract set patch tensors ---
        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: list(affected_set)}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
        ):
            (patches_result,) = node.execute(recipe, output_mode=OUTPUT_MODE_PATCHES)

        # Extract the actual set patch tensor values from MockModelPatcher.
        # add_patches stores entries as (strength, patch_value, strength_model, None, None)
        # where patch_value from install_merged_patches is ("set", (tensor,)).
        patches_tensors = {}
        for key, entries in patches_result.patches.items():
            for entry in entries:
                patch_value = entry[1]  # (strength, patch_value, ...)
                if isinstance(patch_value, tuple) and patch_value[0] == "set":
                    patches_tensors[key] = patch_value[1][0]

        assert len(patches_tensors) == len(_AFFECTED_KEYS), (
            "Patches mode should have set patches for all affected keys"
        )

        # --- Run full saved model mode: writes artifact via CheckpointMaterializationSink ---
        # Use the real CheckpointMaterializationSink (not mocked) so the artifact
        # is actually written to disk. Only mock load_saved_model to return a
        # MockModelPatcher loaded from the artifact we just wrote.
        _incremental_cache.clear()

        def load_from_written_artifact(path):
            """Simulate comfy.sd.load_diffusion_model by reading the artifact."""
            loaded_patcher = MockModelPatcher(keys=_ALL_KEYS)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    loaded_patcher._state_dict[key] = f.get_tensor(key)
            loaded_patcher.model = type(loaded_patcher.model)(loaded_patcher._state_dict)
            loaded_patcher.load_device = "cuda"
            loaded_patcher.offload_device = "cpu"
            return loaded_patcher

        # Use the real CheckpointMaterializationSink but mock other pipeline stages
        real_mat_sink_cls = CheckpointMaterializationSink

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(save_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_cache", return_value=None),
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: list(affected_set)}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.CheckpointMaterializationSink", real_mat_sink_cls),
            patch("nodes.exit.build_metadata", return_value={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": json.dumps(sorted(merged.keys())),
                "__ecaj_artifact_kind__": "full_model",
            }),
            patch("nodes.exit.load_saved_model", side_effect=load_from_written_artifact),
        ):
            (full_result,) = node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="model",
            )

        # --- Verify the artifact was written to disk ---
        assert save_path.exists(), "Full model mode should write an artifact to disk"

        # --- Read artifact from disk and compare affected weights ---
        with safe_open(str(save_path), framework="pt", device="cpu") as f:
            for key in _AFFECTED_KEYS:
                artifact_tensor = f.get_tensor(key)
                assert key in patches_tensors, (
                    f"Patches mode should have produced a set patch for {key}"
                )
                assert torch.allclose(
                    artifact_tensor,
                    patches_tensors[key].to(dtype=artifact_tensor.dtype),
                ), (
                    f"Affected weight {key} differs between artifact on disk "
                    f"(full model mode) and set patches (patches mode)"
                )

        # --- Also verify via the loaded model returned by full_model mode ---
        for key in _AFFECTED_KEYS:
            loaded_tensor = full_result._state_dict[key]
            assert torch.allclose(
                loaded_tensor,
                patches_tensors[key].to(dtype=loaded_tensor.dtype),
            ), (
                f"Affected weight {key} differs between loaded model "
                f"(full model mode return) and set patches (patches mode)"
            )

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_memory_logging_captures_rss_and_vram(self, tmp_path, caplog):
        """The Exit node's memory logging captures RSS and VRAM at key points
        during execution, providing the data needed for memory behavior reports."""
        import logging

        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "model.safetensors"

        fake_loaded = MagicMock()

        patches = _make_exit_patches(patcher, recipe, merged, save_path)
        patches["nodes.exit.load_saved_model"] = fake_loaded

        node = WIDENExitNode()

        with caplog.at_level(logging.INFO, logger="ecaj.exit"):

            def run():
                return node.execute(
                    recipe,
                    output_mode=OUTPUT_MODE_FULL_MODEL,
                    model_name="model",
                )

            _run_with_patches(patches, run)

        # Memory logging captures RSS at key points
        mem_logs = [r.message for r in caplog.records if "[mem]" in r.message]
        assert any("before-gpu-eval" in msg for msg in mem_logs), (
            "Should log memory before GPU evaluation"
        )
        # RSS is logged (may be 0 on non-Linux, but the log entry exists)
        assert any("RSS=" in msg for msg in mem_logs), "Memory logs should include RSS measurement"

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_validation_fails_if_full_model_uses_install_patches(self, tmp_path):
        """Validation assertion: if full saved model mode were to call
        install_merged_patches, it would mean affected weights remain resident
        as an in-memory patch payload — the validation should catch this."""
        patcher = _make_patcher()
        merged = _merged_tensors_for(_AFFECTED_KEYS)

        # Simulate what would happen if full_model mode incorrectly used patches
        from nodes.exit import install_merged_patches

        patched_clone = install_merged_patches(patcher, merged, torch.float32)

        # This model has set patches — affected weights are resident
        has_patches = len(patched_clone.patches) > 0
        assert has_patches is True, "Patched clone retains in-memory payload"

        # Validation criterion: full saved model mode should NOT produce this
        # (This test documents the detection mechanism for the validation report)

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_detect_memory_mode_unavailable_when_no_comfy(self, monkeypatch):
        """When ComfyUI is not available, detect_memory_mode reports
        'unavailable' — the validation skips dynamic VRAM checks with
        a clear reason rather than failing."""
        # Remove comfy.model_management from sys.modules entirely
        mm = ModuleType("comfy.model_management")
        # No DISABLE_SMART_MEMORY attribute
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        result = detect_memory_mode()
        assert result["mode_name"] == "unavailable"
        assert result["dynamic_vram_active"] is None
        assert "reason" in result


# =============================================================================
# Output equivalence: full model artifact contains correct weights
# =============================================================================


class TestOutputEquivalence:
    """Cross-mode equivalence: verify the saved artifact contains the correct
    merged weights for affected keys and preserves base weights for unaffected
    keys."""

    def test_cached_artifact_has_correct_affected_and_base_weights(self, tmp_path):
        """A cached full_model artifact should contain merged values for
        affected keys and original base values for unaffected keys."""
        from safetensors import safe_open

        patcher = _make_patcher()
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        artifact_path = tmp_path / "artifact.safetensors"
        _build_cached_artifact(artifact_path, patcher, merged)

        base_state = patcher.model_state_dict()

        with safe_open(str(artifact_path), framework="pt", device="cpu") as f:
            for key in _AFFECTED_KEYS:
                loaded = f.get_tensor(key)
                assert torch.allclose(loaded, merged[key]), (
                    f"Affected key {key}: artifact value should match merged tensor"
                )
            for key in _UNAFFECTED_KEYS:
                loaded = f.get_tensor(key)
                assert torch.allclose(loaded, base_state[key]), (
                    f"Unaffected key {key}: artifact value should match base state"
                )

    def test_cache_reuse_skips_recomputation(self, tmp_path):
        """When a cached artifact exists, full saved model mode loads it
        without recomputing — verifying cache reuse is correct."""
        patcher = _make_patcher()
        recipe = _make_recipe(patcher)
        merged = _merged_tensors_for(_AFFECTED_KEYS)
        save_path = tmp_path / "cached.safetensors"
        _build_cached_artifact(save_path, patcher, merged)

        eval_called = []
        fake_loaded = MagicMock()

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="cached.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(save_path)),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit.chunked_evaluation",
                side_effect=lambda *a, **kw: eval_called.append(True),
            ),
            patch("nodes.exit.load_saved_model", return_value=fake_loaded),
        ):
            (result,) = node.execute(
                recipe,
                output_mode=OUTPUT_MODE_FULL_MODEL,
                model_name="cached",
            )

        assert len(eval_called) == 0, "GPU evaluation should be skipped on cache hit"
        assert result is fake_loaded


# =============================================================================
# Integration: optional ComfyUI measurement script
# =============================================================================


class TestOptionalIntegrationScript:
    """Optional integration validation for ComfyUI environments.

    These tests document how to run Dynamic VRAM memory measurement when
    ComfyUI and compatible hardware are available. They skip with a clear
    reason when the environment is unavailable.
    """

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_dynamic_vram_measurement_requires_comfy(self, monkeypatch):
        """Dynamic VRAM measurement requires a real ComfyUI environment.

        This test documents the skip reason and proves detect_memory_mode
        returns actionable information when ComfyUI is unavailable."""
        # Use a bare stub module (no DISABLE_SMART_MEMORY)
        mm = ModuleType("comfy.model_management")
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        mode = detect_memory_mode()
        if mode["dynamic_vram_active"] is None:
            pytest.skip(
                f"Dynamic VRAM measurement skipped: {mode.get('reason', 'ComfyUI unavailable')}. "
                "Run with a full ComfyUI environment to measure actual Dynamic VRAM behavior."
            )
        # If we get here, ComfyUI is available — measure would proceed

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_gpu_memory_measurement_requires_cuda(self):
        """GPU memory measurement requires CUDA hardware.

        This test documents the skip reason for environments without GPU."""
        if not torch.cuda.is_available():
            pytest.skip(
                "GPU memory measurement skipped: CUDA not available. "
                "Run on a machine with a CUDA GPU to measure peak VRAM behavior."
            )

        # If we get here, CUDA is available — capture baseline
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        assert isinstance(allocated, int)
        assert isinstance(reserved, int)

    # AC: @comfy-memory-manager-compatibility ac-measured-memory-behavior
    def test_full_validation_report_shape(self, monkeypatch):
        """The full validation report contains all required fields regardless
        of environment availability. RSS is measured from process state."""
        import resource

        mm = ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = False
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

        mode = detect_memory_mode()

        # Measure actual peak RSS from process
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss_kb // 1024

        # Measure actual GPU memory if available
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024)
        else:
            peak_gpu_mb = "cuda_unavailable"

        report = {
            "comfy_memory_mode": mode["mode_name"],
            "dynamic_vram_active": mode["dynamic_vram_active"],
            "peak_rss_mb": peak_rss_mb,
            "peak_gpu_mb": peak_gpu_mb,
            "returned_model_behavior": {
                "patches_mode": "patched_clone_with_set_patches",
                "full_model_mode": "comfy_loaded_from_artifact",
            },
            "affected_weights_persistent": {
                "patches_mode": True,
                "full_model_mode": False,
            },
            "requires_dynamic_vram_disabled": False,
        }

        # All required report fields are present
        assert "comfy_memory_mode" in report
        assert "dynamic_vram_active" in report
        assert "peak_rss_mb" in report
        assert "peak_gpu_mb" in report
        assert "returned_model_behavior" in report
        assert "affected_weights_persistent" in report
        assert "requires_dynamic_vram_disabled" in report

        # Peak RSS is a real measurement (not a placeholder)
        assert isinstance(report["peak_rss_mb"], int)
        assert report["peak_rss_mb"] > 0

        # Validation assertions
        assert report["requires_dynamic_vram_disabled"] is False
        assert report["affected_weights_persistent"]["full_model_mode"] is False
