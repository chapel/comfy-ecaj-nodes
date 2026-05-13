"""Tests for checkpoint save_model return paths without deep-copying Comfy internals.

AC coverage for:
- @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
- @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
- @saved-model-artifact-safety ac-missing-metadata-not-reused
- @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
- @saved-model-artifact-safety ac-failed-return-not-successful
- @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
- @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
- @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload

Task: @task-checkpoint-save-returned-model,
      @task-checkpoint-save-temp-payload-release-hardening
Spec: @checkpoint-loadable-saved-model-output,
      @comfy-memory-manager-compatibility,
      @saved-model-artifact-safety
"""

from __future__ import annotations

import contextlib
import copy
from unittest.mock import MagicMock, patch

import pytest

from lib.recipe import (
    RecipeBase,
    RecipeLoRA,
    RecipeMerge,
)
from nodes.exit import WIDENExitNode
from tests.conftest import make_checkpoint_components


class TestTemporaryCheckpointModelRelease:
    """Release of checkpoint-save temporaries must also return freed native
    CPU arenas to the OS, not only drop Python references.

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    """

    def test_release_trims_native_heap_after_python_gc(self):
        """The live Comfy failure mode leaves large freed anonymous CPU arenas
        swapped inside the process; cleanup should invoke native heap trim after
        unpatching and Python GC."""
        from nodes.exit import _release_temporary_checkpoint_model

        temporary_model = object()
        events: list[str] = []

        with (
            patch(
                "nodes.exit._unpatch_loaded_clones",
                side_effect=lambda model: events.append("unpatch"),
            ),
            patch("nodes.exit.gc.collect", side_effect=lambda: events.append("gc")),
            patch("nodes.exit.torch.cuda.is_available", return_value=False),
            patch(
                "nodes.exit._trim_native_heap",
                create=True,
                side_effect=lambda: events.append("trim"),
            ),
        ):
            _release_temporary_checkpoint_model(temporary_model)

        assert events == ["unpatch", "gc", "trim"]

    def test_release_clears_temp_model_patch_payloads_before_gc(self):
        """Because the caller still has a local reference while invoking the
        release helper, the helper must sever patch-tensor references itself
        before GC/trim can reclaim native CPU pages."""
        from nodes.exit import _release_temporary_checkpoint_model

        class TempModel:
            def __init__(self):
                self.patches = {
                    "diffusion_model.k": [
                        (1.0, ("set", (object(),)), 1.0, None, None),
                    ],
                }

        temporary_model = TempModel()

        with (
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.gc.collect"),
            patch("nodes.exit.torch.cuda.is_available", return_value=False),
            patch("nodes.exit._trim_native_heap", create=True),
        ):
            _release_temporary_checkpoint_model(temporary_model)

        assert temporary_model.patches == {}


# =============================================================================
# No deepcopy on checkpoint-style save_model return paths
# =============================================================================


class TestNoDeepCopyOnCheckpointReturn:
    """No checkpoint-style save_model path may call copy.deepcopy on
    cloned.model, ModelPatcher.model, diffusion_model, CLIP, or VAE internals.

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    """

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_cache_miss_checkpoint_does_not_deepcopy(self, mock_model_patcher, tmp_path):
        """Cache-miss checkpoint save must not call copy.deepcopy anywhere."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        original_deepcopy = copy.deepcopy

        def deepcopy_trap(obj, memo=None):
            # Allow deepcopy of simple types (dicts, lists, etc.) but reject
            # any object that looks like a Comfy model internal.
            if hasattr(obj, "diffusion_model") or hasattr(obj, "load_state_dict"):
                raise AssertionError(
                    f"copy.deepcopy called on Comfy model internal: {type(obj).__name__}"
                )
            return original_deepcopy(obj, memo)

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._load_checkpoint_artifact", return_value=mock_model_patcher.clone()),
            patch("copy.deepcopy", side_effect=deepcopy_trap),
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )
            mock_install.return_value = mock_model_patcher.clone()

            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is not None

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_cache_hit_checkpoint_does_not_deepcopy(self, mock_model_patcher, tmp_path):
        """Cache-hit checkpoint load must not call copy.deepcopy anywhere."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        original_deepcopy = copy.deepcopy

        def deepcopy_trap(obj, memo=None):
            if hasattr(obj, "diffusion_model") or hasattr(obj, "load_state_dict"):
                raise AssertionError(
                    f"copy.deepcopy called on Comfy model internal: {type(obj).__name__}"
                )
            return original_deepcopy(obj, memo)

        mock_loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit._load_checkpoint_artifact",
                return_value=mock_loaded_model,
            ) as mock_ckpt_load,
            patch("copy.deepcopy", side_effect=deepcopy_trap),
        ):
            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is not None
            mock_ckpt_load.assert_called_once()

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_noop_cache_hit_checkpoint_does_not_deepcopy(self, mock_model_patcher, tmp_path):
        """No-op RecipeBase checkpoint cache hit must not call copy.deepcopy."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        original_deepcopy = copy.deepcopy

        def deepcopy_trap(obj, memo=None):
            if hasattr(obj, "diffusion_model") or hasattr(obj, "load_state_dict"):
                raise AssertionError(
                    f"copy.deepcopy called on Comfy model internal: {type(obj).__name__}"
                )
            return original_deepcopy(obj, memo)

        mock_loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._load_checkpoint_artifact", return_value=mock_loaded_model),
            patch("copy.deepcopy", side_effect=deepcopy_trap),
        ):
            result = node.execute(base, save_model=True, model_name="model")

            assert result[0] is not None


# =============================================================================
# Cache miss releases temporary merged MODEL and reloads saved checkpoint
# =============================================================================


class TestCacheMissReloadsSavedCheckpoint:
    """Cache-miss checkpoint save returns a Comfy-loaded artifact MODEL, not
    the temporary dense ModelPatcher used only for checkpoint serialization.

    AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    """

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    # AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    def test_cache_miss_returns_loaded_checkpoint_artifact_not_temp_model(
        self, mock_model_patcher, tmp_path,
    ):
        """After a fresh checkpoint save, the returned MODEL must come from
        _load_checkpoint_artifact so Comfy owns the returned model lifecycle."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        temp_model = mock_model_patcher.clone()
        loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch(
                "nodes.exit.install_merged_patches",
                return_value=temp_model,
            ) as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._release_temporary_checkpoint_model") as mock_release,
            patch(
                "nodes.exit._load_checkpoint_artifact",
                return_value=loaded_model,
            ) as mock_ckpt_load,
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )

            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is loaded_model
            assert result[0] is not temp_model
            mock_install.assert_called_once()
            mock_release.assert_called_once_with(temp_model)
            mock_ckpt_load.assert_called_once_with(save_path)

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    # AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    def test_cache_miss_releases_temp_model_before_loading_checkpoint_artifact(
        self, mock_model_patcher, tmp_path,
    ):
        """The temporary merged ModelPatcher is released after checkpoint save
        and before the artifact-backed MODEL is loaded for the return value."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()
        temp_model = mock_model_patcher.clone()
        loaded_model = mock_model_patcher.clone()
        events: list[str] = []

        def save_checkpoint(*args, **kwargs):
            events.append("save")

        def release_temp(model):
            assert model is temp_model
            events.append("release")

        def load_checkpoint(path):
            assert path == save_path
            events.append("load")
            return loaded_model

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=temp_model),
            patch("nodes.exit.save_comfy_checkpoint", side_effect=save_checkpoint),
            patch("nodes.exit.check_ram_preflight"),
            patch(
                "nodes.exit._release_temporary_checkpoint_model",
                side_effect=release_temp,
            ),
            patch("nodes.exit._load_checkpoint_artifact", side_effect=load_checkpoint),
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )

            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is loaded_model
            assert events == ["save", "release", "load"]


# =============================================================================
# Cache hit invokes Comfy checkpoint load path
# =============================================================================


class TestCacheHitUsesComfyCheckpointLoader:
    """Cache-hit checkpoint load must invoke Comfy's supported checkpoint load
    path and skip analyze_recipe/GPU merge work.

    AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    """

    # AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    def test_cache_hit_calls_load_checkpoint_artifact(self, mock_model_patcher, tmp_path):
        """Checkpoint cache hit must call _load_checkpoint_artifact, not
        _load_model_from_artifact."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        mock_loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit._load_checkpoint_artifact",
                return_value=mock_loaded_model,
            ) as mock_ckpt_load,
            patch("nodes.exit._load_model_from_artifact") as mock_diffusion_load,
        ):
            result = node.execute(merge, save_model=True, model_name="model")

            mock_ckpt_load.assert_called_once_with(save_path)
            mock_diffusion_load.assert_not_called()
            assert result[0] is mock_loaded_model

    # AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    def test_cache_hit_skips_analyze_recipe_and_gpu_merge(self, mock_model_patcher, tmp_path):
        """Checkpoint cache hit must skip recipe analysis and GPU merge."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._load_checkpoint_artifact", return_value=mock_model_patcher.clone()),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.chunked_evaluation") as mock_eval,
            patch("nodes.exit.install_merged_patches") as mock_install,
        ):
            node.execute(merge, save_model=True, model_name="model")

            mock_analyze.assert_not_called()
            mock_eval.assert_not_called()
            mock_install.assert_not_called()

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_noop_cache_hit_uses_checkpoint_loader(self, mock_model_patcher, tmp_path):
        """No-op RecipeBase checkpoint cache hit must use _load_checkpoint_artifact."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        mock_loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit._load_checkpoint_artifact",
                return_value=mock_loaded_model,
            ) as mock_ckpt_load,
            patch("nodes.exit._load_model_from_artifact") as mock_diffusion_load,
        ):
            result = node.execute(base, save_model=True, model_name="model")

            mock_ckpt_load.assert_called_once_with(save_path)
            mock_diffusion_load.assert_not_called()
            assert result[0] is mock_loaded_model


# =============================================================================
# Cache-hit load failure surfaces clear diagnostic
# =============================================================================


class TestCacheHitLoadFailureDiagnostic:
    """If the cache-hit Comfy load path fails, raise a clear error naming
    the saved checkpoint path.

    AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
    """

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_load_failure_raises_with_checkpoint_path(self, tmp_path):
        """_load_checkpoint_artifact must raise RuntimeError naming the
        checkpoint path when the Comfy loader fails."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "model.safetensors")

        with patch("comfy.sd.load_checkpoint_guess_config", side_effect=RuntimeError("corrupt")):
            with pytest.raises(RuntimeError, match=r"model\.safetensors"):
                _load_checkpoint_artifact(save_path)

    # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
    def test_load_failure_does_not_return_partial_model(self, tmp_path):
        """When load fails, no partial MODEL is returned — error propagates."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "nonexistent.safetensors")

        with patch("comfy.sd.load_checkpoint_guess_config", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="nonexistent"):
                _load_checkpoint_artifact(save_path)

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_load_failure_in_execute_propagates(self, mock_model_patcher, tmp_path):
        """When _load_checkpoint_artifact fails during cache-hit execute,
        the error propagates — the artifact is not treated as successful reuse."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch(
                "nodes.exit._load_checkpoint_artifact",
                side_effect=RuntimeError("Failed to load checkpoint: model.safetensors"),
            ),
        ):
            with pytest.raises(RuntimeError, match="model.safetensors"):
                node.execute(merge, save_model=True, model_name="model")


# =============================================================================
# Downstream MODEL consumers receive usable MODEL from both paths
# =============================================================================


class TestDownstreamModelUsability:
    """Connected downstream MODEL consumers receive a non-None usable MODEL
    from both cache-miss and cache-hit paths.

    AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    """

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_cache_miss_returns_non_none_model(self, mock_model_patcher, tmp_path):
        """Cache-miss checkpoint save must return a non-None MODEL."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=mock_model_patcher.clone()),
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._load_checkpoint_artifact", return_value=mock_model_patcher.clone()),
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )

            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is not None
            assert hasattr(result[0], "model_state_dict")

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_cache_hit_returns_non_none_model(self, mock_model_patcher, tmp_path):
        """Cache-hit checkpoint load must return a non-None MODEL."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        mock_loaded_model = mock_model_patcher.clone()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._load_checkpoint_artifact", return_value=mock_loaded_model),
        ):
            result = node.execute(merge, save_model=True, model_name="model")

            assert result[0] is not None
            assert result[0] is mock_loaded_model
            assert hasattr(result[0], "model_state_dict")


# =============================================================================
# _load_checkpoint_artifact unit tests
# =============================================================================


class TestLoadCheckpointArtifact:
    """Unit tests for _load_checkpoint_artifact: the function that wraps
    comfy.sd.load_checkpoint_guess_config for cache-hit checkpoint loading.

    AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    """

    # AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    def test_calls_load_checkpoint_guess_config(self, tmp_path):
        """Must call comfy.sd.load_checkpoint_guess_config with the correct path."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "model.safetensors")
        mock_model = MagicMock(name="loaded_model")

        load_result = [mock_model, None, None]
        with patch(
            "comfy.sd.load_checkpoint_guess_config",
            return_value=load_result,
        ) as mock_load:
            result = _load_checkpoint_artifact(save_path)

            mock_load.assert_called_once_with(
                save_path, output_vae=True, output_clip=True,
            )
            assert result is mock_model

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_returns_only_model_component(self, tmp_path):
        """Must return only the MODEL, not CLIP or VAE from the loaded tuple."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "model.safetensors")
        mock_model = MagicMock(name="model")
        mock_clip = MagicMock(name="clip")
        mock_vae = MagicMock(name="vae")

        load_result = [mock_model, mock_clip, mock_vae]
        with patch(
            "comfy.sd.load_checkpoint_guess_config",
            return_value=load_result,
        ):
            result = _load_checkpoint_artifact(save_path)

            assert result is mock_model
            assert result is not mock_clip
            assert result is not mock_vae

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_returned_model_is_comfy_loaded(self, tmp_path):
        """The MODEL returned by _load_checkpoint_artifact is Comfy-loaded,
        not constructed by WIDEN. Comfy owns its memory lifecycle."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "model.safetensors")

        # Simulate Comfy returning a properly loaded ModelPatcher
        mock_model = MagicMock(name="comfy_loaded_model")
        mock_model.model = MagicMock(name="model_internals")

        load_result = [mock_model, None, None]
        with patch(
            "comfy.sd.load_checkpoint_guess_config",
            return_value=load_result,
        ):
            result = _load_checkpoint_artifact(save_path)

            # Must be the exact object Comfy returned, not a clone or copy
            assert result is mock_model

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_error_message_includes_path(self, tmp_path):
        """On loader failure, the error must include the checkpoint file path."""
        from nodes.exit import _load_checkpoint_artifact

        save_path = str(tmp_path / "my_checkpoint.safetensors")

        with patch("comfy.sd.load_checkpoint_guess_config", side_effect=Exception("bad file")):
            with pytest.raises(RuntimeError) as exc_info:
                _load_checkpoint_artifact(save_path)

            assert "my_checkpoint.safetensors" in str(exc_info.value)


# =============================================================================
# save_model=False patch-mode unchanged
# =============================================================================


class TestPatchModeUnchanged:
    """save_model=False patch-mode return behavior must remain unchanged.

    This test verifies that the changes do not affect the existing patch-mode
    path, which returns a ModelPatcher with merged weights as set patches.
    """

    def test_patch_mode_does_not_use_checkpoint_loader(self, mock_model_patcher, tmp_path):
        """save_model=False must not use _load_checkpoint_artifact."""
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=mock_model_patcher.clone()),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._load_checkpoint_artifact") as mock_ckpt_load,
            patch("nodes.exit._load_model_from_artifact") as mock_diffusion_load,
        ):
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )

            result = node.execute(merge, save_model=False)

            mock_ckpt_load.assert_not_called()
            mock_diffusion_load.assert_not_called()
            assert result[0] is not None


# =============================================================================
# Checkpoint save failure paths release the temporary payload
# =============================================================================


def _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher):
    mock_loader = MagicMock()
    mock_loader.cleanup = MagicMock()
    mock_loader.loaded_bytes = 0
    mock_analyze.return_value = MagicMock(
        model_patcher=mock_model_patcher,
        arch="sdxl",
        loader=mock_loader,
        set_affected={},
        affected_keys=set(),
    )
    mock_analyze_models.return_value = MagicMock(
        model_loaders={},
        model_affected={},
        all_model_keys=frozenset(),
    )


class TestCheckpointSaveFailureReleasesTempPayload:
    """When save_comfy_checkpoint, post-save bookkeeping, or _load_checkpoint_artifact
    raise during a cache-miss save, the temporary save-time merge payload must
    still be released before control returns to ComfyUI, and the original
    failure must propagate.

    AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
    AC: @saved-model-artifact-safety ac-failed-return-not-successful
    """

    # AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
    # AC: @saved-model-artifact-safety ac-failed-return-not-successful
    def test_save_comfy_checkpoint_failure_releases_temp_and_propagates(
        self, mock_model_patcher, tmp_path,
    ):
        """If save_comfy_checkpoint raises after the temp ModelPatcher exists,
        finally still releases the temp payload and the save error propagates."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()
        temp_model = mock_model_patcher.clone()
        release_calls: list[object] = []

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=temp_model),
            patch(
                "nodes.exit.save_comfy_checkpoint",
                side_effect=RuntimeError("save failed: disk full"),
            ),
            patch("nodes.exit.check_ram_preflight"),
            patch(
                "nodes.exit._release_temporary_checkpoint_model",
                side_effect=release_calls.append,
            ),
            patch("nodes.exit._load_checkpoint_artifact") as mock_ckpt_load,
        ):
            _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher)

            with pytest.raises(RuntimeError, match="save failed: disk full"):
                node.execute(merge, save_model=True, model_name="model")

        assert release_calls == [temp_model]
        mock_ckpt_load.assert_not_called()

    # AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
    # AC: @saved-model-artifact-safety ac-failed-return-not-successful
    def test_post_save_publication_failure_releases_temp_and_propagates(
        self, mock_model_patcher, tmp_path,
    ):
        """If post-save finalization (cache bookkeeping after the artifact is
        written) raises while the temp ModelPatcher is alive, finally still
        releases the temp payload and the error propagates."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()
        temp_model = mock_model_patcher.clone()
        release_calls: list[object] = []

        class ExplodingCache:
            def clear(self):
                raise RuntimeError("finalize failed: cache state")

        with contextlib.ExitStack() as stack:
            for cm in [
                patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
                patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
                patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
                patch("nodes.exit.compute_base_identity", return_value="base_id"),
                patch("nodes.exit.compute_lora_stats", return_value={}),
                patch("nodes.exit.serialize_recipe", return_value="{}"),
                patch("nodes.exit.validate_checkpoint_components"),
                patch("nodes.exit.check_checkpoint_cache", return_value=False),
                patch("nodes.exit._unpatch_loaded_clones"),
                patch("nodes.exit.ProgressBar", None),
                patch("nodes.exit.compile_plan", return_value=MagicMock()),
                patch("nodes.exit.chunked_evaluation", return_value={}),
                patch("nodes.exit.compile_batch_groups", return_value={}),
                patch("nodes.exit.install_merged_patches", return_value=temp_model),
                patch("nodes.exit.save_comfy_checkpoint"),
                patch("nodes.exit.check_ram_preflight"),
                patch("nodes.exit._incremental_cache", ExplodingCache()),
                patch(
                    "nodes.exit._release_temporary_checkpoint_model",
                    side_effect=release_calls.append,
                ),
            ]:
                stack.enter_context(cm)
            mock_analyze = stack.enter_context(patch("nodes.exit.analyze_recipe"))
            mock_analyze_models = stack.enter_context(
                patch("nodes.exit.analyze_recipe_models")
            )
            mock_ckpt_load = stack.enter_context(
                patch("nodes.exit._load_checkpoint_artifact")
            )
            _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher)

            with pytest.raises(RuntimeError, match="finalize failed: cache state"):
                node.execute(
                    merge,
                    save_model=True,
                    model_name="model",
                    enable_cache=False,
                )

        assert release_calls == [temp_model]
        mock_ckpt_load.assert_not_called()

    # AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
    # AC: @saved-model-artifact-safety ac-failed-return-not-successful
    def test_artifact_reload_failure_releases_temp_and_propagates(
        self, mock_model_patcher, tmp_path,
    ):
        """If _load_checkpoint_artifact raises after a successful save, the
        temporary save-time merge payload has already been released before the
        reload attempt and the reload error propagates."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()
        temp_model = mock_model_patcher.clone()
        events: list[str] = []

        def release_temp(model):
            assert model is temp_model
            events.append("release")

        def load_checkpoint(path):
            events.append("load-attempt")
            raise RuntimeError("reload failed: cannot read artifact")

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=temp_model),
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
            patch(
                "nodes.exit._release_temporary_checkpoint_model",
                side_effect=release_temp,
            ),
            patch("nodes.exit._load_checkpoint_artifact", side_effect=load_checkpoint),
        ):
            _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher)

            with pytest.raises(RuntimeError, match="reload failed"):
                node.execute(merge, save_model=True, model_name="model")

        assert events == ["release", "load-attempt"]

    # AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
    def test_failure_before_temp_model_created_skips_release(
        self, mock_model_patcher, tmp_path,
    ):
        """If install_merged_patches itself fails, no temporary payload was
        ever created, so _release_temporary_checkpoint_model must not be
        called. The install error propagates."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch(
                "nodes.exit.install_merged_patches",
                side_effect=RuntimeError("install failed: no model"),
            ),
            patch("nodes.exit.save_comfy_checkpoint") as mock_save,
            patch("nodes.exit.check_ram_preflight"),
            patch(
                "nodes.exit._release_temporary_checkpoint_model",
            ) as mock_release,
            patch("nodes.exit._load_checkpoint_artifact") as mock_ckpt_load,
        ):
            _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher)

            with pytest.raises(RuntimeError, match="install failed: no model"):
                node.execute(merge, save_model=True, model_name="model")

        mock_save.assert_not_called()
        mock_release.assert_not_called()
        mock_ckpt_load.assert_not_called()


# =============================================================================
# Cache-miss success order: save → release → load
# =============================================================================


class TestCheckpointCacheMissSuccessOrder:
    """In the cache-miss success path, the explicit order is: checkpoint save,
    temporary-model release, then artifact load for the returned MODEL.

    AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    """

    # AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    def test_cache_miss_success_order_save_release_load(
        self, mock_model_patcher, tmp_path,
    ):
        """In the cache-miss success path, save_comfy_checkpoint runs first,
        then the temporary save-time merge payload is released, then the
        artifact is loaded for the returned MODEL."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()
        temp_model = mock_model_patcher.clone()
        loaded_model = mock_model_patcher.clone()
        events: list[str] = []

        def record_save(*_args, **_kwargs):
            events.append("save")

        def record_release(model):
            assert model is temp_model
            events.append("release")

        def record_load(path):
            assert path == save_path
            events.append("load")
            return loaded_model

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=temp_model),
            patch("nodes.exit.save_comfy_checkpoint", side_effect=record_save),
            patch("nodes.exit.check_ram_preflight"),
            patch(
                "nodes.exit._release_temporary_checkpoint_model",
                side_effect=record_release,
            ),
            patch(
                "nodes.exit._load_checkpoint_artifact",
                side_effect=record_load,
            ),
        ):
            _stub_analyze(mock_analyze, mock_analyze_models, mock_model_patcher)

            result = node.execute(merge, save_model=True, model_name="model")

        assert events == ["save", "release", "load"]
        assert result[0] is loaded_model


# =============================================================================
# _clear_temporary_model_patch_payloads attribute coverage
# =============================================================================


class TestClearTemporaryModelPatchPayloads:
    """`_clear_temporary_model_patch_payloads` severs all known temporary
    patch-payload containers on the ModelPatcher clone before GC runs, and
    tolerates payload containers being absent.

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
    """

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_clears_patches_object_patches_and_weight_wrapper_patches(self):
        """All three known payload containers — patches, object_patches, and
        weight_wrapper_patches — are cleared when present."""
        from nodes.exit import _clear_temporary_model_patch_payloads

        class TempModel:
            def __init__(self):
                self.patches = {"diffusion_model.k": [(1.0, ("set", (object(),)))]}
                self.object_patches = {"sub.module": object()}
                self.weight_wrapper_patches = {"diffusion_model.k": [object()]}

        temporary_model = TempModel()
        _clear_temporary_model_patch_payloads(temporary_model)

        assert temporary_model.patches == {}
        assert temporary_model.object_patches == {}
        assert temporary_model.weight_wrapper_patches == {}

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_tolerates_missing_attributes(self):
        """Clearing must not require every container attribute to exist on the
        ModelPatcher clone; missing attributes are skipped silently."""
        from nodes.exit import _clear_temporary_model_patch_payloads

        class TempModelPartial:
            def __init__(self):
                self.patches = {"diffusion_model.k": [(1.0, ("set", (object(),)))]}

        temporary_model = TempModelPartial()
        _clear_temporary_model_patch_payloads(temporary_model)

        assert temporary_model.patches == {}
        assert not hasattr(temporary_model, "object_patches")
        assert not hasattr(temporary_model, "weight_wrapper_patches")

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_skips_attributes_without_clear_method(self):
        """Attributes that exist but do not support clear() are skipped — the
        helper must not raise when a container is unusable."""
        from nodes.exit import _clear_temporary_model_patch_payloads

        class TempModelWeird:
            def __init__(self):
                self.patches = None
                self.object_patches = 0
                self.weight_wrapper_patches = "not-a-dict"

        temporary_model = TempModelWeird()
        _clear_temporary_model_patch_payloads(temporary_model)

        assert temporary_model.patches is None
        assert temporary_model.object_patches == 0
        assert temporary_model.weight_wrapper_patches == "not-a-dict"


# =============================================================================
# _release_temporary_checkpoint_model order: clear → gc → trim
# =============================================================================


class TestReleaseTemporaryCheckpointModelOrder:
    """`_release_temporary_checkpoint_model` severs temporary patch payloads
    before invoking Python GC and native heap trim so freed pages are
    actually returnable to the OS allocator.

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    """

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_clear_runs_before_gc_and_trim(self):
        """Patch-payload containers must be cleared BEFORE gc.collect and the
        native heap trim run, otherwise the freed native pages cannot be
        returned to the OS in this release pass."""
        from nodes.exit import _release_temporary_checkpoint_model

        events: list[str] = []

        class TrackedContainer:
            def __init__(self, label):
                self._label = label
                self._cleared = False

            def clear(self):
                events.append(f"clear:{self._label}")
                self._cleared = True

        class TempModel:
            def __init__(self):
                self.patches = TrackedContainer("patches")
                self.object_patches = TrackedContainer("object_patches")
                self.weight_wrapper_patches = TrackedContainer("weight_wrapper_patches")

        temporary_model = TempModel()

        with (
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.gc.collect", side_effect=lambda: events.append("gc")),
            patch("nodes.exit.torch.cuda.is_available", return_value=False),
            patch(
                "nodes.exit._trim_native_heap",
                create=True,
                side_effect=lambda: events.append("trim"),
            ),
        ):
            _release_temporary_checkpoint_model(temporary_model)

        # All three containers were cleared before gc and trim ran.
        clear_events = [e for e in events if e.startswith("clear:")]
        assert clear_events == [
            "clear:patches",
            "clear:object_patches",
            "clear:weight_wrapper_patches",
        ]
        first_gc = events.index("gc")
        first_trim = events.index("trim")
        last_clear = max(i for i, name in enumerate(events) if name.startswith("clear:"))
        assert last_clear < first_gc < first_trim
        assert temporary_model.patches._cleared
        assert temporary_model.object_patches._cleared
        assert temporary_model.weight_wrapper_patches._cleared

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_gc_and_trim_still_run_when_no_patch_attrs(self):
        """When the temporary model has no patch-payload attributes at all,
        Python GC and the native heap trim still run. Severing is a best-effort
        prerequisite, not a gate."""
        from nodes.exit import _release_temporary_checkpoint_model

        events: list[str] = []

        class TempModelNoAttrs:
            pass

        temporary_model = TempModelNoAttrs()

        with (
            patch(
                "nodes.exit._unpatch_loaded_clones",
                side_effect=lambda m: events.append("unpatch"),
            ),
            patch("nodes.exit.gc.collect", side_effect=lambda: events.append("gc")),
            patch("nodes.exit.torch.cuda.is_available", return_value=False),
            patch(
                "nodes.exit._trim_native_heap",
                create=True,
                side_effect=lambda: events.append("trim"),
            ),
        ):
            _release_temporary_checkpoint_model(temporary_model)

        assert events == ["unpatch", "gc", "trim"]
