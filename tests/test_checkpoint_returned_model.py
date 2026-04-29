"""Tests for checkpoint save_model return paths without deep-copying Comfy internals.

AC coverage for:
- @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
- @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
- @saved-model-artifact-safety ac-missing-metadata-not-reused
- @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
- @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

Task: @task-checkpoint-save-returned-model
Spec: @checkpoint-loadable-saved-model-output
"""

from __future__ import annotations

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
            patch("nodes.exit.compile_plan", return_value=MagicMock()),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
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
# Cache miss returns merged MODEL without invoking checkpoint loader
# =============================================================================


class TestCacheMissReturnsInMemoryModel:
    """Cache-miss checkpoint save must return the merged MODEL from the
    in-memory WIDEN merge result. It must not reload the just-saved artifact
    solely to produce the return value.

    AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    """

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_cache_miss_returns_install_merged_patches_result(self, mock_model_patcher, tmp_path):
        """The returned MODEL must be the result of install_merged_patches,
        not a model loaded from the artifact."""
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

        expected_model = mock_model_patcher.clone()

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
                return_value=expected_model,
            ) as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.check_ram_preflight"),
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

            assert result[0] is expected_model, (
                "Cache miss must return the in-memory merged MODEL from "
                "install_merged_patches, not reload from artifact"
            )
            mock_install.assert_called_once()

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_cache_miss_does_not_invoke_checkpoint_loader(self, mock_model_patcher, tmp_path):
        """Cache miss must not call _load_checkpoint_artifact or
        load_checkpoint_guess_config solely for the return value."""
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

            node.execute(merge, save_model=True, model_name="model")

            mock_ckpt_load.assert_not_called()
            mock_diffusion_load.assert_not_called()


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
