"""Tests for checkpoint CLIP/VAE component sourcing through WIDEN Entry payload.

Covers @saved-model-artifact-safety ACs:
  ac-missing-components-fail-before-work
  ac-missing-component-diagnostic-names-requirement
  ac-missing-component-diagnostic-gives-guidance
"""

from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from lib.recipe import CheckpointComponents, RecipeBase, RecipeLoRA, RecipeMerge, RecipeModel
from nodes.entry import WIDENEntryNode
from nodes.exit import WIDENExitNode, validate_checkpoint_components
from tests.conftest import MockModelPatcher

# =============================================================================
# Entry node: optional CLIP and VAE inputs
# =============================================================================


class TestEntryNodeAcceptsCLIPAndVAE:
    """WIDEN Entry accepts optional CLIP and VAE and stores them on RecipeBase."""

    def test_entry_with_clip_and_vae_stores_checkpoint_components(self):
        """Entry node stores CLIP and VAE as CheckpointComponents on RecipeBase."""
        patcher = MockModelPatcher()
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")

        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher, clip=clip, vae=vae)

        assert isinstance(recipe, RecipeBase)
        assert recipe.checkpoint_components is not None
        assert isinstance(recipe.checkpoint_components, CheckpointComponents)
        assert recipe.checkpoint_components.clip is clip
        assert recipe.checkpoint_components.vae is vae

    def test_entry_input_types_includes_optional_clip_and_vae(self):
        """INPUT_TYPES declares optional CLIP and VAE inputs."""
        input_types = WIDENEntryNode.INPUT_TYPES()

        assert "optional" in input_types
        assert "clip" in input_types["optional"]
        assert input_types["optional"]["clip"] == ("CLIP",)
        assert "vae" in input_types["optional"]
        assert input_types["optional"]["vae"] == ("VAE",)

    def test_entry_required_model_unchanged(self):
        """Required MODEL input is unchanged."""
        input_types = WIDENEntryNode.INPUT_TYPES()

        assert "required" in input_types
        assert "model" in input_types["required"]
        assert input_types["required"]["model"] == ("MODEL",)


class TestEntryNodeWithoutCLIPVAE:
    """WIDEN Entry with no CLIP/VAE still returns a valid WIDEN recipe."""

    def test_entry_without_clip_vae_has_none_checkpoint_components(self):
        """Entry node without CLIP/VAE produces RecipeBase with checkpoint_components=None."""
        patcher = MockModelPatcher()
        node = WIDENEntryNode()
        (recipe,) = node.entry(patcher)

        assert isinstance(recipe, RecipeBase)
        assert recipe.checkpoint_components is None

    def test_existing_recipes_without_checkpoint_components_remain_valid(self):
        """Directly constructed RecipeBase without checkpoint_components is valid."""
        patcher = MockModelPatcher()
        recipe = RecipeBase(model_patcher=patcher, arch="sdxl")

        assert recipe.checkpoint_components is None


# =============================================================================
# CheckpointComponents dataclass
# =============================================================================


class TestCheckpointComponentsDataclass:
    """CheckpointComponents is frozen and stores clip and vae."""

    def test_checkpoint_components_is_frozen(self):
        """CheckpointComponents is immutable."""
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=clip, vae=vae)

        with pytest.raises(AttributeError):
            cc.clip = MagicMock()  # type: ignore[misc]

    def test_checkpoint_components_stores_references(self):
        """CheckpointComponents stores the exact references passed in."""
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=clip, vae=vae)

        assert cc.clip is clip
        assert cc.vae is vae


# =============================================================================
# AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
# =============================================================================


class TestMissingComponentsFailBeforeWork:
    """AC: @saved-model-artifact-safety ac-missing-components-fail-before-work

    Given: a checkpoint-style save is requested without CLIP and VAE.
    When: the Exit node validates the save request.
    Then: the workflow fails before expensive merge work or artifact publication begins.
    """

    def test_save_model_missing_both_clip_and_vae_raises_before_analyze(self):
        """save_model=True with checkpoint intent but no CLIP/VAE raises ValueError."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        # Use explicit CheckpointComponents(clip=None, vae=None) to signal
        # checkpoint intent while having both components missing.
        patcher = MockModelPatcher()
        cc = CheckpointComponents(clip=None, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with pytest.raises(ValueError):
            node.execute(merge, save_model=True, model_name="test.safetensors")

    def test_save_model_missing_clip_only_raises(self):
        """save_model=True with VAE but no CLIP raises ValueError."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        patcher = MockModelPatcher()
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=None, vae=vae)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with pytest.raises(ValueError):
            node.execute(merge, save_model=True, model_name="test.safetensors")

    def test_save_model_missing_vae_only_raises(self):
        """save_model=True with CLIP but no VAE raises ValueError."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        patcher = MockModelPatcher()
        clip = MagicMock(name="clip")
        cc = CheckpointComponents(clip=clip, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with pytest.raises(ValueError):
            node.execute(merge, save_model=True, model_name="test.safetensors")

    def test_save_model_noop_recipe_missing_components_raises(self):
        """save_model=True on RecipeBase (no-op) with checkpoint intent but no CLIP/VAE raises."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        patcher = MockModelPatcher()
        cc = CheckpointComponents(clip=None, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)

        node = WIDENExitNode()

        with pytest.raises(ValueError):
            node.execute(base, save_model=True, model_name="test.safetensors")

    def test_validation_happens_before_analyze_recipe(self):
        """Validation occurs before analyze_recipe is called."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        patcher = MockModelPatcher()
        cc = CheckpointComponents(clip=None, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.walk_to_base") as mock_walk,
        ):
            mock_walk.return_value = base

            with pytest.raises(ValueError):
                node.execute(merge, save_model=True, model_name="test.safetensors")

            # analyze_recipe should NOT have been called
            mock_analyze.assert_not_called()

    def test_validation_happens_before_model_state_dict(self):
        """Validation occurs before model_state_dict() is called."""
        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        patcher = MockModelPatcher()
        patcher.model_state_dict = MagicMock(wraps=patcher.model_state_dict)
        cc = CheckpointComponents(clip=None, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with pytest.raises(ValueError):
            node.execute(merge, save_model=True, model_name="test.safetensors")

        # model_state_dict should NOT have been called
        patcher.model_state_dict.assert_not_called()

    def test_checkpoint_recipe_model_without_base_components_fails_early(self):
        """RecipeModel(source_dir='checkpoints') with base.checkpoint_components=None fails.

        AC: @saved-model-artifact-safety ac-missing-components-fail-before-work

        A recipe tree classified as checkpoint-style by _recipe_has_checkpoint_components
        (via RecipeModel source_dir) must still fail validation when the base
        has no checkpoint_components, rather than proceeding into merge work
        and crashing with AttributeError.
        """
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        model = RecipeModel(
            path="clip.safetensors",
            strength=1.0,
            source_dir="checkpoints",
        )
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=0.5)

        node = WIDENExitNode()

        with patch("nodes.exit.analyze_recipe") as mock_analyze:
            with pytest.raises(ValueError, match=r"(?i)CLIP.*VAE|VAE.*CLIP"):
                node.execute(merge, save_model=True, model_name="test.safetensors")

            # analyze_recipe must NOT have been called — failure is before work
            mock_analyze.assert_not_called()


# =============================================================================
# AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
# =============================================================================


class TestMissingComponentDiagnosticNamesRequirement:
    """AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement

    Given: a checkpoint-style save is requested without required components.
    When: the Exit node reports the validation failure.
    Then: the error names the missing requirement.
    """

    def test_missing_both_names_clip_and_vae(self):
        """Error names both CLIP and VAE when both are missing."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError, match=r"(?i)CLIP.*VAE|VAE.*CLIP"):
            validate_checkpoint_components(base, save_model=True)

    def test_missing_clip_names_clip(self):
        """Error names CLIP when only CLIP is missing."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
        patcher = MockModelPatcher()
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=None, vae=vae)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)

        with pytest.raises(ValueError, match=r"(?i)CLIP"):
            validate_checkpoint_components(base, save_model=True)

    def test_missing_vae_names_vae(self):
        """Error names VAE when only VAE is missing."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
        patcher = MockModelPatcher()
        clip = MagicMock(name="clip")
        cc = CheckpointComponents(clip=clip, vae=None)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)

        with pytest.raises(ValueError, match=r"(?i)VAE"):
            validate_checkpoint_components(base, save_model=True)

    def test_error_mentions_checkpoint_save_model(self):
        """Error message mentions checkpoint save_model requirement."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_components(base, save_model=True)

        msg = str(exc_info.value)
        assert "save_model" in msg.lower() or "checkpoint" in msg.lower()


# =============================================================================
# AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance
# =============================================================================


class TestMissingComponentDiagnosticGivesGuidance:
    """AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance

    Given: a checkpoint-style save is requested without required components.
    When: the Exit node reports the validation failure.
    Then: the error explains how the workflow can provide the missing requirement.
    """

    def test_guidance_mentions_checkpointloadersimple(self):
        """Error gives guidance mentioning CheckpointLoaderSimple."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_components(base, save_model=True)

        msg = str(exc_info.value)
        assert "CheckpointLoaderSimple" in msg

    def test_guidance_mentions_widen_entry(self):
        """Error gives guidance mentioning connecting to WIDEN Entry."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_components(base, save_model=True)

        msg = str(exc_info.value)
        assert "WIDEN Entry" in msg

    def test_guidance_mentions_model_clip_vae_outputs(self):
        """Error gives guidance mentioning MODEL, CLIP, and VAE outputs."""
        # AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_components(base, save_model=True)

        msg = str(exc_info.value)
        assert "MODEL" in msg
        assert "CLIP" in msg
        assert "VAE" in msg


# =============================================================================
# save_model=False: no CLIP/VAE required
# =============================================================================


class TestSaveModelFalseNoValidation:
    """save_model=False must not require CLIP or VAE."""

    def test_save_model_false_without_clip_vae_succeeds(self):
        """save_model=False with no CLIP/VAE does not raise."""
        patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")

        # validate_checkpoint_components should be a no-op for save_model=False
        validate_checkpoint_components(base, save_model=False)  # should not raise

    def test_save_model_false_preserves_patch_output(self, mock_model_patcher):
        """save_model=False returns patched model without requiring CLIP/VAE."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")

        node = WIDENExitNode()
        (result,) = node.execute(base, save_model=False)

        # Should return a clone (patch mode behavior preserved)
        assert result is not mock_model_patcher


# =============================================================================
# Valid MODEL+CLIP+VAE proceeds past validation
# =============================================================================


class TestValidComponentsPassValidation:
    """Valid MODEL+CLIP+VAE proceeds to the checkpoint artifact writer seam."""

    def test_valid_components_pass_validation(self):
        """validate_checkpoint_components does not raise with valid components."""
        patcher = MockModelPatcher()
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=clip, vae=vae)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)

        # Should not raise
        validate_checkpoint_components(base, save_model=True)

    def test_valid_components_proceed_to_artifact_writer_in_execute(self):
        """With valid CLIP+VAE, execute proceeds to the checkpoint save seam."""
        patcher = MockModelPatcher()
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        cc = CheckpointComponents(clip=clip, vae=vae)
        base = RecipeBase(model_patcher=patcher, arch="sdxl", checkpoint_components=cc)
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()
        mock_loader = MagicMock(name="loader")
        mock_loader.loaded_bytes = 0
        mock_loader.cleanup = MagicMock()

        # With valid components, execution should proceed past validation and
        # reach the checkpoint save seam (save_comfy_checkpoint) without
        # real checkpoint I/O.
        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_model_analyze,
            patch("nodes.exit.walk_to_base", return_value=base),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit._build_lora_resolver") as mock_lr,
            patch("nodes.exit._build_model_resolver") as mock_mr,
            patch("nodes.exit.compute_base_identity", return_value="base-id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="test.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value="/tmp/test.safetensors"),
            patch("nodes.exit.serialize_recipe", return_value="serialized"),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.compile_plan", return_value=object()),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=patcher.clone()),
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch.multiple(
                "nodes.exit",
                check_ram_preflight=DEFAULT,
                _load_checkpoint_artifact=MagicMock(return_value=patcher.clone()),
            ),
            patch("nodes.exit.ProgressBar", None),
        ):
            mock_lr.return_value = lambda name: None
            mock_mr.return_value = lambda name, src: None
            mock_analyze.return_value = SimpleNamespace(
                loader=mock_loader,
                set_affected={},
                affected_keys=set(),
                arch="sdxl",
            )
            mock_model_analyze.return_value = SimpleNamespace(
                model_affected={},
                model_loaders={},
                all_model_keys=frozenset(),
            )

            # Should not raise ValueError — validation passed
            node.execute(merge, save_model=True, model_name="test.safetensors")
            mock_analyze.assert_called_once()
            mock_model_analyze.assert_called_once()
            # Checkpoint-style recipe routes to save_comfy_checkpoint
            mock_save_ckpt.assert_called_once()
            call_kwargs = mock_save_ckpt.call_args
            assert call_kwargs.kwargs["clip"] is clip
            assert call_kwargs.kwargs["vae"] is vae
