"""Tests for checkpoint artifact save through Comfy checkpoint semantics.

AC coverage for:
- @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
- @saved-model-artifact-safety ac-missing-metadata-not-reused
- @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
- @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
- @saved-model-artifact-safety ac-no-partial-publication
- @saved-model-artifact-safety ac-existing-valid-artifact-preserved
- @exit-model-persistence ac-2, ac-8, ac-10
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from lib.persistence import (
    check_checkpoint_cache,
)
from lib.recipe import (
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
)
from nodes.exit import WIDENExitNode, save_comfy_checkpoint
from tests.conftest import make_checkpoint_components

# Representative Comfy checkpoint-style tensors for mock side effects.
# Must include all three component groups to pass _classify_temp_artifact.
_MOCK_CHECKPOINT_TENSORS = {
    "model.diffusion_model.input_blocks.0.weight": torch.zeros(1),
    "conditioner.embedders.0.weight": torch.zeros(1),
    "first_stage_model.decoder.weight": torch.zeros(1),
}


# =============================================================================
# save_comfy_checkpoint — unit tests
# =============================================================================


class TestSaveComfyCheckpoint:
    """Tests for save_comfy_checkpoint wrapper function."""

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_calls_comfy_sd_save_checkpoint(self, tmp_path):
        """save_comfy_checkpoint must call comfy.sd.save_checkpoint with
        merged MODEL, CLIP, VAE, and ecaj metadata."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        with patch("comfy.sd.save_checkpoint") as mock_save:
            # comfy.sd.save_checkpoint writes to the temp path — simulate by
            # creating the temp file so os.fsync/os.replace succeed.
            def side_effect(path, model_arg, **kwargs):
                save_file(
                    _MOCK_CHECKPOINT_TENSORS, path,
                    metadata=kwargs.get("metadata", {}),
                )
            mock_save.side_effect = side_effect

            save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args
            # Positional: (tmp_path, model)
            assert call_kwargs.args[1] is model
            # Keywords: clip, vae, metadata
            assert call_kwargs.kwargs["clip"] is clip
            assert call_kwargs.kwargs["vae"] is vae
            assert call_kwargs.kwargs["metadata"] is metadata

    # AC: @exit-model-persistence ac-10
    def test_atomic_publication(self, tmp_path):
        """save_comfy_checkpoint must write to temp, then atomically replace target."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        temp_paths_seen = []

        def side_effect(path, model_arg, **kwargs):
            # Record that we're writing to a temp path, not the final path
            temp_paths_seen.append(path)
            assert path != save_path, "Must write to temp path, not final target"
            assert ".ecaj_tmp_" in path, "Temp path should contain .ecaj_tmp_ prefix"
            save_file(
                _MOCK_CHECKPOINT_TENSORS, path,
                metadata=kwargs.get("metadata", {}),
            )

        with patch("comfy.sd.save_checkpoint", side_effect=side_effect):
            save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        # Final file should exist at save_path
        assert os.path.exists(save_path)
        # Temp file should NOT exist (was renamed)
        for tp in temp_paths_seen:
            assert not os.path.exists(tp)

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_failed_save_does_not_publish(self, tmp_path):
        """If comfy.sd.save_checkpoint fails, no artifact is published."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1"}

        with patch("comfy.sd.save_checkpoint", side_effect=RuntimeError("save failed")):
            with pytest.raises(RuntimeError, match="save failed"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        # No file should exist at save_path
        assert not os.path.exists(save_path)

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_failed_save_preserves_existing_ecaj_artifact(self, tmp_path):
        """If save fails, a previously valid ecaj artifact at the target remains intact."""
        save_path = str(tmp_path / "model.safetensors")
        # Create a pre-existing valid ecaj artifact
        original_tensor = torch.randn(4, 4)
        save_file(
            {"weight": original_tensor},
            save_path,
            metadata={"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"},
        )

        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        with patch("comfy.sd.save_checkpoint", side_effect=RuntimeError("save failed")):
            with pytest.raises(RuntimeError, match="save failed"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        # Original artifact must still exist and be unchanged
        assert os.path.exists(save_path)
        loaded = load_file(save_path)
        assert torch.equal(loaded["weight"], original_tensor)

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_refuses_overwrite_non_ecaj_file(self, tmp_path):
        """save_comfy_checkpoint must refuse to overwrite a non-ecaj file."""
        save_path = str(tmp_path / "model.safetensors")
        # Create a pre-existing non-ecaj safetensors file (no __ecaj_version__)
        original_tensor = torch.randn(4, 4)
        save_file({"weight": original_tensor}, save_path)

        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        with patch("comfy.sd.save_checkpoint") as mock_save:
            with pytest.raises(ValueError, match="not an ecaj-saved model"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

            # comfy.sd.save_checkpoint must NOT have been called
            mock_save.assert_not_called()

        # Original file must be preserved
        loaded = load_file(save_path)
        assert torch.equal(loaded["weight"], original_tensor)

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_failed_save_cleans_up_temp(self, tmp_path):
        """If save fails, the temp file is cleaned up."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1"}

        def side_effect(path, model_arg, **kwargs):
            # Write something to temp before failing
            save_file({"dummy": torch.zeros(1)}, path)
            raise RuntimeError("save failed after write")

        with patch("comfy.sd.save_checkpoint", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="save failed after write"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        # No temp files should remain
        tmp_files = [f for f in os.listdir(str(tmp_path)) if f.startswith(".ecaj_tmp_")]
        assert tmp_files == []

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_classification_failure_does_not_publish(self, tmp_path):
        """If comfy.sd.save_checkpoint writes a temp file without ecaj metadata,
        _classify_temp_artifact rejects it and the artifact is not published."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        # Caller passes correct metadata, but comfy.sd.save_checkpoint ignores
        # it (simulating a bug or misconfiguration).
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        def side_effect(path, model_arg, **kwargs):
            # Write without ecaj metadata — simulates Comfy ignoring our metadata
            save_file({"dummy": torch.zeros(1)}, path)

        with patch("comfy.sd.save_checkpoint", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="missing ecaj metadata"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        # No file at save_path
        assert not os.path.exists(save_path)
        # No temp files should remain
        tmp_files = [f for f in os.listdir(str(tmp_path)) if f.startswith(".ecaj_tmp_")]
        assert tmp_files == []

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_wrong_artifact_kind_not_published(self, tmp_path):
        """If temp artifact has wrong artifact_kind, it is not published."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        def side_effect(path, model_arg, **kwargs):
            # Write with wrong artifact kind
            save_file(
                {"dummy": torch.zeros(1)}, path,
                metadata={"__ecaj_version__": "1", "__ecaj_artifact_kind__": "diffusion"},
            )

        with patch("comfy.sd.save_checkpoint", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="artifact kind"):
                save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=metadata)

        assert not os.path.exists(save_path)

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_incomplete_checkpoint_components_not_published(self, tmp_path):
        """If temp checkpoint artifact is missing required component prefixes
        (e.g. only conditioner.* keys), it must not be published."""
        save_path = str(tmp_path / "model.safetensors")
        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")
        metadata = {"__ecaj_version__": "1", "__ecaj_artifact_kind__": "checkpoint"}

        def side_effect(path, model_arg, **kwargs):
            # Write a checkpoint-metadata-valid file but with only conditioner keys
            save_file(
                {"conditioner.embedders.0.weight": torch.zeros(4, 4)},
                path,
                metadata=kwargs.get("metadata", {}),
            )

        with patch("comfy.sd.save_checkpoint", side_effect=side_effect):
            with pytest.raises(
                RuntimeError, match="missing required component prefixes"
            ):
                save_comfy_checkpoint(
                    save_path, model, clip=clip, vae=vae, metadata=metadata
                )

        assert not os.path.exists(save_path)
        # No temp files should remain
        tmp_files = [
            f for f in os.listdir(str(tmp_path)) if f.startswith(".ecaj_tmp_")
        ]
        assert tmp_files == []


# =============================================================================
# Checkpoint save integration — exit node routes checkpoint saves to Comfy path
# =============================================================================


class TestCheckpointSaveRouting:
    """Verify checkpoint-style saves call save_comfy_checkpoint instead of
    writing through MaterializationSink."""

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    # AC: @exit-model-persistence ac-2
    def test_checkpoint_save_calls_save_comfy_checkpoint(self, mock_model_patcher, tmp_path):
        """Checkpoint-style recipe must call save_comfy_checkpoint with merged
        MODEL, CLIP, VAE, and ecaj metadata."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        compose = RecipeCompose(branches=(lora, model))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation") as mock_chunked,
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch("nodes.exit.check_ram_preflight"),
        ):
            affected_key = "diffusion_model.input_blocks.0.0.weight"
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )
            merged_model = mock_model_patcher.clone()
            mock_install.return_value = merged_model
            mock_chunked.return_value = {affected_key: torch.randn(4, 4)}

            node.execute(merge, save_model=True, model_name="model")

            # save_comfy_checkpoint MUST have been called
            mock_save_ckpt.assert_called_once()
            call_kwargs = mock_save_ckpt.call_args
            # save_path
            assert call_kwargs.args[0] == save_path
            # merged_model
            assert call_kwargs.args[1] is merged_model
            # CLIP and VAE from checkpoint_components
            assert call_kwargs.kwargs["clip"] is cc.clip
            assert call_kwargs.kwargs["vae"] is cc.vae
            # metadata must include checkpoint artifact kind
            meta = call_kwargs.kwargs["metadata"]
            assert meta["__ecaj_artifact_kind__"] == "checkpoint"
            assert meta["__ecaj_checkpoint_components__"] == "true"

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_checkpoint_save_does_not_use_materialization_sink(self, mock_model_patcher, tmp_path):
        """Checkpoint-style save must NOT use MaterializationSink."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.MaterializationSink") as mock_sink_cls,
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
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )
            mock_install.return_value = mock_model_patcher.clone()

            node.execute(merge, save_model=True, model_name="model")

            # MaterializationSink must NOT be instantiated for checkpoint saves
            mock_sink_cls.assert_not_called()

    def test_non_checkpoint_save_uses_materialization_sink(self, mock_model_patcher, tmp_path):
        """Non-checkpoint (diffusion-only, no checkpoint_components) save must use
        MaterializationSink, not save_comfy_checkpoint.

        Note: validate_checkpoint_components normally rejects save_model=True
        without checkpoint_components. This test bypasses validation to verify
        the branching logic in _execute_full_saved_model.
        """
        # No checkpoint_components → diffusion-only source
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        mock_sink = MagicMock()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),  # bypass validation
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.streaming_evaluation_to_sink"),
            patch("nodes.exit.MaterializationSink", return_value=mock_sink) as mock_sink_cls,
            patch("nodes.exit._load_model_from_artifact") as mock_load,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch("nodes.exit.check_ram_preflight"),
        ):
            affected_key = "diffusion_model.input_blocks.0.0.weight"
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )
            mock_load.return_value = mock_model_patcher.clone()

            node.execute(merge, save_model=True, model_name="model")

            # MaterializationSink MUST have been used
            mock_sink_cls.assert_called_once()
            # save_comfy_checkpoint must NOT have been called
            mock_save_ckpt.assert_not_called()

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_checkpoint_base_plus_lora_routes_as_checkpoint(self, mock_model_patcher, tmp_path):
        """RecipeBase with checkpoint_components + LoRA merge must route through
        save_comfy_checkpoint, not MaterializationSink."""
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
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation") as mock_chunked,
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch("nodes.exit.MaterializationSink") as mock_sink_cls,
            patch("nodes.exit.check_ram_preflight"),
        ):
            affected_key = "diffusion_model.input_blocks.0.0.weight"
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )
            mock_install.return_value = mock_model_patcher.clone()
            mock_chunked.return_value = {affected_key: torch.randn(4, 4)}

            node.execute(merge, save_model=True, model_name="model")

            # save_comfy_checkpoint MUST be called (checkpoint-style)
            mock_save_ckpt.assert_called_once()
            call_kwargs = mock_save_ckpt.call_args
            assert call_kwargs.kwargs["clip"] is cc.clip
            assert call_kwargs.kwargs["vae"] is cc.vae
            # MaterializationSink must NOT be used
            mock_sink_cls.assert_not_called()


# =============================================================================
# Checkpoint artifact metadata and cache classification
# =============================================================================


class TestCheckpointArtifactMetadata:
    """AC: @saved-model-artifact-safety — metadata must be written and required
    for checkpoint cache reuse."""

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_artifact_kind_metadata_is_written(self, mock_model_patcher, tmp_path):
        """Checkpoint save must include artifact_kind='checkpoint' in metadata."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint"),
            patch("nodes.exit.build_metadata") as mock_build_meta,
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
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )
            mock_build_meta.return_value = {
                "__ecaj_version__": "1",
                "__ecaj_artifact_kind__": "checkpoint",
            }
            mock_install.return_value = mock_model_patcher.clone()

            node.execute(merge, save_model=True, model_name="model")

            mock_build_meta.assert_called_once()
            call_kwargs = mock_build_meta.call_args
            assert call_kwargs.kwargs.get("artifact_kind") == "checkpoint"
            assert call_kwargs.kwargs.get("checkpoint_components") is True
            assert call_kwargs.kwargs.get("base_identity") == "base_id"


# =============================================================================
# Internal-format artifact rejection for checkpoint cache
# =============================================================================


class TestInternalFormatRejection:
    """AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache —
    Internal WIDEN artifacts with diffusion_model/noise_augmentor/model_sampling-only
    keys must be rejected as checkpoint cache hits."""

    _BASE_IDENTITY = "base_sha256_abc"
    _DEPS = json.dumps({"lora.safetensors": [1234.5, 100]})

    def _make_file(self, path, tensors, **kwargs):
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": kwargs.get("recipe_hash", "abc123"),
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": kwargs.get("artifact_kind", "checkpoint"),
            "__ecaj_base_identity__": kwargs.get("base_identity", self._BASE_IDENTITY),
            "__ecaj_dependency_fingerprints__": kwargs.get("deps", self._DEPS),
            "__ecaj_checkpoint_components__": "true",
        }
        save_file(tensors, str(path), metadata=metadata)

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_diffusion_model_only_keys_rejected(self, tmp_path):
        """Artifact with only diffusion_model.* keys is rejected as checkpoint cache hit."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_noise_augmentor_only_keys_rejected(self, tmp_path):
        """Artifact with diffusion_model + noise_augmentor keys only is rejected."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "diffusion_model.layers.0.weight": torch.randn(4, 4),
            "noise_augmentor.weight": torch.randn(4, 4),
            "model_sampling.sigmas": torch.randn(4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_checkpoint_with_vae_keys_accepted(self, tmp_path):
        """Artifact with diffusion + VAE keys (proper checkpoint) IS accepted."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
            "cond_stage_model.transformer.weight": torch.randn(4, 4),
        })
        assert check_checkpoint_cache(str(path), "abc123", self._BASE_IDENTITY, self._DEPS) is True

    # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
    def test_wrong_artifact_kind_rejected(self, tmp_path):
        """Artifact with artifact_kind='diffusion' is rejected for checkpoint cache."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        }, artifact_kind="diffusion")
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_missing_metadata_rejected(self, tmp_path):
        """Artifact without ecaj metadata raises ValueError (non-ecaj file)."""
        path = tmp_path / "model.safetensors"
        save_file({"weight": torch.randn(4, 4)}, str(path))
        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_checkpoint_cache(str(path), "abc123", self._BASE_IDENTITY, self._DEPS)

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_conditioner_only_rejected(self, tmp_path):
        """Metadata-valid artifact with only conditioner.* keys is rejected —
        missing diffusion and VAE components."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "conditioner.embedders.0.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_missing_vae_component_rejected(self, tmp_path):
        """Artifact with diffusion + conditioning but no VAE keys is rejected."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_missing_conditioning_component_rejected(self, tmp_path):
        """Artifact with diffusion + VAE but no conditioning keys is rejected."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_missing_diffusion_component_rejected(self, tmp_path):
        """Artifact with conditioning + VAE but no diffusion keys is rejected."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_comfy_style_full_checkpoint_accepted(self, tmp_path):
        """Artifact with Comfy-style model.* + conditioner.* + first_stage_model.*
        keys is accepted."""
        path = tmp_path / "model.safetensors"
        self._make_file(path, {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        })
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is True


# =============================================================================
# Failed checkpoint save safety
# =============================================================================


class TestCheckpointSaveSafety:
    """AC: @saved-model-artifact-safety — failed saves must not clobber existing
    valid artifacts or leave partial artifacts."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # AC: @exit-model-persistence ac-10
    def test_failed_checkpoint_save_no_artifact_published(self, mock_model_patcher, tmp_path):
        """If checkpoint save fails, no artifact is published at the save path."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint", side_effect=RuntimeError("save failed")),
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
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )
            mock_install.return_value = mock_model_patcher.clone()

            with pytest.raises(RuntimeError, match="save failed"):
                node.execute(merge, save_model=True, model_name="model")

        # No artifact published
        assert not os.path.exists(save_path)

    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    def test_failed_checkpoint_save_preserves_existing(self, mock_model_patcher, tmp_path):
        """If checkpoint save fails, existing valid artifact at target is preserved."""
        save_path = str(tmp_path / "model.safetensors")
        # Create pre-existing valid artifact
        original = torch.randn(4, 4)
        save_file({"weight": original}, save_path)

        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint", side_effect=RuntimeError("fail")),
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
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )
            mock_install.return_value = mock_model_patcher.clone()

            with pytest.raises(RuntimeError):
                node.execute(merge, save_model=True, model_name="model")

        # Original artifact preserved
        loaded = load_file(save_path)
        assert torch.equal(loaded["weight"], original)


# =============================================================================
# Monkeypatched Comfy save/load smoke test
# =============================================================================


class TestCheckpointSaveLoadSmokeTest:
    """Monkeypatched Comfy save/load smoke test proving the checkpoint path
    is selected without requiring real model weights.

    AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    AC: @exit-model-persistence ac-8
    """

    def test_checkpoint_save_load_round_trip(self, tmp_path):
        """Behavioral save/load smoke test: save via save_comfy_checkpoint,
        then load through a monkeypatched CheckpointLoaderSimple-style path
        that returns MODEL, CLIP, and VAE components.

        This proves:
        1. The saved artifact contains Comfy checkpoint-style keys.
        2. A checkpoint loader can split those keys into diffusion (MODEL),
           conditioning (CLIP), and decode (VAE) components.
        3. No internal-format-only keys leak through.
        """
        save_path = str(tmp_path / "model.safetensors")

        # Representative checkpoint-style state dict that Comfy would produce
        # via model.state_dict_for_saving(clip_sd, vae_sd).
        comfy_checkpoint_tensors = {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.conv_in.weight": torch.randn(4, 4),
        }

        ecaj_metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "test_hash",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_checkpoint_components__": "true",
        }

        # Simulate comfy.sd.save_checkpoint by writing the state dict directly.
        def mock_comfy_save(path, model, clip=None, vae=None, metadata=None, **kwargs):
            save_file(comfy_checkpoint_tensors, path, metadata=metadata or {})

        model = MagicMock(name="model")
        clip = MagicMock(name="clip")
        vae = MagicMock(name="vae")

        # --- SAVE side ---
        with patch("comfy.sd.save_checkpoint", side_effect=mock_comfy_save):
            save_comfy_checkpoint(save_path, model, clip=clip, vae=vae, metadata=ecaj_metadata)

        # --- LOAD side: CheckpointLoaderSimple-style component extraction ---
        # Comfy's load_checkpoint_guess_config reads the safetensors file and
        # partitions keys by prefix into MODEL, CLIP, and VAE state dicts.
        # We replicate that routing to prove our artifact is loadable.
        from safetensors import safe_open

        with safe_open(save_path, framework="pt") as f:
            loaded_metadata = f.metadata()
            all_keys = set(f.keys())
            loaded_state = {k: f.get_tensor(k) for k in f.keys()}

        # Partition keys exactly as CheckpointLoaderSimple does:
        # model.* → MODEL (diffusion), conditioner.* → CLIP, first_stage_model.* → VAE
        model_keys = {k for k in all_keys if k.startswith("model.")}
        clip_keys = {k for k in all_keys if k.startswith("conditioner.")}
        vae_keys = {k for k in all_keys if k.startswith("first_stage_model.")}

        # All three component sets must be non-empty for a valid checkpoint
        assert model_keys, "Checkpoint loader found no MODEL (model.*) keys"
        assert clip_keys, "Checkpoint loader found no CLIP (conditioner.*) keys"
        assert vae_keys, "Checkpoint loader found no VAE (first_stage_model.*) keys"

        # Components must cover all keys — no unclassified orphan keys
        classified = model_keys | clip_keys | vae_keys
        orphan_keys = all_keys - classified
        assert not orphan_keys, (
            f"Checkpoint has keys not routable to MODEL/CLIP/VAE: {orphan_keys}"
        )

        # Simulate returning (MODEL, CLIP, VAE) from the loader — verify
        # each component's state dict contains actual tensor data.
        model_sd = {k: loaded_state[k] for k in model_keys}
        clip_sd = {k: loaded_state[k] for k in clip_keys}
        vae_sd = {k: loaded_state[k] for k in vae_keys}

        assert all(isinstance(v, torch.Tensor) for v in model_sd.values())
        assert all(isinstance(v, torch.Tensor) for v in clip_sd.values())
        assert all(isinstance(v, torch.Tensor) for v in vae_sd.values())

        # Must NOT contain internal-format-only keys
        for key in all_keys:
            assert not key.startswith("diffusion_model."), (
                f"Found internal-format key {key!r} — checkpoint must use "
                "model.diffusion_model.* prefix, not bare diffusion_model.*"
            )

        # Ecaj metadata must survive the save/load round trip
        assert loaded_metadata is not None
        assert loaded_metadata.get("__ecaj_artifact_kind__") == "checkpoint"

    def test_checkpoint_artifact_not_accepted_as_internal_format_cache(self, tmp_path):
        """A proper checkpoint artifact with Comfy keys must be accepted by
        check_checkpoint_cache (not rejected as internal-only)."""
        path = tmp_path / "model.safetensors"
        base_id = "test_base_id"
        deps = json.dumps({})

        # Create checkpoint with Comfy key prefixes + ecaj metadata
        tensors = {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        }
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "hash1",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_base_identity__": base_id,
            "__ecaj_dependency_fingerprints__": deps,
            "__ecaj_checkpoint_components__": "true",
        }
        save_file(tensors, str(path), metadata=metadata)

        result = check_checkpoint_cache(str(path), "hash1", base_id, deps)
        assert result is True, "Proper checkpoint artifact must be accepted as cache hit"


# =============================================================================
# save_model=False preserves in-memory patch output behavior
# =============================================================================


class TestPatchModePreserved:
    """AC: @exit-model-persistence ac-1 — save_model=False must preserve
    in-memory patch output behavior (no checkpoint save)."""

    def test_save_model_false_does_not_call_save_comfy_checkpoint(self, mock_model_patcher):
        """save_model=False must not invoke save_comfy_checkpoint."""
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=make_checkpoint_components(),
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENExitNode()

        with (
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation") as mock_chunked,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch("nodes.exit.check_ram_preflight"),
        ):
            affected_key = "diffusion_model.input_blocks.0.0.weight"
            mock_loader = MagicMock()
            mock_loader.cleanup = MagicMock()
            mock_loader.loaded_bytes = 0
            mock_analyze.return_value = MagicMock(
                model_patcher=mock_model_patcher,
                arch="sdxl",
                loader=mock_loader,
                set_affected={str(id(lora)): {affected_key}},
                affected_keys={affected_key},
            )
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={},
                model_affected={},
                all_model_keys=frozenset(),
            )
            mock_chunked.return_value = {
                affected_key: torch.randn(4, 4, dtype=torch.float32),
            }

            result = node.execute(merge, save_model=False, model_name="model")

            # save_comfy_checkpoint must NOT be called
            mock_save_ckpt.assert_not_called()
            # Result must be a tuple with a model
            assert len(result) == 1
            assert result[0] is not None


# =============================================================================
# Checkpoint save returns usable merged MODEL
# =============================================================================


class TestCheckpointSaveReturnModel:
    """AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable —
    checkpoint save must return a usable MODEL for downstream consumers."""

    def test_checkpoint_save_returns_merged_model(self, mock_model_patcher, tmp_path):
        """Checkpoint save must return the in-memory merged MODEL, not None."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

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
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch("nodes.exit.analyze_recipe_models") as mock_analyze_models,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.chunked_evaluation", return_value={}),
            patch("nodes.exit.install_merged_patches", return_value=expected_model),
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
            mock_model_loader = MagicMock()
            mock_model_loader.cleanup = MagicMock()
            mock_model_loader.loaded_bytes = 0
            mock_analyze_models.return_value = MagicMock(
                model_loaders={str(id(model)): mock_model_loader},
                model_affected={str(id(model)): frozenset()},
                all_model_keys=frozenset(),
            )

            result = node.execute(merge, save_model=True, model_name="model")

            assert len(result) == 1
            assert result[0] is expected_model


# =============================================================================
# Checkpoint cache hit returns correct MODEL with Comfy-style keys
# =============================================================================


class TestCheckpointCacheHitModelLoading:
    """AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    AC: @exit-model-persistence ac-8

    Valid checkpoint cache hits must load merged weights from the artifact
    into the returned MODEL, including when the artifact uses Comfy
    checkpoint-style key prefixes (model.diffusion_model.*).
    """

    # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    def test_load_model_from_checkpoint_artifact_updates_weights(
        self, mock_model_patcher, tmp_path,
    ):
        """_load_model_from_artifact must load Comfy checkpoint keys
        (model.diffusion_model.*) into the returned model's state dict."""
        from nodes.exit import _load_model_from_artifact

        save_path = str(tmp_path / "model.safetensors")

        # Simulate a Comfy checkpoint artifact with model.diffusion_model.* keys
        merged_weight = torch.randn(4, 4)
        artifact_tensors = {
            "model.diffusion_model.input_blocks.0.0.weight": merged_weight,
            "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        }
        save_file(artifact_tensors, save_path)

        original_weight = mock_model_patcher.model_state_dict()[
            "diffusion_model.input_blocks.0.0.weight"
        ].clone()

        result = _load_model_from_artifact(save_path, mock_model_patcher, torch.float32)

        # The returned model's state dict must reflect the artifact weights,
        # not the original base weights.
        loaded_weight = result.model_state_dict()["diffusion_model.input_blocks.0.0.weight"]
        assert torch.equal(loaded_weight, merged_weight), (
            "Checkpoint cache hit must load merged weights from model.diffusion_model.* keys"
        )
        assert not torch.equal(loaded_weight, original_weight), (
            "Returned model must not still have the original base weights"
        )

    # AC: @exit-model-persistence ac-8
    def test_load_model_from_internal_artifact_still_works(self, mock_model_patcher, tmp_path):
        """_load_model_from_artifact must continue to work with internal-format
        (diffusion_model.*) artifacts."""
        from nodes.exit import _load_model_from_artifact

        save_path = str(tmp_path / "model.safetensors")

        merged_weight = torch.randn(4, 4)
        artifact_tensors = {
            "diffusion_model.input_blocks.0.0.weight": merged_weight,
            "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        }
        save_file(artifact_tensors, save_path)

        result = _load_model_from_artifact(save_path, mock_model_patcher, torch.float32)

        loaded_weight = result.model_state_dict()["diffusion_model.input_blocks.0.0.weight"]
        assert torch.equal(loaded_weight, merged_weight), (
            "Internal-format artifact loading must still work"
        )


# =============================================================================
# Base-only checkpoint saves through checkpoint path (not diffusion)
# =============================================================================


class TestBaseOnlyCheckpointSave:
    """AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind

    A RecipeBase with checkpoint_components (no merge) in save_model mode
    must save through the checkpoint path, not the diffusion/MaterializationSink path.
    """

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_noop_checkpoint_base_uses_save_comfy_checkpoint(self, mock_model_patcher, tmp_path):
        """RecipeBase (no-op) with checkpoint_components must call save_comfy_checkpoint."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=False),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.install_merged_patches") as mock_install,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
            patch("nodes.exit.MaterializationSink") as mock_sink_cls,
        ):
            mock_install.return_value = mock_model_patcher.clone()

            result = node.execute(base, save_model=True, model_name="model")

            # save_comfy_checkpoint MUST be called
            mock_save_ckpt.assert_called_once()
            call_kwargs = mock_save_ckpt.call_args
            assert call_kwargs.kwargs["clip"] is cc.clip
            assert call_kwargs.kwargs["vae"] is cc.vae
            # metadata must include checkpoint artifact kind
            meta = call_kwargs.kwargs["metadata"]
            assert meta["__ecaj_artifact_kind__"] == "checkpoint"
            # MaterializationSink must NOT be used
            mock_sink_cls.assert_not_called()
            # Must return a model
            assert len(result) == 1
            assert result[0] is not None

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_noop_diffusion_base_uses_materialization_sink(self, mock_model_patcher, tmp_path):
        """RecipeBase (no-op) WITHOUT checkpoint_components must use MaterializationSink.

        Note: validate_checkpoint_components normally rejects save_model=True
        without checkpoint_components. This test bypasses validation to verify
        the noop branching logic.
        """
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        mock_sink = MagicMock()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),  # bypass validation
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.MaterializationSink", return_value=mock_sink) as mock_sink_cls,
            patch("nodes.exit._load_model_from_artifact") as mock_load,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
        ):
            mock_load.return_value = mock_model_patcher.clone()

            node.execute(base, save_model=True, model_name="model")

            # MaterializationSink MUST be used
            mock_sink_cls.assert_called_once()
            # save_comfy_checkpoint must NOT be called
            mock_save_ckpt.assert_not_called()

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_noop_checkpoint_cache_hit_uses_checkpoint_cache(self, mock_model_patcher, tmp_path):
        """RecipeBase (no-op) checkpoint with cache hit must use check_checkpoint_cache,
        not check_full_model_cache with artifact_kind=diffusion."""
        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )

        save_path = str(tmp_path / "model.safetensors")
        node = WIDENExitNode()

        with (
            patch("nodes.exit.validate_model_name", return_value="model.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash1"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_checkpoint_cache", return_value=True) as mock_ckpt_cache,
            patch("nodes.exit.check_full_model_cache") as mock_full_cache,
            patch("nodes.exit._unpatch_loaded_clones"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._load_model_from_artifact") as mock_load,
            patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt,
        ):
            mock_load.return_value = mock_model_patcher.clone()

            node.execute(base, save_model=True, model_name="model")

            # check_checkpoint_cache MUST be called (not check_full_model_cache)
            mock_ckpt_cache.assert_called_once()
            mock_full_cache.assert_not_called()
            # save_comfy_checkpoint must NOT be called (cache hit)
            mock_save_ckpt.assert_not_called()
            # Model must be loaded from artifact
            mock_load.assert_called_once()


# =============================================================================
# _recipe_has_checkpoint_components — classification tests
# =============================================================================


class TestRecipeHasCheckpointComponents:
    """Verify _recipe_has_checkpoint_components correctly identifies
    checkpoint-style recipes from RecipeBase.checkpoint_components."""

    def test_recipe_base_with_checkpoint_components(self, mock_model_patcher):
        """RecipeBase with valid checkpoint_components must be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        assert _recipe_has_checkpoint_components(base) is True

    def test_recipe_base_without_checkpoint_components(self, mock_model_patcher):
        """RecipeBase without checkpoint_components must not be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        assert _recipe_has_checkpoint_components(base) is False

    def test_merge_with_checkpoint_base_plus_lora(self, mock_model_patcher):
        """RecipeMerge with checkpoint RecipeBase + LoRA target must be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        cc = make_checkpoint_components()
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=cc,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        assert _recipe_has_checkpoint_components(merge) is True

    def test_merge_with_diffusion_base_plus_lora(self, mock_model_patcher):
        """RecipeMerge with diffusion-only RecipeBase + LoRA must not be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        assert _recipe_has_checkpoint_components(merge) is False

    def test_recipe_model_with_checkpoints_source(self):
        """RecipeModel with source_dir='checkpoints' must be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        model = RecipeModel(path="clip.safetensors", strength=1.0, source_dir="checkpoints")
        assert _recipe_has_checkpoint_components(model) is True

    def test_recipe_model_with_diffusion_source(self):
        """RecipeModel with source_dir='diffusion_models' must not be detected."""
        from nodes.exit import _recipe_has_checkpoint_components

        model = RecipeModel(path="unet.safetensors", strength=1.0, source_dir="diffusion_models")
        assert _recipe_has_checkpoint_components(model) is False
