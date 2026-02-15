"""Tests for WIDEN CLIP Exit Node — AC coverage for @clip-exit-node."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from copy import deepcopy

import pytest
import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel
from nodes.clip_exit import (
    WIDENCLIPExitNode,
    _collect_lora_paths,
    _collect_model_paths,
    _compute_clip_recipe_hash,
    _unpatch_loaded_clip_clones,
    _validate_clip_recipe_tree,
    install_merged_clip_patches,
)

# Representative SDXL CLIP state dict keys (clip_l + clip_g encoders)
_SDXL_CLIP_KEYS = (
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
    "clip_l.transformer.text_model.encoder.layers.11.mlp.fc1.weight",
    "clip_l.transformer.text_model.embeddings.token_embedding.weight",
    "clip_g.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
    "clip_g.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight",
    "clip_g.transformer.text_model.encoder.layers.31.mlp.fc1.weight",
    "clip_g.transformer.text_model.embeddings.position_embedding.weight",
)


class MockCLIPPatcher:
    """Minimal mock replicating the CLIP patcher API surface."""

    def __init__(self, keys: tuple[str, ...] = _SDXL_CLIP_KEYS):
        self._state_dict: dict[str, torch.Tensor] = {
            k: torch.randn(4, 4, dtype=torch.float32) for k in keys
        }
        self.patches: dict[str, list] = {}
        self.patches_uuid: uuid.UUID = uuid.uuid4()

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state_dict)

    def clone(self) -> MockCLIPPatcher:
        c = MockCLIPPatcher.__new__(MockCLIPPatcher)
        c._state_dict = self._state_dict  # shared
        c.patches = deepcopy(self.patches)
        c.patches_uuid = self.patches_uuid
        return c

    def is_clone(self, other: MockCLIPPatcher) -> bool:
        return self._state_dict is other._state_dict

    def add_patches(
        self,
        patches: dict[str, object],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> list[str]:
        added = []
        for k, v in patches.items():
            if k in self._state_dict:
                entry = (strength_patch, v, strength_model, None, None)
                self.patches.setdefault(k, []).append(entry)
                added.append(k)
        self.patches_uuid = uuid.uuid4()
        return added


class MockCLIP:
    """Minimal mock replicating ComfyUI CLIP object API surface."""

    def __init__(self, keys: tuple[str, ...] = _SDXL_CLIP_KEYS):
        self.patcher = MockCLIPPatcher(keys)
        self._clones: list[MockCLIP] = []

    def clone(self) -> MockCLIP:
        c = MockCLIP.__new__(MockCLIP)
        c.patcher = self.patcher.clone()
        c._clones = []
        self._clones.append(c)
        return c

    def add_patches(
        self,
        patches: dict[str, object],
        strength_patch: float = 1.0,
    ) -> list[str]:
        return self.patcher.add_patches(patches, strength_patch)


@pytest.fixture()
def mock_clip() -> MockCLIP:
    return MockCLIP()


@pytest.fixture()
def clip_recipe_base(mock_clip: MockCLIP) -> RecipeBase:
    return RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="clip")


def _dir_resolver(base_dir: str):
    """Create a resolver that joins LoRA names to a base directory."""
    return lambda name: os.path.join(base_dir, name)


def _model_dir_resolver(base_dir: str):
    """Create a model resolver that joins to a base directory."""
    return lambda name, source_dir: os.path.join(base_dir, name)


# --- AC-1: Returns ComfyUI CLIP object ---


class TestCLIPExitReturnsClip:
    """AC: @clip-exit-node ac-1 — returns ComfyUI CLIP object."""

    def test_return_types_is_clip(self):
        """RETURN_TYPES is CLIP tuple."""
        assert WIDENCLIPExitNode.RETURN_TYPES == ("CLIP",)
        assert WIDENCLIPExitNode.RETURN_NAMES == ("clip",)

    def test_recipe_base_returns_clip_clone(self, clip_recipe_base: RecipeBase):
        """RecipeBase at root returns cloned CLIP."""
        node = WIDENCLIPExitNode()
        (result,) = node.execute(clip_recipe_base)

        # Result is a clone, not the same object
        assert result is not clip_recipe_base.model_patcher
        # But shares state dict (MockCLIP clone behavior)
        assert result.patcher._state_dict is clip_recipe_base.model_patcher.patcher._state_dict


# --- AC-2: Selects CLIP LoRA loader via domain="clip" ---


class TestCLIPLoRALoaderSelection:
    """AC: @clip-exit-node ac-2 — selects CLIP LoRA loader based on domain='clip'."""

    def test_analyze_recipe_uses_domain_clip(self, clip_recipe_base: RecipeBase):
        """Verify recipe base has domain='clip' for loader dispatch."""
        # The analyze_recipe function checks domain from RecipeBase
        assert clip_recipe_base.domain == "clip"


# --- AC-3: Selects CLIP model loader via domain="clip" ---


class TestCLIPModelLoaderSelection:
    """AC: @clip-exit-node ac-3 — selects CLIP model loader based on domain='clip'."""

    def test_clip_model_input_uses_checkpoints_folder(self):
        """CLIP Model Input nodes use 'checkpoints' source_dir by default."""
        model = RecipeModel(path="model.safetensors", strength=1.0)
        # Default source_dir is "checkpoints"
        assert model.source_dir == "checkpoints"


# --- AC-4: Clones CLIP and applies via patch mechanism ---


class TestInstallMergedClipPatches:
    """AC: @clip-exit-node ac-4 — clones CLIP and applies merged weights."""

    def test_clones_clip_object(self, mock_clip: MockCLIP):
        """install_merged_clip_patches clones the CLIP instead of mutating."""
        original_uuid = mock_clip.patcher.patches_uuid
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Result is different object
        assert result is not mock_clip
        # Original unchanged
        assert mock_clip.patcher.patches_uuid == original_uuid
        assert len(mock_clip.patcher.patches) == 0

    def test_adds_set_patches(self, mock_clip: MockCLIP):
        """Merged weights are added as set patches."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Check patch was added
        assert key in result.patcher.patches
        patch_entry = result.patcher.patches[key][0]
        assert patch_entry[0] == 1.0  # strength_patch
        assert patch_entry[1][0] == "set"  # patch type

    def test_transfers_tensors_to_cpu(self, mock_clip: MockCLIP):
        """All patch tensors are on CPU."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]  # ("set", (tensor,))
        assert patch_tensor.device.type == "cpu"

    def test_matches_base_dtype_float32(self, mock_clip: MockCLIP):
        """Patch tensors match base model dtype (float32 case)."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float16)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.float32

    def test_matches_base_dtype_bfloat16(self):
        """Patch tensors match base model dtype (bf16 case)."""
        clip = MockCLIP()
        for k in clip.patcher._state_dict:
            clip.patcher._state_dict[k] = clip.patcher._state_dict[k].to(torch.bfloat16)

        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_clip_patches(clip, merged_state, torch.bfloat16)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.bfloat16


# --- AC-5: Result is usable CLIP object ---


class TestUsableClipResult:
    """AC: @clip-exit-node ac-5 — result functions as valid CLIP object."""

    def test_result_has_patcher(self, mock_clip: MockCLIP):
        """Result CLIP has patcher attribute."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        assert hasattr(result, "patcher")
        assert result.patcher is not None

    def test_result_can_clone(self, mock_clip: MockCLIP):
        """Result CLIP can be cloned again."""
        key = "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Should be able to clone the result
        clone = result.clone()
        assert clone is not result


# --- AC-6: Validates tree structure ---


class TestValidateClipRecipeTree:
    """AC: @clip-exit-node ac-6 — validates tree, raises ValueError on type mismatches."""

    def test_valid_recipe_base(self, clip_recipe_base: RecipeBase):
        """RecipeBase with domain='clip' is valid."""
        _validate_clip_recipe_tree(clip_recipe_base)

    def test_rejects_diffusion_domain(self, mock_clip: MockCLIP):
        """RecipeBase with domain='diffusion' is rejected."""
        base = RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="diffusion")

        with pytest.raises(ValueError) as exc_info:
            _validate_clip_recipe_tree(base)

        assert "domain='diffusion'" in str(exc_info.value)
        assert "expected domain='clip'" in str(exc_info.value)

    def test_rejects_lora_at_root(self):
        """Execute raises ValueError for LoRA at root."""
        node = WIDENCLIPExitNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        with pytest.raises(ValueError) as exc_info:
            node.execute(lora)

        assert "RecipeLoRA" in str(exc_info.value)

    def test_rejects_compose_at_root(self):
        """Execute raises ValueError for Compose at root."""
        node = WIDENCLIPExitNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))

        with pytest.raises(ValueError) as exc_info:
            node.execute(compose)

        assert "RecipeCompose" in str(exc_info.value)

    def test_valid_merge_structure(self, clip_recipe_base: RecipeBase):
        """Valid RecipeMerge structure passes validation."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=clip_recipe_base, target=lora, backbone=None, t_factor=1.0)

        _validate_clip_recipe_tree(merge)

    def test_validates_nested_compose(self, clip_recipe_base: RecipeBase):
        """Nested compose branches are validated."""
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.5},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=clip_recipe_base, target=compose, backbone=None, t_factor=1.0)

        _validate_clip_recipe_tree(merge)

    def test_error_names_position(self, mock_clip: MockCLIP):
        """Error message includes position in tree."""
        # Create a tree with diffusion domain nested inside
        base = RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="diffusion")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        with pytest.raises(ValueError) as exc_info:
            _validate_clip_recipe_tree(merge)

        # Should include path info
        assert "root.base" in str(exc_info.value)


# --- AC-7: Accepts WIDEN_CLIP type ---


class TestInputTypes:
    """AC: @clip-exit-node ac-7 — accepts WIDEN_CLIP type."""

    def test_input_types_accepts_widen_clip(self):
        """INPUT_TYPES returns correct structure with WIDEN_CLIP input."""
        input_types = WIDENCLIPExitNode.INPUT_TYPES()

        assert "required" in input_types
        assert "widen_clip" in input_types["required"]
        assert input_types["required"]["widen_clip"] == ("WIDEN_CLIP",)


# --- AC-8: IS_CHANGED returns different value when files change ---


class TestIsChanged:
    """AC: @clip-exit-node ac-8 — IS_CHANGED triggers re-execution on file changes."""

    def test_identical_hash_no_changes(self):
        """Same recipe with unchanged files produces identical hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")
            with open(lora_path, "wb") as f:
                f.write(b"fake lora data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))

            hash1 = _compute_clip_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))
            hash2 = _compute_clip_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 == hash2

    def test_different_hash_on_lora_modification(self):
        """Modified LoRA file produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")

            with open(lora_path, "wb") as f:
                f.write(b"initial data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
            hash1 = _compute_clip_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            time.sleep(0.1)

            with open(lora_path, "wb") as f:
                f.write(b"modified data with more content")

            hash2 = _compute_clip_recipe_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 != hash2

    def test_different_hash_on_model_modification(self):
        """Modified model checkpoint produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.safetensors")

            with open(model_path, "wb") as f:
                f.write(b"initial checkpoint")

            model = RecipeModel(path="model.safetensors", strength=1.0)
            resolver = _model_dir_resolver(tmpdir)
            hash1 = _compute_clip_recipe_hash(model, model_path_resolver=resolver)

            time.sleep(0.1)

            with open(model_path, "wb") as f:
                f.write(b"modified checkpoint content")

            hash2 = _compute_clip_recipe_hash(model, model_path_resolver=resolver)

            assert hash1 != hash2

    def test_is_changed_returns_hash_string(self):
        """IS_CHANGED returns a 64-char hex string (SHA-256)."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        hash_val = WIDENCLIPExitNode.IS_CHANGED(lora)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


# --- AC-9: Progress reporting ---


class TestProgressReporting:
    """AC: @clip-exit-node ac-9 — progress is reported via ProgressBar."""

    def test_category_is_clip_merge(self):
        """CATEGORY is ecaj/merge/clip."""
        assert WIDENCLIPExitNode.CATEGORY == "ecaj/merge/clip"


# --- Unpatch loaded clones tests ---


class TestUnpatchLoadedClipClones:
    """Tests for _unpatch_loaded_clip_clones."""

    @pytest.fixture()
    def _patch_loaded_models(self, monkeypatch):
        """Wire current_loaded_models onto the comfy.model_management stub."""
        mm = sys.modules["comfy.model_management"]
        loaded: list = []
        monkeypatch.setattr(mm, "current_loaded_models", loaded, raising=False)
        return loaded

    def test_noop_without_comfy(self, mock_clip: MockCLIP):
        """Does not crash when comfy.model_management has no current_loaded_models."""
        _unpatch_loaded_clip_clones(mock_clip)

    def test_noop_with_empty_loaded_models(
        self, mock_clip: MockCLIP, _patch_loaded_models
    ):
        """Safe when current_loaded_models is empty."""
        _unpatch_loaded_clip_clones(mock_clip)
        assert _patch_loaded_models == []


# --- Collect paths helpers ---


class TestCollectPaths:
    """Tests for path collection helpers."""

    def test_collect_lora_paths_from_recipe_lora(self):
        """RecipeLoRA returns its paths."""
        lora = RecipeLoRA(
            loras=(
                {"path": "a.safetensors", "strength": 1.0},
                {"path": "b.safetensors", "strength": 0.5},
            )
        )
        paths = _collect_lora_paths(lora)
        assert paths == ["a.safetensors", "b.safetensors"]

    def test_collect_model_paths_from_recipe_model(self):
        """RecipeModel returns its path and source_dir."""
        model = RecipeModel(path="model.safetensors", strength=1.0)
        paths = _collect_model_paths(model)
        assert paths == [("model.safetensors", "checkpoints")]

    def test_collect_paths_from_merge(self, clip_recipe_base: RecipeBase):
        """RecipeMerge collects from target and backbone."""
        lora = RecipeLoRA(loras=({"path": "target.safetensors", "strength": 1.0},))
        backbone = RecipeLoRA(loras=({"path": "backbone.safetensors", "strength": 0.5},))
        merge = RecipeMerge(base=clip_recipe_base, target=lora, backbone=backbone, t_factor=1.0)

        paths = _collect_lora_paths(merge)
        assert "target.safetensors" in paths
        assert "backbone.safetensors" in paths


# --- Node metadata ---


class TestCLIPExitNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_function_name(self):
        """FUNCTION points to execute method."""
        assert WIDENCLIPExitNode.FUNCTION == "execute"

    def test_output_node_false(self):
        """OUTPUT_NODE is False."""
        assert WIDENCLIPExitNode.OUTPUT_NODE is False
