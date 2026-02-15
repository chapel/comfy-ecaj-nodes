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

from lib.analysis import collect_lora_paths, collect_model_paths, compute_recipe_file_hash
from lib.recipe import (
    BlockConfig,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
)
from nodes.clip_exit import (
    WIDENCLIPExitNode,
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

    # AC: @clip-exit-node ac-1
    def test_return_types_is_clip(self):
        """RETURN_TYPES is CLIP tuple."""
        assert WIDENCLIPExitNode.RETURN_TYPES == ("CLIP",)
        assert WIDENCLIPExitNode.RETURN_NAMES == ("clip",)

    # AC: @clip-exit-node ac-1
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

    # AC: @clip-exit-node ac-2
    def test_analyze_recipe_dispatches_to_clip_lora_loader(
        self, clip_recipe_base: RecipeBase, tmp_path,
    ):
        """analyze_recipe dispatches to SDXL CLIP LoRA loader when domain='clip'."""
        from lib.analysis import analyze_recipe
        from lib.lora.sdxl_clip import SDXLCLIPLoader

        # Create a minimal LoRA safetensors file
        lora_path = tmp_path / "test_clip.safetensors"
        from safetensors.torch import save_file

        # SDXL CLIP LoRA key (te1 = CLIP-L)
        down_key = "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"
        up_key = "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"
        lora_tensors = {
            down_key: torch.randn(4, 768),
            up_key: torch.randn(768, 4),
        }
        save_file(lora_tensors, str(lora_path))

        lora = RecipeLoRA(loras=({"path": str(lora_path), "strength": 1.0},))
        merge = RecipeMerge(
            base=clip_recipe_base, target=lora, backbone=None, t_factor=1.0,
        )

        result = analyze_recipe(merge)
        # Loader should be SDXLCLIPLoader (not SDXLLoader)
        assert isinstance(result.loader, SDXLCLIPLoader)
        assert result.domain == "clip"
        result.loader.cleanup()

    # AC: @clip-exit-node ac-2
    def test_analyze_recipe_uses_domain_clip(self, clip_recipe_base: RecipeBase):
        """Verify recipe base has domain='clip' for loader dispatch."""
        assert clip_recipe_base.domain == "clip"


# --- AC-3: Selects CLIP model loader via domain="clip" ---


class TestCLIPModelLoaderSelection:
    """AC: @clip-exit-node ac-3 — selects CLIP model loader based on domain='clip'."""

    # AC: @clip-exit-node ac-3
    def test_analyze_recipe_models_uses_clip_model_loader(self, tmp_path):
        """analyze_recipe_models(domain='clip') creates CLIPModelLoader instances."""
        from safetensors.torch import save_file

        from lib.analysis import analyze_recipe_models
        from lib.clip_model_loader import CLIPModelLoader

        # Create a minimal checkpoint with CLIP keys
        ckpt_path = tmp_path / "model.safetensors"
        clip_key = (
            "conditioner.embedders.0.transformer.text_model"
            ".encoder.layers.0.self_attn.q_proj.weight"
        )
        tensors = {clip_key: torch.randn(768, 768)}
        save_file(tensors, str(ckpt_path))

        model = RecipeModel(path=str(ckpt_path), strength=1.0)
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="clip")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        result = analyze_recipe_models(merge, "sdxl", domain="clip")
        try:
            # Should have created CLIPModelLoader, not ModelLoader
            assert len(result.model_loaders) == 1
            loader = next(iter(result.model_loaders.values()))
            assert isinstance(loader, CLIPModelLoader)
            # Keys should be in CLIP format (clip_l.*)
            assert any(k.startswith("clip_l.") for k in result.all_model_keys)
        finally:
            for ldr in result.model_loaders.values():
                ldr.cleanup()

    # AC: @clip-exit-node ac-3
    def test_analyze_recipe_models_diffusion_uses_model_loader(self, tmp_path):
        """analyze_recipe_models(domain='diffusion') creates ModelLoader instances."""
        from safetensors.torch import save_file

        from lib.analysis import analyze_recipe_models
        from lib.model_loader import ModelLoader

        # Create a minimal checkpoint with diffusion keys
        ckpt_path = tmp_path / "model.safetensors"
        tensors = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(
                320, 4, 3, 3,
            ),
        }
        save_file(tensors, str(ckpt_path))

        model = RecipeModel(path=str(ckpt_path), strength=1.0)
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        merge = RecipeMerge(base=base, target=model, backbone=None, t_factor=1.0)

        result = analyze_recipe_models(merge, "sdxl", domain="diffusion")
        try:
            assert len(result.model_loaders) == 1
            loader = next(iter(result.model_loaders.values()))
            assert isinstance(loader, ModelLoader)
        finally:
            for ldr in result.model_loaders.values():
                ldr.cleanup()


# --- AC-4: Clones CLIP and applies via patch mechanism ---


class TestInstallMergedClipPatches:
    """AC: @clip-exit-node ac-4 — clones CLIP and applies merged weights."""

    # AC: @clip-exit-node ac-4
    def test_clones_clip_object(self, mock_clip: MockCLIP):
        """install_merged_clip_patches clones the CLIP instead of mutating."""
        original_uuid = mock_clip.patcher.patches_uuid
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Result is different object
        assert result is not mock_clip
        # Original unchanged
        assert mock_clip.patcher.patches_uuid == original_uuid
        assert len(mock_clip.patcher.patches) == 0

    # AC: @clip-exit-node ac-4
    def test_adds_set_patches(self, mock_clip: MockCLIP):
        """Merged weights are added as set patches."""
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Check patch was added
        assert key in result.patcher.patches
        patch_entry = result.patcher.patches[key][0]
        assert patch_entry[0] == 1.0  # strength_patch
        assert patch_entry[1][0] == "set"  # patch type

    # AC: @clip-exit-node ac-4
    def test_transfers_tensors_to_cpu(self, mock_clip: MockCLIP):
        """All patch tensors are on CPU."""
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]  # ("set", (tensor,))
        assert patch_tensor.device.type == "cpu"

    # AC: @clip-exit-node ac-4
    def test_matches_base_dtype_float32(self, mock_clip: MockCLIP):
        """Patch tensors match base model dtype (float32 case)."""
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4, dtype=torch.float16)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.float32

    # AC: @clip-exit-node ac-4
    def test_matches_base_dtype_bfloat16(self):
        """Patch tensors match base model dtype (bf16 case)."""
        clip = MockCLIP()
        for k in clip.patcher._state_dict:
            clip.patcher._state_dict[k] = clip.patcher._state_dict[k].to(torch.bfloat16)

        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4, dtype=torch.float32)}
        result = install_merged_clip_patches(clip, merged_state, torch.bfloat16)

        patch_entry = result.patcher.patches[key][0]
        patch_tensor = patch_entry[1][1][0]
        assert patch_tensor.dtype == torch.bfloat16


# --- AC-5: Result is usable CLIP object ---


class TestUsableClipResult:
    """AC: @clip-exit-node ac-5 — result functions as valid CLIP object."""

    # AC: @clip-exit-node ac-5
    def test_result_has_patcher_with_state_dict(self, mock_clip: MockCLIP):
        """Result CLIP has patcher with accessible state dict."""
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        assert hasattr(result, "patcher")
        assert result.patcher is not None
        # State dict is accessible through the patcher
        state = result.patcher.model_state_dict()
        assert key in state

    # AC: @clip-exit-node ac-5
    def test_result_can_clone(self, mock_clip: MockCLIP):
        """Result CLIP can be cloned again."""
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Should be able to clone the result
        clone = result.clone()
        assert clone is not result

    # AC: @clip-exit-node ac-5
    def test_result_accepts_downstream_patches(self, mock_clip: MockCLIP):
        """Result CLIP accepts additional patches (downstream LoRA compatibility).

        ComfyUI applies downstream LoRA patches additively to the merged
        result. This test verifies the result supports add_patches() after
        merge patches are already installed.
        """
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # Simulate downstream LoRA adding patches to the merged result
        downstream_patch = {key: ("set", (torch.randn(4, 4),))}
        added = result.add_patches(downstream_patch, strength_patch=0.5)

        # Downstream patch was accepted
        assert key in added
        # Both the merge patch and downstream patch are present
        assert len(result.patcher.patches[key]) == 2

    # AC: @clip-exit-node ac-5
    def test_result_preserves_all_clip_keys(self, mock_clip: MockCLIP):
        """Result CLIP preserves all original state dict keys."""
        # Merge only one key
        key = _SDXL_CLIP_KEYS[0]
        merged_state = {key: torch.randn(4, 4)}
        result = install_merged_clip_patches(mock_clip, merged_state, torch.float32)

        # All original keys are still in the state dict
        state = result.patcher.model_state_dict()
        for k in _SDXL_CLIP_KEYS:
            assert k in state, f"Missing key: {k}"


# --- AC-6: Validates tree structure ---


class TestValidateClipRecipeTree:
    """AC: @clip-exit-node ac-6 — validates tree, raises ValueError on type mismatches."""

    # AC: @clip-exit-node ac-6
    def test_valid_recipe_base(self, clip_recipe_base: RecipeBase):
        """RecipeBase with domain='clip' is valid."""
        _validate_clip_recipe_tree(clip_recipe_base)

    # AC: @clip-exit-node ac-6
    def test_rejects_diffusion_domain(self, mock_clip: MockCLIP):
        """RecipeBase with domain='diffusion' is rejected."""
        base = RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="diffusion")

        with pytest.raises(ValueError) as exc_info:
            _validate_clip_recipe_tree(base)

        assert "domain='diffusion'" in str(exc_info.value)
        assert "expected domain='clip'" in str(exc_info.value)

    # AC: @clip-exit-node ac-6
    def test_rejects_lora_at_root(self):
        """Execute raises ValueError for LoRA at root."""
        node = WIDENCLIPExitNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        with pytest.raises(ValueError) as exc_info:
            node.execute(lora)

        assert "RecipeLoRA" in str(exc_info.value)

    # AC: @clip-exit-node ac-6
    def test_rejects_compose_at_root(self):
        """Execute raises ValueError for Compose at root."""
        node = WIDENCLIPExitNode()
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora,))

        with pytest.raises(ValueError) as exc_info:
            node.execute(compose)

        assert "RecipeCompose" in str(exc_info.value)

    # AC: @clip-exit-node ac-6
    def test_valid_merge_structure(self, clip_recipe_base: RecipeBase):
        """Valid RecipeMerge structure passes validation."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=clip_recipe_base, target=lora, backbone=None, t_factor=1.0)

        _validate_clip_recipe_tree(merge)

    # AC: @clip-exit-node ac-6
    def test_validates_nested_compose(self, clip_recipe_base: RecipeBase):
        """Nested compose branches are validated."""
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.5},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=clip_recipe_base, target=compose, backbone=None, t_factor=1.0)

        _validate_clip_recipe_tree(merge)

    # AC: @clip-exit-node ac-6
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

    # AC: @clip-exit-node ac-7
    def test_input_types_accepts_widen_clip(self):
        """INPUT_TYPES returns correct structure with WIDEN_CLIP input."""
        input_types = WIDENCLIPExitNode.INPUT_TYPES()

        assert "required" in input_types
        assert "widen_clip" in input_types["required"]
        assert input_types["required"]["widen_clip"] == ("WIDEN_CLIP",)


# --- AC-8: IS_CHANGED returns different value when files change ---


class TestIsChanged:
    """AC: @clip-exit-node ac-8 — IS_CHANGED triggers re-execution on file changes."""

    # AC: @clip-exit-node ac-8
    def test_identical_hash_no_changes(self):
        """Same recipe with unchanged files produces identical hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")
            with open(lora_path, "wb") as f:
                f.write(b"fake lora data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))

            hash1 = compute_recipe_file_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))
            hash2 = compute_recipe_file_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 == hash2

    # AC: @clip-exit-node ac-8
    def test_different_hash_on_lora_modification(self):
        """Modified LoRA file produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "lora_a.safetensors")

            with open(lora_path, "wb") as f:
                f.write(b"initial data")

            lora = RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))
            hash1 = compute_recipe_file_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            time.sleep(0.1)

            with open(lora_path, "wb") as f:
                f.write(b"modified data with more content")

            hash2 = compute_recipe_file_hash(lora, lora_path_resolver=_dir_resolver(tmpdir))

            assert hash1 != hash2

    # AC: @clip-exit-node ac-8
    def test_different_hash_on_model_modification(self):
        """Modified model checkpoint produces different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.safetensors")

            with open(model_path, "wb") as f:
                f.write(b"initial checkpoint")

            model = RecipeModel(path="model.safetensors", strength=1.0)
            resolver = _model_dir_resolver(tmpdir)
            hash1 = compute_recipe_file_hash(model, model_path_resolver=resolver)

            time.sleep(0.1)

            with open(model_path, "wb") as f:
                f.write(b"modified checkpoint content")

            hash2 = compute_recipe_file_hash(model, model_path_resolver=resolver)

            assert hash1 != hash2

    # AC: @clip-exit-node ac-8
    def test_is_changed_returns_hash_string(self):
        """IS_CHANGED returns a 64-char hex string (SHA-256)."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        hash_val = WIDENCLIPExitNode.IS_CHANGED(lora)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64


# --- AC-9: Progress reporting ---


class TestProgressReporting:
    """AC: @clip-exit-node ac-9 — progress is reported via ProgressBar."""

    # AC: @clip-exit-node ac-9
    def test_execute_creates_progress_bar_and_updates_per_group(self, monkeypatch):
        """execute() creates ProgressBar(N) and calls update(1) N times for N batch groups."""
        import nodes.clip_exit as clip_exit_mod

        progress_calls = []

        class FakeProgressBar:
            def __init__(self, total):
                progress_calls.append(("init", total))

            def update(self, n):
                progress_calls.append(("update", n))

        class FakeLoader:
            def cleanup(self):
                pass

        class FakeAnalysis:
            def __init__(self):
                self.loader = FakeLoader()
                self.set_affected = {}
                self.affected_keys = set()
                self.arch = "sdxl"

        class FakeModelAnalysis:
            def __init__(self):
                self.model_affected = {}
                self.model_loaders = {}
                self.all_model_keys = set()

        # 3 batch groups → expect ProgressBar(3) + 3× update(1)
        affected = set(_SDXL_CLIP_KEYS[:5])

        class FakeSig:
            """Minimal OpSignature stub with shape attribute."""

            def __init__(self, n):
                self.shape = (4, 4)
                self._n = n

            def __hash__(self):
                return self._n

            def __eq__(self, other):
                return self._n == other._n

        fake_batch_groups = {
            FakeSig(0): [_SDXL_CLIP_KEYS[0], _SDXL_CLIP_KEYS[1]],
            FakeSig(1): [_SDXL_CLIP_KEYS[2], _SDXL_CLIP_KEYS[3]],
            FakeSig(2): [_SDXL_CLIP_KEYS[4]],
        }

        monkeypatch.setattr(clip_exit_mod, "ProgressBar", FakeProgressBar)
        monkeypatch.setattr(clip_exit_mod, "_build_lora_resolver", lambda: lambda name: name)
        monkeypatch.setattr(
            clip_exit_mod, "_build_clip_model_resolver", lambda: lambda name, sd: name,
        )
        monkeypatch.setattr(
            clip_exit_mod, "analyze_recipe", lambda *a, **kw: FakeAnalysis(),
        )
        monkeypatch.setattr(
            clip_exit_mod, "analyze_recipe_models", lambda *a, **kw: FakeModelAnalysis(),
        )
        monkeypatch.setattr(
            clip_exit_mod, "get_keys_to_process", lambda all_k, aff: affected,
        )
        monkeypatch.setattr(
            clip_exit_mod, "compile_batch_groups", lambda *a, **kw: fake_batch_groups,
        )
        monkeypatch.setattr(clip_exit_mod, "compile_plan", lambda *a, **kw: object())
        monkeypatch.setattr(clip_exit_mod, "compute_batch_size", lambda *a, **kw: 32)
        monkeypatch.setattr(
            clip_exit_mod,
            "chunked_evaluation",
            lambda **kw: {k: torch.randn(4, 4) for k in kw["keys"]},
        )

        mock_clip = MockCLIP()
        base = RecipeBase(model_patcher=mock_clip, arch="sdxl", domain="clip")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        recipe = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        node = WIDENCLIPExitNode()
        (result,) = node.execute(recipe)

        # ProgressBar(3) was created — one per batch group
        assert ("init", 3) in progress_calls
        # update(1) called once per batch group
        assert progress_calls.count(("update", 1)) == 3
        # Result is a valid CLIP clone with patches installed
        assert result is not mock_clip
        assert len(result.patcher.patches) > 0

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
        paths = collect_lora_paths(lora)
        assert paths == ["a.safetensors", "b.safetensors"]

    def test_collect_model_paths_from_recipe_model(self):
        """RecipeModel returns its path and source_dir."""
        model = RecipeModel(path="model.safetensors", strength=1.0)
        paths = collect_model_paths(model)
        assert paths == [("model.safetensors", "checkpoints")]

    def test_collect_paths_from_merge(self, clip_recipe_base: RecipeBase):
        """RecipeMerge collects from target and backbone."""
        lora = RecipeLoRA(loras=({"path": "target.safetensors", "strength": 1.0},))
        backbone = RecipeLoRA(loras=({"path": "backbone.safetensors", "strength": 0.5},))
        merge = RecipeMerge(base=clip_recipe_base, target=lora, backbone=backbone, t_factor=1.0)

        paths = collect_lora_paths(merge)
        assert "target.safetensors" in paths
        assert "backbone.safetensors" in paths


# --- AC-10: Domain propagation to per-block helpers ---


class TestDomainPropagation:
    """AC: @clip-exit-node ac-10 — domain='clip' propagates to classify_key."""

    # AC: @clip-exit-node ac-10
    def test_per_block_lora_strength_uses_clip_domain(self):
        """_apply_per_block_lora_strength classifies CLIP keys correctly with domain='clip'."""
        from lib.per_block import _apply_per_block_lora_strength

        # CLIP-L layer 5 key → block group "CL05"
        keys = [_SDXL_CLIP_KEYS[0]]  # clip_l...layers.0...
        base = torch.ones(1, 4, 4)
        lora_applied = torch.full((1, 4, 4), 2.0)  # delta = 1.0

        # Block override: CL00 → strength 0.0 (zero out the delta)
        block_config = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.0),))

        # Without domain="clip", classify_key returns None → override is ignored
        result_diffusion = _apply_per_block_lora_strength(
            keys, base, lora_applied, block_config, "sdxl", "cpu", torch.float32,
            domain="diffusion",
        )
        # delta preserved (override not matched)
        assert torch.equal(result_diffusion, lora_applied)

        # With domain="clip", classify_key returns "CL00" → override applies
        result_clip = _apply_per_block_lora_strength(
            keys, base, lora_applied, block_config, "sdxl", "cpu", torch.float32,
            domain="clip",
        )
        # delta zeroed out (strength 0.0)
        assert torch.equal(result_clip, base)

    # AC: @clip-exit-node ac-10
    def test_widen_filter_per_block_uses_clip_domain(self):
        """_apply_widen_filter_per_block passes domain to _get_block_t_factors."""
        from lib.per_block import _get_block_t_factors

        keys = [_SDXL_CLIP_KEYS[0]]  # clip_l...layers.0... → "CL00"
        block_config = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.5),))

        # Without domain="clip": key unrecognized, falls back to default t_factor
        groups_diffusion = _get_block_t_factors(keys, block_config, "sdxl", 1.0)
        assert 1.0 in groups_diffusion

        # With domain="clip": key classified as CL00, override 0.5 applied
        groups_clip = _get_block_t_factors(keys, block_config, "sdxl", 1.0, domain="clip")
        assert 0.5 in groups_clip

    # AC: @clip-exit-node ac-10
    def test_execute_plan_passes_domain_to_per_block(self, monkeypatch):
        """execute_plan(domain='clip') propagates domain to per-block calls."""
        from lib.per_block import _apply_per_block_lora_strength as orig_fn
        from lib.recipe_eval import EvalPlan, OpApplyLoRA, execute_plan

        captured_domains = []

        def tracking_per_block(*args, **kwargs):
            # domain is the 8th positional arg (index 7) or keyword
            if "domain" in kwargs:
                captured_domains.append(kwargs["domain"])
            elif len(args) > 7:
                captured_domains.append(args[7])
            return orig_fn(*args, **kwargs)

        import lib.recipe_eval as recipe_eval_mod
        monkeypatch.setattr(recipe_eval_mod, "_apply_per_block_lora_strength", tracking_per_block)

        # Build a minimal plan with an OpApplyLoRA that has a block_config
        block_config = BlockConfig(arch="sdxl", block_overrides=(("CL00", 0.5),))
        op = OpApplyLoRA(set_id="test", block_config=block_config, input_reg=0, out_reg=1)
        plan = EvalPlan(ops=(op,), result_reg=1, dead_after=((),))

        keys = [_SDXL_CLIP_KEYS[0]]
        base_batch = torch.randn(1, 4, 4)

        # Mock loader to return no-op delta specs
        class FakeLoader:
            def get_delta_specs(self, keys, key_indices, set_id):
                return []

        execute_plan(
            plan=plan, keys=keys, base_batch=base_batch,
            loader=FakeLoader(), widen=None, device="cpu", dtype=torch.float32,
            arch="sdxl", domain="clip",
        )

        assert captured_domains == ["clip"]


# --- Node metadata ---


class TestCLIPExitNodeMetadata:
    """Test ComfyUI node metadata is correct."""

    def test_function_name(self):
        """FUNCTION points to execute method."""
        assert WIDENCLIPExitNode.FUNCTION == "execute"

    def test_output_node_false(self):
        """OUTPUT_NODE is False."""
        assert WIDENCLIPExitNode.OUTPUT_NODE is False
