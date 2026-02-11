"""Shared test fixtures — MockModelPatcher, recipe builders, ComfyUI API mocks."""

import sys
import uuid
from copy import deepcopy
from types import ModuleType

import pytest
import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

# ---------------------------------------------------------------------------
# MockModelPatcher — faithful stand-in for comfy.model_patcher.ModelPatcher
# ---------------------------------------------------------------------------

# Representative SDXL-like diffusion_model keys (4x4 float32 tensors)
# AC: @comfyui-mocking ac-4
_SDXL_KEYS = (
    "diffusion_model.input_blocks.0.0.weight",
    "diffusion_model.input_blocks.1.0.weight",
    "diffusion_model.middle_block.0.weight",
    "diffusion_model.output_blocks.0.0.weight",
)

# Representative Z-Image/S3-DiT keys with layers + noise_refiner pattern
# AC: @comfyui-mocking ac-4
_ZIMAGE_KEYS = (
    "diffusion_model.layers.0.attention.qkv.weight",
    "diffusion_model.layers.10.attention.qkv.weight",
    "diffusion_model.layers.25.attention.qkv.weight",
    "diffusion_model.noise_refiner.weight",
    "diffusion_model.context_refiner.weight",
)


_DIFFUSION_PREFIX = "diffusion_model."


class _MockDiffusionModel:
    """Stub for ModelPatcher.model.diffusion_model — provides state_dict()."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._full_state_dict = state_dict

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k.removeprefix(_DIFFUSION_PREFIX): v
            for k, v in self._full_state_dict.items()
            if k.startswith(_DIFFUSION_PREFIX)
        }


class _MockBaseModel:
    """Stub for ModelPatcher.model — holds diffusion_model."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.diffusion_model = _MockDiffusionModel(state_dict)


class MockModelPatcher:
    """Minimal mock replicating the ModelPatcher API surface used by WIDEN nodes.

    # AC: @testing-infrastructure ac-2
    4x4 float32 tensors, SDXL-like keys, implements model_state_dict,
    clone, add_patches, get_key_patches, patches_uuid, and
    model.diffusion_model state dict access.
    """

    def __init__(
        self,
        *,
        keys: tuple[str, ...] = _SDXL_KEYS,
        tensor_shape: tuple[int, ...] = (4, 4),
    ):
        self._state_dict: dict[str, torch.Tensor] = {
            k: torch.randn(tensor_shape, dtype=torch.float32) for k in keys
        }
        self.model = _MockBaseModel(self._state_dict)
        self.patches: dict[str, list] = {}
        self.patches_uuid: uuid.UUID = uuid.uuid4()

    # -- public API matching real ModelPatcher --

    def model_state_dict(self, filter_prefix: str | None = None) -> dict[str, torch.Tensor]:
        if filter_prefix is None:
            return dict(self._state_dict)
        return {k: v for k, v in self._state_dict.items() if k.startswith(filter_prefix)}

    def clone(self) -> "MockModelPatcher":
        """Shallow clone — independent patches, shared underlying tensors."""
        c = MockModelPatcher.__new__(MockModelPatcher)
        c._state_dict = self._state_dict  # shared, like real clone()
        c.model = _MockBaseModel(c._state_dict)
        c.patches = deepcopy(self.patches)
        c.patches_uuid = uuid.uuid4()
        return c

    def add_patches(
        self,
        patches: dict[str, object],
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> list[str]:
        """Register patches for keys that exist in model state dict."""
        added = []
        for k, v in patches.items():
            if k in self._state_dict:
                entry = (strength_patch, v, strength_model, None, None)
                self.patches.setdefault(k, []).append(entry)
                added.append(k)
        self.patches_uuid = uuid.uuid4()
        return added

    def get_key_patches(self, filter_prefix: str | None = None) -> dict[str, list]:
        """Return patches dict filtered by prefix, including original weight."""
        sd = self.model_state_dict(filter_prefix)
        result = {}
        for k, weight in sd.items():
            base = [(weight, lambda w: w)]
            result[k] = base + self.patches.get(k, [])
        return result


# ---------------------------------------------------------------------------
# Recipe fixtures (AC-3)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_model_patcher() -> MockModelPatcher:
    return MockModelPatcher()


@pytest.fixture()
def recipe_base(mock_model_patcher: MockModelPatcher) -> RecipeBase:
    return RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")


@pytest.fixture()
def recipe_single_lora() -> RecipeLoRA:
    return RecipeLoRA(loras=({"path": "lora_a.safetensors", "strength": 1.0},))


@pytest.fixture()
def recipe_multi_lora() -> RecipeLoRA:
    return RecipeLoRA(
        loras=(
            {"path": "lora_a.safetensors", "strength": 1.0},
            {"path": "lora_b.safetensors", "strength": 0.5},
        )
    )


@pytest.fixture()
def recipe_compose(recipe_single_lora: RecipeLoRA) -> RecipeCompose:
    lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.8},))
    return RecipeCompose(branches=(recipe_single_lora, lora_b))


@pytest.fixture()
def recipe_chain(recipe_base: RecipeBase, recipe_single_lora: RecipeLoRA) -> RecipeMerge:
    merge_a = RecipeMerge(base=recipe_base, target=recipe_single_lora, backbone=None, t_factor=1.0)
    lora_b = RecipeLoRA(loras=({"path": "lora_b.safetensors", "strength": 0.5},))
    return RecipeMerge(base=merge_a, target=lora_b, backbone=None, t_factor=0.7)


@pytest.fixture()
def recipe_full(recipe_base: RecipeBase, recipe_compose: RecipeCompose) -> RecipeMerge:
    """Full pattern: compose (2 branches) merged into chain."""
    # AC: @comfyui-mocking ac-2
    # First merge with compose target
    merge_a = RecipeMerge(base=recipe_base, target=recipe_compose, backbone=None, t_factor=0.8)
    # Chain with additional LoRA
    lora_c = RecipeLoRA(loras=({"path": "lora_c.safetensors", "strength": 0.6},))
    return RecipeMerge(base=merge_a, target=lora_c, backbone=None, t_factor=0.5)


# ---------------------------------------------------------------------------
# Architecture-specific fixtures (AC-4)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sdxl_state_dict_keys() -> tuple[str, ...]:
    """Representative SDXL state dict key patterns.

    # AC: @comfyui-mocking ac-4
    Provides input_blocks, middle_block, and output_blocks keys.
    """
    return _SDXL_KEYS


@pytest.fixture()
def zimage_state_dict_keys() -> tuple[str, ...]:
    """Representative Z-Image state dict key patterns.

    # AC: @comfyui-mocking ac-4
    Provides layers and noise_refiner/context_refiner keys.
    """
    return _ZIMAGE_KEYS


@pytest.fixture()
def mock_model_patcher_zimage() -> MockModelPatcher:
    """MockModelPatcher with Z-Image architecture keys.

    # AC: @comfyui-mocking ac-4
    """
    return MockModelPatcher(keys=_ZIMAGE_KEYS)


# ---------------------------------------------------------------------------
# ComfyUI API mocks (AC-3) — autouse so tests run without ComfyUI installed
# ---------------------------------------------------------------------------


def _make_stub_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__package__ = name
    mod.__path__ = []
    return mod


@pytest.fixture(autouse=True)
def _mock_comfyui_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject stub modules so imports like 'import folder_paths' don't fail."""
    folder_paths_mod = _make_stub_module("folder_paths")
    # Mock get_filename_list for LoRA node dropdown (AC-3 @lora-node)
    folder_paths_mod.get_filename_list = lambda folder: ["test_lora.safetensors"]

    stubs = {
        "folder_paths": folder_paths_mod,
        "comfy": _make_stub_module("comfy"),
        "comfy.utils": _make_stub_module("comfy.utils"),
        "comfy.model_management": _make_stub_module("comfy.model_management"),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
