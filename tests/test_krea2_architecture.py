"""Tests for Krea 2 architecture detection and routing."""

import tempfile

import pytest
import torch
from safetensors.torch import save_file

from lib.analysis import analyze_recipe, analyze_recipe_models
from lib.block_classify import classify_key, classify_layer_type
from lib.lora import LOADER_REGISTRY, get_loader
from lib.lora.krea2 import Krea2Loader
from lib.model_loader import ModelLoader, _detect_architecture_from_keys
from lib.recipe import RecipeBase, RecipeMerge, RecipeModel
from nodes.entry import (
    _SUPPORTED_ARCHITECTURES,
    UnsupportedArchitectureError,
    WIDENEntryNode,
    detect_architecture,
)
from tests.conftest import MockModelPatcher

KREA2_KEYS = (
    "diffusion_model.first.weight",
    "diffusion_model.blocks.0.mod.lin",
    "diffusion_model.blocks.0.attn.wq.weight",
    "diffusion_model.blocks.0.attn.wk.weight",
    "diffusion_model.blocks.0.attn.wv.weight",
    "diffusion_model.blocks.0.attn.wo.weight",
    "diffusion_model.blocks.0.mlp.gate.weight",
    "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
    "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.weight",
    "diffusion_model.txtfusion.projector.weight",
    "diffusion_model.txtfusion.refiner_blocks.0.postnorm.scale",
    "diffusion_model.txtmlp.1.weight",
    "diffusion_model.tproj.1.weight",
    "diffusion_model.last.linear.weight",
)


def _write_krea2_checkpoint() -> str:
    tensors = {key: torch.randn(4, 4) for key in KREA2_KEYS}
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


# AC: @krea2-architecture-support ac-krea2-detected-from-krea-signature
def test_entry_detects_krea2_from_complete_krea_signature() -> None:
    patcher = MockModelPatcher(keys=KREA2_KEYS)

    assert detect_architecture(patcher) == "krea2"

    (recipe,) = WIDENEntryNode().entry(patcher)
    assert recipe.arch == "krea2"
    assert recipe.model_patcher is patcher


# AC: @krea2-architecture-support ac-krea2-detected-from-krea-signature
def test_model_loader_detects_krea2_checkpoint_keys() -> None:
    checkpoint = _write_krea2_checkpoint()

    with ModelLoader(checkpoint) as loader:
        assert loader.arch == "krea2"
        assert "diffusion_model.txtfusion.projector.weight" in loader.affected_keys
        tensors = loader.get_weights(["diffusion_model.blocks.0.attn.wq.weight"])
        assert tensors[0].shape == (4, 4)

    assert _detect_architecture_from_keys(frozenset(KREA2_KEYS)) == "krea2"


# AC: @krea2-architecture-support ac-ambiguous-or-unknown-architecture-rejected
def test_incomplete_krea2_signature_is_rejected_with_missing_evidence() -> None:
    incomplete = (
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.blocks.0.mod.lin",
        "diffusion_model.last.linear.weight",
    )
    patcher = MockModelPatcher(keys=incomplete)

    with pytest.raises(UnsupportedArchitectureError) as exc_info:
        detect_architecture(patcher)

    message = str(exc_info.value)
    assert "Could not detect model architecture" in message
    assert "krea2" in message
    assert "missing" in message
    assert "text_fusion" in message


# AC: @krea2-architecture-support ac-ambiguous-or-unknown-architecture-rejected
def test_ambiguous_krea2_and_flux_signature_is_rejected() -> None:
    mixed_keys = KREA2_KEYS + (
        "diffusion_model.double_blocks.0.img_attn.qkv.weight",
        "diffusion_model.single_blocks.0.linear1.weight",
    )
    patcher = MockModelPatcher(keys=mixed_keys)

    with pytest.raises(UnsupportedArchitectureError) as exc_info:
        detect_architecture(patcher)

    message = str(exc_info.value)
    assert "Ambiguous model architecture" in message
    assert "krea2" in message
    assert "flux" in message
    assert "Evidence" in message


# AC: @krea2-architecture-support ac-ambiguous-or-unknown-architecture-rejected
def test_unknown_signature_error_reports_available_evidence() -> None:
    patcher = MockModelPatcher(keys=("diffusion_model.unknown.weight",))

    with pytest.raises(UnsupportedArchitectureError) as exc_info:
        detect_architecture(patcher)

    message = str(exc_info.value)
    assert "Could not detect model architecture" in message
    assert "Evidence" in message
    assert "Supported architectures" in message


# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_krea2_recipe_routes_to_krea_loader_and_classifier_paths() -> None:
    assert "krea2" in _SUPPORTED_ARCHITECTURES
    assert LOADER_REGISTRY["krea2"] is Krea2Loader
    assert isinstance(get_loader("krea2"), Krea2Loader)

    assert classify_key("diffusion_model.blocks.0.attn.wq.weight", "krea2") == "B00"
    assert (
        classify_key("diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight", "krea2")
        == "TF_REF1"
    )
    assert (
        classify_layer_type("diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight", "krea2")
        == "attention"
    )
    assert (
        classify_layer_type("diffusion_model.blocks.0.mlp.gate.weight", "krea2") == "feed_forward"
    )


# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_krea2_recipe_analysis_uses_krea_loader_without_loras() -> None:
    base = RecipeBase(model_patcher=object(), arch="krea2")

    analysis = analyze_recipe(base)

    assert analysis.arch == "krea2"
    assert isinstance(analysis.loader, Krea2Loader)
    assert analysis.affected_keys == set()
    analysis.loader.cleanup()


# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_krea2_model_analysis_accepts_krea2_checkpoint() -> None:
    checkpoint = _write_krea2_checkpoint()
    base = RecipeBase(model_patcher=object(), arch="krea2")
    model = RecipeModel(path=checkpoint)
    recipe = RecipeMerge(base=base, target=model, backbone=None, t_factor=0.5)

    result = analyze_recipe_models(recipe, base_arch="krea2")

    loader = next(iter(result.model_loaders.values()))
    assert isinstance(loader, ModelLoader)
    assert loader.arch == "krea2"
    assert "diffusion_model.blocks.0.attn.wq.weight" in result.all_model_keys
    loader.cleanup()


# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_krea2_loader_routes_to_implemented_package_compatibility() -> None:
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(
            {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (torch.ones(2, 3)),
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (torch.ones(4, 2)),
            },
            f.name,
        )
        path = f.name

    loader = get_loader("krea2")
    try:
        loader.load(path, set_id="krea")
        assert loader.affected_keys == frozenset({"diffusion_model.blocks.0.attn.wq.weight"})
        assert loader.affected_keys_for_set("krea") == {"diffusion_model.blocks.0.attn.wq.weight"}
        assert (
            len(
                loader.get_delta_specs(
                    list(loader.affected_keys),
                    {
                        "diffusion_model.blocks.0.attn.wq.weight": 0,
                    },
                )
            )
            == 1
        )
        assert loader.loaded_bytes > 0
    finally:
        loader.cleanup()
