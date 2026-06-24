"""Tests for Krea 2 block and layer controls."""

from lib.block_classify import classify_key, classify_key_krea2, classify_layer_type
from lib.executor import _get_block_t_factors
from lib.recipe import BlockConfig, RecipeBase, RecipeLoRA
from nodes.block_config_krea2 import WIDENBlockConfigKrea2Node
from nodes.lora import WIDENLoRANode
from nodes.merge import WIDENMergeNode


def _default_kwargs(node_cls=WIDENBlockConfigKrea2Node) -> dict[str, float]:
    return {name: spec[1]["default"] for name, spec in node_cls.INPUT_TYPES()["required"].items()}


# AC: @krea2-block-and-layer-controls ac-main-model-regions-are-controllable
def test_krea2_main_region_override_affects_only_matching_main_block() -> None:
    keys = [
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.blocks.1.attn.wq.weight",
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
        "diffusion_model.first.weight",
    ]
    config = BlockConfig(arch="krea2", block_overrides=(("B00", 0.25),))

    groups = _get_block_t_factors(keys, config, "krea2", default_t_factor=1.0)

    assert groups == {0.25: [0], 1.0: [1, 2, 3]}
    assert classify_key_krea2(keys[0]) == "B00"
    assert classify_key_krea2(keys[1]) == "B01"


# AC: @krea2-block-and-layer-controls ac-text-fusion-regions-are-controllable
def test_krea2_text_fusion_override_affects_text_fusion_region_not_catchall() -> None:
    keys = [
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
        "diffusion_model.txtfusion.layerwise_blocks.1.attn.wq.weight",
        "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight",
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.unknown.weight",
    ]
    config = BlockConfig(arch="krea2", block_overrides=(("TF_LW0", 0.4),))

    groups = _get_block_t_factors(keys, config, "krea2", default_t_factor=1.0)

    assert groups == {0.4: [0], 1.0: [1, 2, 3, 4]}
    assert classify_key(keys[0], "krea2") == "TF_LW0"
    assert classify_key(keys[1], "krea2") == "TF_LW1"
    assert classify_key(keys[2], "krea2") == "TF_REF0"
    assert classify_key(keys[3], "krea2") == "B00"
    assert classify_key(keys[4], "krea2") is None


# AC: @krea2-block-and-layer-controls ac-layer-category-controls-are-controllable
def test_krea2_layer_categories_apply_only_to_matching_weight_categories() -> None:
    keys = [
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.blocks.0.mlp.down.weight",
        "diffusion_model.blocks.0.prenorm.scale",
        "diffusion_model.tproj.1.weight",
        "diffusion_model.blocks.0.mod.lin",
        "diffusion_model.unknown.weight",
    ]
    config = BlockConfig(
        arch="krea2",
        block_overrides=(),
        layer_type_overrides=(
            ("attention", 0.5),
            ("feed_forward", 0.25),
            ("norm", 0.75),
            ("embedding_projection", 1.5),
            ("structural", 0.0),
        ),
    )

    groups = _get_block_t_factors(keys, config, "krea2", default_t_factor=1.0)

    assert groups == {
        0.5: [0],
        0.25: [1],
        0.75: [2],
        1.5: [3],
        0.0: [4],
        1.0: [5],
    }
    assert classify_layer_type(keys[0], "krea2") == "attention"
    assert classify_layer_type(keys[1], "krea2") == "feed_forward"
    assert classify_layer_type(keys[2], "krea2") == "norm"
    assert classify_layer_type(keys[3], "krea2") == "embedding_projection"
    assert classify_layer_type(keys[4], "krea2") == "structural"
    assert classify_layer_type(keys[5], "krea2") is None


# AC: @krea2-block-and-layer-controls ac-layer-category-controls-are-controllable
def test_krea2_block_config_node_documents_supported_groups() -> None:
    inputs = WIDENBlockConfigKrea2Node.INPUT_TYPES()["required"]

    for expected in (
        "B00",
        "B27",
        "TF_LW0",
        "TF_REF1",
        "TF_PROJECTOR",
        "FIRST",
        "TPROJ",
        "LAST",
        "attention",
        "feed_forward",
        "norm",
        "embedding_projection",
        "structural",
    ):
        assert expected in inputs

    kwargs = _default_kwargs()
    kwargs.update({"B00": 0.25, "TF_LW0": 0.4, "embedding_projection": 1.5})
    (config,) = WIDENBlockConfigKrea2Node().create_config(**kwargs)

    assert config.arch == "krea2"
    assert dict(config.block_overrides)["B00"] == 0.25
    assert dict(config.block_overrides)["TF_LW0"] == 0.4
    assert dict(config.layer_type_overrides)["embedding_projection"] == 1.5


# AC: @krea2-block-and-layer-controls ac-main-model-regions-are-controllable
# AC: @krea2-block-and-layer-controls ac-text-fusion-regions-are-controllable
def test_krea2_block_config_is_accepted_by_lora_and_merge_consumers() -> None:
    kwargs = _default_kwargs()
    kwargs.update({"B00": 0.5, "TF_REF1": 1.25, "structural": 0.2})
    (config,) = WIDENBlockConfigKrea2Node().create_config(**kwargs)

    (lora,) = WIDENLoRANode().add_lora("krea-style.safetensors", 1.0, block_config=config)
    base = RecipeBase(model_patcher=object(), arch="krea2")
    target = RecipeLoRA(loras=({"path": "target.safetensors", "strength": 1.0},))
    (merge,) = WIDENMergeNode().merge(base, target, 0.7, block_config=config)

    assert lora.block_config is config
    assert merge.block_config is config
    assert merge.base is base
