"""Source-real CPU regressions for the loader audit."""

import pytest
import torch
from safetensors.torch import save_file

from lib.architecture import detect_supported_architecture
from lib.clip_model_loader import CLIPModelLoader
from lib.gpu_ops import apply_lora_batch_gpu
from lib.lora.flux import FluxLoader
from lib.lora.krea2 import Krea2Loader
from lib.lora.qwen import QwenLoader
from lib.lora.sdxl import SDXLLoader
from lib.lora.sdxl_clip import SDXLCLIPLoader
from lib.lora.zimage import ZImageLoader
from lib.model_loader import ModelLoader


# AC: @lora-loaders ac-4
@pytest.mark.parametrize(
    "cls,stem",
    [
        (SDXLLoader, "lora_unet_input_blocks_1_0"),
        (Krea2Loader, "diffusion_model.blocks.0.attn.wq"),
        (QwenLoader, "transformer.transformer_blocks.0.attn.to_q"),
        (ZImageLoader, "transformer.layers.0.attention.to_q"),
        (FluxLoader, "transformer.double_blocks.0.img_attn.to_q"),
        (SDXLCLIPLoader, "lora_te1_text_model_encoder_layers_0_self_attn_q_proj"),
    ],
)
@pytest.mark.parametrize("bad", ["orphan", "lokr", "nan", "integer", "rank"])
def test_invalid_package_is_atomic(tmp_path, cls, stem, bad):
    data = {
        stem + ".lora_up.weight": torch.ones(4, 2),
        stem + ".lora_down.weight": torch.ones(2, 4),
    }
    if bad == "orphan":
        data[stem + "_bad.lora_down.weight"] = torch.ones(2, 4)
    elif bad == "lokr":
        data[stem + ".lokr_w1"] = torch.ones(2, 2)
    elif bad == "nan":
        data[stem + ".alpha"] = torch.tensor(float("nan"))
    elif bad == "integer":
        data[stem + ".lora_up.weight"] = torch.ones(4, 2, dtype=torch.int64)
    else:
        data[stem + ".lora_up.weight"] = torch.ones(4, 3)
    path = tmp_path / "bad.safetensors"
    save_file(data, path)
    loader = cls()
    with pytest.raises(ValueError):
        loader.load(str(path))
    assert loader.loaded_bytes == 0
    assert not loader.affected_keys


# AC: @lora-loaders ac-1
@pytest.mark.parametrize(
    "cls,stem,key",
    [
        (
            SDXLLoader,
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj",
            "diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight",
        ),
        (
            SDXLLoader,
            "unet.up_blocks.0.resnets.0.conv1",
            "diffusion_model.output_blocks.0.0.in_layers.2.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.attn.add_k_proj",
            "diffusion_model.double_blocks.0.txt_attn.qkv.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.attn.to_q",
            "diffusion_model.double_blocks.0.img_attn.qkv.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.attn.to_out.0",
            "diffusion_model.double_blocks.0.img_attn.proj.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.attn.to_add_out",
            "diffusion_model.double_blocks.0.txt_attn.proj.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.ff.net.0.proj",
            "diffusion_model.double_blocks.0.img_mlp.0.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.ff_context.net.2",
            "diffusion_model.double_blocks.0.txt_mlp.2.weight",
        ),
        (
            FluxLoader,
            "transformer.transformer_blocks.0.norm1_context.linear",
            "diffusion_model.double_blocks.0.txt_mod.lin.weight",
        ),
        (
            FluxLoader,
            "transformer.single_transformer_blocks.0.attn.to_v",
            "diffusion_model.single_blocks.0.linear1.weight",
        ),
        (
            FluxLoader,
            "transformer.single_transformer_blocks.0.norm.linear",
            "diffusion_model.single_blocks.0.modulation.lin.weight",
        ),
        (
            FluxLoader,
            "transformer.single_transformer_blocks.0.proj_out",
            "diffusion_model.single_blocks.0.linear2.weight",
        ),
        (
            FluxLoader,
            "transformer.single_transformer_blocks.0.proj_mlp",
            "diffusion_model.single_blocks.0.linear1.weight",
        ),
        (
            ZImageLoader,
            "lycoris_layers_0_feed_forward_linear_1",
            "diffusion_model.layers.0.feed_forward.linear_1.weight",
        ),
        (
            QwenLoader,
            "lora_unet_transformer_blocks_0_attn_add_q_proj",
            "diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
        ),
    ],
)
def test_source_real_module_delta(tmp_path, cls, stem, key):
    path = tmp_path / "mapped.safetensors"
    up, down = torch.arange(8.0).reshape(4, 2), torch.arange(8.0).reshape(2, 4)
    save_file({stem + ".lora_up.weight": up, stem + ".lora_down.weight": down}, path)
    loader = cls()
    loader.load(str(path))
    assert loader.affected_keys == {key}
    (spec,) = loader.get_delta_specs([key], {key: 0})
    torch.testing.assert_close(spec.scale * (spec.up @ spec.down), up @ down)
    if spec.kind != "offset_mlp":  # Dispatch fix belongs to the gpu_ops lane.
        shape = (12, 4) if spec.offset else (4, 4)
        if spec.offset and "single_blocks" in key:
            shape = (28, 4)
        loader.validate_compatible_keys({key}, {key: shape})
        actual = apply_lora_batch_gpu([key], torch.zeros(1, *shape), [spec], "cpu", torch.float32)
        expected = torch.zeros(shape)
        if spec.offset:
            start, length = spec.offset
            expected[start : start + length] = up @ down
        else:
            expected = up @ down
        torch.testing.assert_close(actual[0], expected)


# AC: @full-model-loader ac-2
@pytest.mark.parametrize(
    "arch,roots",
    [
        ("flux", ["double_blocks.0.img_attn.qkv", "single_blocks.0.linear1"]),
        ("zimage", ["layers.0.attention.out", "noise_refiner.0.attention.qkv"]),
        (
            "qwen",
            [
                "transformer_blocks.0.attn.add_q_proj",
                "transformer_blocks.0.attn.to_q",
                "txt_norm",
                "img_in",
            ],
        ),
    ],
)
def test_bare_native_checkpoint(tmp_path, arch, roots):
    data = {k + ".weight": torch.full((2, 3), float(i + 1)) for i, k in enumerate(roots)}
    data["unrelated.weight"] = torch.zeros(2, 3)
    path = tmp_path / "native.safetensors"
    save_file(data, path)
    with ModelLoader(str(path)) as loader:
        assert loader.arch == arch
        assert len(loader.affected_keys) == len(roots)
        for root in roots:
            torch.testing.assert_close(
                loader.get_weights(["diffusion_model." + root + ".weight"])[0],
                data[root + ".weight"],
            )


# AC: @full-model-loader ac-7
@pytest.mark.parametrize("collision", [False, True])
def test_checkpoint_rejects_unusable_or_aliases(tmp_path, collision):
    data = {"unrelated.weight": torch.ones(2, 3)}
    if collision:
        data = {
            "diffusion_model.layers.0.attention.out.weight": torch.ones(2, 3),
            "model.diffusion_model.layers.0.attention.out.weight": torch.zeros(2, 3),
        }
    path = tmp_path / "bad.safetensors"
    save_file(data, path)
    with pytest.raises(ValueError):
        ModelLoader(str(path))


# AC: @lora-loaders ac-4
def test_analysis_cleans_partial_acquisition(tmp_path, monkeypatch):
    from lib import analysis
    from lib.recipe import RecipeBase, RecipeLoRA, RecipeMerge

    path = tmp_path / "good.safetensors"
    save_file(
        {
            "lora_unet_input_blocks_1_0.lora_up.weight": torch.ones(4, 2),
            "lora_unet_input_blocks_1_0.lora_down.weight": torch.ones(2, 3),
        },
        path,
    )
    loader = SDXLLoader()
    monkeypatch.setattr(analysis, "get_loader", lambda *args: loader)
    target = RecipeLoRA(
        loras=(
            {"path": str(path), "strength": 1.0},
            {"path": str(tmp_path / "missing.safetensors"), "strength": 1.0},
        )
    )
    recipe = RecipeMerge(
        base=RecipeBase(model_patcher=object(), arch="sdxl"),
        target=target,
        backbone=None,
        t_factor=1.0,
    )
    with pytest.raises(FileNotFoundError):
        analysis.analyze_recipe(recipe)
    assert loader.loaded_bytes == 0
    assert not loader.affected_keys


# AC: @recipe-domain-field ac-3
def test_canonical_layer_and_clip_domain():
    from lib.block_classify import classify_layer_type, filter_changed_keys

    assert (
        classify_layer_type("diffusion_model.layers.0.attention.qkv.weight", "zimage")
        == "attention"
    )
    key = "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    assert filter_changed_keys({key}, set(), {"feed_forward"}, "sdxl", "clip") == {key}


# AC: @lora-loaders ac-2
@pytest.mark.parametrize("bad", ["duplicate", "nonfinite_factor", "strength", "spatial_up"])
def test_additional_invalid_pair_inputs(tmp_path, bad):
    stem = "lora_unet_input_blocks_1_0"
    data = {
        stem + ".lora_up.weight": torch.ones(4, 2),
        stem + ".lora_down.weight": torch.ones(2, 3),
    }
    if bad == "duplicate":
        data[stem + ".lora_B.weight"] = torch.zeros(4, 2)
    elif bad == "nonfinite_factor":
        data[stem + ".lora_up.weight"][0, 0] = float("inf")
    elif bad == "spatial_up":
        data = {
            stem + ".lora_up.weight": torch.ones(4, 2, 3, 3),
            stem + ".lora_down.weight": torch.ones(2, 3, 1, 1),
        }
    path = tmp_path / "invalid.safetensors"
    save_file(data, path)
    loader = SDXLLoader()
    with pytest.raises(ValueError):
        loader.load(str(path), strength=float("inf") if bad == "strength" else 1.0)
    assert not loader.affected_keys
    assert loader.loaded_bytes == 0


# AC: @lora-loaders ac-2
def test_reconstructed_target_validation(tmp_path):
    stem = "transformer.single_transformer_blocks.0.proj_mlp"
    path = tmp_path / "slice.safetensors"
    save_file(
        {
            stem + ".lora_up.weight": torch.ones(16, 2),
            stem + ".lora_down.weight": torch.ones(2, 4),
        },
        path,
    )
    loader = FluxLoader()
    loader.load(str(path))
    key = "diffusion_model.single_blocks.0.linear1.weight"
    loader.validate_compatible_keys({key}, {key: (28, 4)})
    with pytest.raises(ValueError, match="slice outside"):
        loader.validate_compatible_keys({key}, {key: (27, 4)})
    with pytest.raises(ValueError, match="Unmapped"):
        loader.validate_compatible_keys(set())


# AC: @lora-loaders ac-1
def test_qwen_native_mlp_roundtrip_and_control(tmp_path):
    from lib.block_classify import classify_layer_type

    stem = "lycoris_transformer_blocks_0_img_mlp_net_0_proj"
    key = "diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight"
    path = tmp_path / "mlp.safetensors"
    save_file(
        {stem + ".lora_up.weight": torch.ones(4, 2), stem + ".lora_down.weight": torch.ones(2, 3)},
        path,
    )
    loader = QwenLoader()
    loader.load(str(path))
    assert loader.affected_keys == {key}
    (spec,) = loader.get_delta_specs([key], {key: 0})
    actual = apply_lora_batch_gpu([key], torch.zeros(1, 4, 3), [spec], "cpu", torch.float32)
    torch.testing.assert_close(actual, torch.full((1, 4, 3), 2.0))
    assert classify_layer_type(key, "qwen") == "feed_forward"


# AC: @lora-loaders ac-2
def test_flux_diffusers_output_modulation_permutation(tmp_path):
    stem = "transformer.norm_out.linear"
    path = tmp_path / "mod.safetensors"
    up, down = torch.arange(16.0).reshape(8, 2), torch.arange(8.0).reshape(2, 4)
    save_file({stem + ".lora_up.weight": up, stem + ".lora_down.weight": down}, path)
    loader = FluxLoader()
    loader.load(str(path))
    key = "diffusion_model.final_layer.adaLN_modulation.1.weight"
    (spec,) = loader.get_delta_specs([key], {key: 0})
    actual = apply_lora_batch_gpu([key], torch.zeros(1, 8, 4), [spec], "cpu", torch.float32)
    raw = up @ down
    torch.testing.assert_close(actual[0], torch.cat([raw[4:], raw[:4]]))


# AC: @full-model-loader ac-8
def test_sdxl_nested_transformers_are_not_qwen():
    keys = {
        f"diffusion_model.middle_block.1.transformer_blocks.{i}.attn{a}.to_{p}.weight"
        for i in range(10)
        for a in (1, 2)
        for p in "qkv"
    }
    keys.update(
        {"diffusion_model.input_blocks.0.0.weight", "diffusion_model.output_blocks.0.0.weight"}
    )
    assert detect_supported_architecture(keys) == "sdxl"


# AC: @clip-model-loader ac-2
def test_real_openclip_g_lazy_slices_and_transpose(tmp_path):
    prefix = "conditioner.embedders.1.model."
    qkv = torch.arange(24.0).reshape(6, 4)
    bias = torch.arange(6.0)
    projection = torch.arange(12.0).reshape(3, 4)
    path = tmp_path / "clip.safetensors"
    save_file(
        {
            prefix + "transformer.resblocks.0.attn.in_proj_weight": qkv,
            prefix + "transformer.resblocks.0.attn.in_proj_bias": bias,
            prefix + "text_projection": projection,
        },
        path,
    )
    with CLIPModelLoader(str(path)) as loader:
        root = "clip_g.transformer.text_model.encoder.layers.0.self_attn."
        for i, p in enumerate("qkv"):
            w, b = loader.get_weights([root + p + "_proj.weight", root + p + "_proj.bias"])
            torch.testing.assert_close(w, qkv[i * 2 : (i + 1) * 2])
            torch.testing.assert_close(b, bias[i * 2 : (i + 1) * 2])
        torch.testing.assert_close(
            loader.get_weights(["clip_g.transformer.text_projection.weight"])[0], projection.T
        )
        assert loader.loaded_bytes == 0


# AC: @lora-loaders ac-2
def test_sdxl_conv_kernel_and_ff(tmp_path):
    stem = "lora_unet_input_blocks_1_0_in_layers_2"
    up = torch.arange(8.0).reshape(4, 2, 1, 1)
    down = torch.arange(54.0).reshape(2, 3, 3, 3)
    ff = "lora_unet_middle_block_1_transformer_blocks_0_ff_net_0_proj"
    path = tmp_path / "conv.safetensors"
    save_file(
        {
            stem + ".lora_up.weight": up,
            stem + ".lora_down.weight": down,
            ff + ".lora_up.weight": torch.ones(4, 2),
            ff + ".lora_down.weight": torch.ones(2, 3),
        },
        path,
    )
    loader = SDXLLoader()
    loader.load(str(path))
    key = "diffusion_model.input_blocks.1.0.in_layers.2.weight"
    spec = loader.get_delta_specs([key], {key: 0})[0]
    assert spec.target_shape == (4, 3, 3, 3)
    applied = apply_lora_batch_gpu([key], torch.zeros(1, 4, 3, 3, 3), [spec], "cpu", torch.float32)
    torch.testing.assert_close(applied[0], (up[:, :, 0, 0] @ down.flatten(1)).reshape(4, 3, 3, 3))
    torch.testing.assert_close(
        (spec.up @ spec.down).reshape(spec.target_shape),
        (up[:, :, 0, 0] @ down.flatten(1)).reshape(4, 3, 3, 3),
    )
    assert (
        "diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight"
        in loader.affected_keys
    )


# AC: @lora-loaders ac-2
def test_flux_expanded_mlp_offset(tmp_path):
    stem = "transformer.single_blocks.0.proj_mlp"
    path = tmp_path / "flux.safetensors"
    save_file(
        {stem + ".lora_B.weight": torch.ones(16, 2), stem + ".lora_A.weight": torch.ones(2, 4)},
        path,
    )
    loader = FluxLoader()
    loader.load(str(path))
    key = "diffusion_model.single_blocks.0.linear1.weight"
    spec = loader.get_delta_specs([key], {key: 0})[0]
    assert spec.offset == (12, 16)
    out = torch.zeros(28, 4)
    start, length = spec.offset
    out[start : start + length] += spec.scale * (spec.up @ spec.down)
    torch.testing.assert_close(out[:12], torch.zeros(12, 4))
    torch.testing.assert_close(out[12:], torch.full((16, 4), 2.0))
