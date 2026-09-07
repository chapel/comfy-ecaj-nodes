"""Source-real CPU regressions for the loader audit."""

import torch
from safetensors.torch import save_file

from lib.architecture import detect_supported_architecture
from lib.clip_model_loader import CLIPModelLoader
from lib.lora.flux import FluxLoader
from lib.lora.sdxl import SDXLLoader


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
