"""LR-03/04/05: source-grounded mappings and Flux slice isolation, CPU only."""

import ast
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.gpu_ops import apply_lora_batch_gpu
from lib.lora.flux import FluxLoader, _parse_flux_lora_key
from lib.lora.zimage import ZImageLoader, _parse_zimage_lora_key


def load_pair(tmp_path, cls, stem, rows=4):
    up = torch.arange(1.0, rows * 2 + 1).reshape(rows, 2)
    down = torch.arange(1.0, 9).reshape(2, 4)
    path = tmp_path / "pair.safetensors"
    save_file({stem + ".lora_up.weight": up, stem + ".lora_down.weight": down}, path)
    loader = cls()
    loader.load(str(path))
    return loader, up @ down


def apply_checked(loader, key, shape):
    loader.validate_compatible_keys({key}, {key: shape})
    specs = loader.get_delta_specs([key], {key: 0})
    return apply_lora_batch_gpu([key], torch.zeros(1, *shape), specs, "cpu", torch.float32)[0]


FLUX_GLOBALS = [
    (f"time_text_embed.{source}_embedder.linear_{index}", f"{target}_in.{layer}_layer")
    for source, target in [("timestep", "time"), ("text", "vector"), ("guidance", "guidance")]
    for index, layer in [(1, "in"), (2, "out")]
] + [
    ("final_layer.linear", "final_layer.linear"),
    ("final_layer.adaLN_modulation.1", "final_layer.adaLN_modulation.1"),
]
ZIMAGE_GLOBALS = [
    ("all_x_embedder.2-1", "x_embedder"),
    ("all_final_layer.2-1.linear", "final_layer.linear"),
    ("all_final_layer.2-1.adaLN_modulation.1", "final_layer.adaLN_modulation.1"),
]


# AC: @flux-klein-support ac-6
# AC: @flux-klein-support ac-7
@pytest.mark.parametrize("source,target", FLUX_GLOBALS)
@pytest.mark.parametrize("prefix", ["transformer.", "lora_unet_", "lycoris_"])
def test_flux_global_numerical(tmp_path, source, target, prefix):
    stem = prefix + (source if prefix.endswith(".") else source.replace(".", "_"))
    loader, delta = load_pair(tmp_path, FluxLoader, stem)
    key = f"diffusion_model.{target}.weight"
    assert loader.affected_keys == {key}
    torch.testing.assert_close(apply_checked(loader, key, (4, 4)), delta)


# AC: @zimage-loader ac-2
@pytest.mark.parametrize(
    "source,target",
    ZIMAGE_GLOBALS
    + [
        (f"layers.0.feed_forward.{name}", f"layers.0.feed_forward.{name}")
        for name in ["w1", "w2", "w3", "linear_1", "linear_2", "linear_3"]
    ],
)
@pytest.mark.parametrize("prefix", ["transformer.", "lycoris_"])
def test_zimage_global_and_preserved_ff_numerical(tmp_path, source, target, prefix):
    stem = prefix + (source if prefix.endswith(".") else source.replace(".", "_"))
    loader, delta = load_pair(tmp_path, ZImageLoader, stem)
    key = f"diffusion_model.{target}.weight"
    assert loader.affected_keys == {key}
    torch.testing.assert_close(apply_checked(loader, key, (4, 4)), delta)


FLUX_SLICES = [
    ("transformer_blocks.0.attn.to_", "double_blocks.0.img_attn.qkv", 12),
    ("transformer_blocks.0.attn.add_", "double_blocks.0.txt_attn.qkv", 12),
    ("single_transformer_blocks.0.attn.to_", "single_blocks.0.linear1", 28),
]


def slice_stem(source, component):
    return "transformer." + source + component + ("_proj" if source.endswith("add_") else "")


# AC: @flux-klein-support ac-4
# AC: @flux-klein-support ac-5
@pytest.mark.parametrize("source,target,rows", FLUX_SLICES)
@pytest.mark.parametrize("component", list("qkv"))
def test_sparse_flux_component_isolated(tmp_path, source, target, rows, component):
    loader, delta = load_pair(tmp_path, FluxLoader, slice_stem(source, component))
    key = f"diffusion_model.{target}.weight"
    actual = apply_checked(loader, key, (rows, 4))
    expected = torch.zeros(rows, 4)
    start = "qkv".index(component) * 4
    expected[start : start + 4] = delta
    torch.testing.assert_close(actual, expected)
    if component != "q":
        assert torch.count_nonzero(actual[:4]) == 0


# AC: @flux-klein-support ac-4
# AC: @flux-klein-support ac-5
@pytest.mark.parametrize("source,target,rows", FLUX_SLICES)
@pytest.mark.parametrize("component", list("qkv"))
@pytest.mark.parametrize("bad_width", [2, 5])
def test_malformed_flux_component_rejected_before_apply(
    tmp_path, source, target, rows, component, bad_width
):
    loader, _ = load_pair(tmp_path, FluxLoader, slice_stem(source, component), bad_width)
    key = f"diffusion_model.{target}.weight"
    # Compatibility is the gate before apply; malformed factors never reach dispatch.
    with pytest.raises(ValueError):
        loader.validate_compatible_keys({key}, {key: (rows, 4)})


# AC: @flux-klein-support ac-5
@pytest.mark.parametrize("expanded", [8, 16])
def test_mlp_expanded_width_not_attention_width(tmp_path, expanded):
    loader, delta = load_pair(
        tmp_path, FluxLoader, "transformer.single_transformer_blocks.0.proj_mlp", expanded
    )
    key = "diffusion_model.single_blocks.0.linear1.weight"
    actual = apply_checked(loader, key, (12 + expanded, 4))
    torch.testing.assert_close(actual[:12], torch.zeros(12, 4))
    torch.testing.assert_close(actual[12:], delta)


@pytest.fixture(scope="module")
def comfy_mappings():
    """Execute actual installed pure functions, never import Comfy/device code."""
    root = os.environ.get("COMFYUI_SOURCE")
    if root is None:
        pytest.skip("Set COMFYUI_SOURCE to run the installed Comfy mapping oracle")
    source = Path(root) / "comfy/utils.py"
    if not source.is_file():
        pytest.skip("Set COMFYUI_SOURCE to run the installed Comfy mapping oracle")
    names = {"flux_to_diffusers", "z_image_to_diffusers", "swap_scale_shift"}
    tree = ast.parse(source.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    assert not torch.cuda.is_initialized()
    return namespace


# AC: @flux-klein-support ac-6
# AC: @flux-klein-support ac-7
# AC: @zimage-loader ac-2
@pytest.mark.parametrize("arch", ["flux", "zimage"])
def test_installed_comfy_normal_matrix_alias_inventory(comfy_mappings, arch):
    if arch == "flux":
        mapping = comfy_mappings["flux_to_diffusers"](
            {"depth": 1, "depth_single_blocks": 1, "hidden_size": 4}, "diffusion_model."
        )
        parser = _parse_flux_lora_key
        prefixes = ["transformer.", "lora_unet_", "lycoris_"]
    else:
        mapping = comfy_mappings["z_image_to_diffusers"](
            {"n_layers": 1, "n_refiner_layers": 1, "dim": 3840}, "diffusion_model."
        )
        parser = _parse_zimage_lora_key
        prefixes = ["transformer.", "lycoris_"]
    checked = []
    for alias, value in mapping.items():
        if not alias.endswith(".weight") or alias.startswith("controlnet_"):
            continue
        # Norm scale vectors are not ordinary two-factor matrices. norm*.linear
        # and adaLN modulation ARE matrices and remain in the inventory.
        if (
            "norm" in alias.lower() and ".linear." not in alias
        ) or alias == "cap_embedder.0.weight":
            continue
        target = value[0] if isinstance(value, tuple) else value
        module = alias.removesuffix(".weight")
        for prefix in prefixes:
            stem = prefix + (module if prefix.endswith(".") else module.replace(".", "_"))
            assert parser(stem + ".lora_up.weight")[0] == target, stem
            checked.append(stem)
    assert len(checked) >= (90 if arch == "flux" else 50)
    assert not torch.cuda.is_initialized()
