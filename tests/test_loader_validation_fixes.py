"""Ordinary-format compatibility regressions, serialized through real CPU apply."""

import pytest
import torch
from safetensors.torch import save_file

from lib.gpu_ops import apply_lora_batch_gpu
from lib.lora.flux import FluxLoader
from lib.lora.krea2 import Krea2Loader
from lib.lora.qwen import QwenLoader
from lib.lora.sdxl import SDXLLoader
from lib.lora.sdxl_clip import SDXLCLIPLoader
from lib.lora.zimage import ZImageLoader

# Source-real witnesses from the independent reviewer and installed Comfy mappings.
CASES = [
    (SDXLLoader, "lora_unet_input_blocks_1_0", "diffusion_model.input_blocks.1.0.weight"),
    (
        SDXLCLIPLoader,
        "lora_te1_text_model_encoder_layers_0_mlp_fc1",
        "clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight",
    ),
    (
        FluxLoader,
        "transformer.double_blocks.0.img_attn.proj",
        "diffusion_model.double_blocks.0.img_attn.proj.weight",
    ),
    (
        ZImageLoader,
        "transformer.layers.0.attention.out",
        "diffusion_model.layers.0.attention.out.weight",
    ),
    (
        QwenLoader,
        "transformer.transformer_blocks.0.attn.to_q",
        "diffusion_model.transformer_blocks.0.attn.to_q.weight",
    ),
    (Krea2Loader, "diffusion_model.blocks.0.attn.wq", "diffusion_model.blocks.0.attn.wq.weight"),
]


def pair(stem, default=False):
    up = torch.arange(8.0).reshape(4, 2)
    down = torch.arange(8.0).reshape(2, 4)
    suffixes = (
        (".lora_B.default.weight", ".lora_A.default.weight")
        if default
        else (".lora_up.weight", ".lora_down.weight")
    )
    return {stem + suffixes[0]: up, stem + suffixes[1]: down}


def apply(loader, key, set_id):
    loader.validate_compatible_keys({key}, {key: (4, 4)})
    specs = loader.get_delta_specs([key], {key: 0}, set_id=set_id)
    assert len(specs) == 1
    return apply_lora_batch_gpu([key], torch.zeros(1, 4, 4), specs, "cpu", torch.float32)[0]


def reject_atomically(tmp_path, cls, stem, key, bad, match):
    good = tmp_path / "good.safetensors"
    save_file(pair(stem), good)
    bad_path = tmp_path / "bad.safetensors"
    save_file(bad, bad_path)
    for populated in (False, True):
        with cls() as loader:
            if populated:
                loader.load(str(good), set_id="prior")
                before = apply(loader, key, "prior").clone()
            size = loader.loaded_bytes
            keys = loader.affected_keys
            with pytest.raises(ValueError, match=match):
                loader.load(str(bad_path), set_id="bad")
            assert loader.loaded_bytes == size
            assert loader.affected_keys == keys
            assert not loader.affected_keys_for_set("bad")
            assert loader.get_delta_specs([key], {key: 0}, set_id="bad") == []
            if populated:
                torch.testing.assert_close(apply(loader, key, "prior"), before)
        assert loader.loaded_bytes == 0
        assert not loader.affected_keys


# AC: @lora-loaders ac-1
# AC: @lora-loaders ac-2
@pytest.mark.parametrize("cls,stem,key", CASES[2:5])
def test_default_factors_numerical(tmp_path, cls, stem, key):
    path = tmp_path / "default.safetensors"
    data = pair(stem, default=True)
    save_file(data, path)
    with cls() as loader:
        loader.load(str(path), set_id="test")
        assert loader.affected_keys == {key}
        torch.testing.assert_close(
            apply(loader, key, "test"), list(data.values())[0] @ list(data.values())[1]
        )
    assert not torch.cuda.is_initialized()


# AC: @lora-loaders ac-4
@pytest.mark.parametrize("cls,stem,key", CASES[2:5])
@pytest.mark.parametrize("bad", ["orphan", "duplicate-up", "duplicate-down"])
def test_default_alias_controls(tmp_path, cls, stem, key, bad):
    data = pair(stem, default=True)
    if bad == "orphan":
        del data[stem + ".lora_B.default.weight"]
    elif bad == "duplicate-up":
        data[stem + ".lora_B.weight"] = torch.ones(4, 2)
    else:
        data[stem + ".lora_A.weight"] = torch.ones(2, 4)
    reject_atomically(tmp_path, cls, stem, key, data, "Incomplete|Duplicate")


# AC: @lora-loaders ac-2
@pytest.mark.parametrize("cls,stem,key", CASES)
@pytest.mark.parametrize(
    "alpha", [None, torch.tensor(4), torch.tensor(4.0), torch.tensor([4], dtype=torch.int32)]
)
def test_scalar_alpha_numerical(tmp_path, cls, stem, key, alpha):
    data = pair(stem)
    up, down = data.values()
    if alpha is not None:
        data[stem + ".alpha"] = alpha
    path = tmp_path / "alpha.safetensors"
    save_file(data, path)
    with cls() as loader:
        loader.load(str(path), strength=0.5, set_id="test")
        expected = (up @ down) * (0.5 if alpha is None else 1.0)
        torch.testing.assert_close(apply(loader, key, "test"), expected)
    assert not torch.cuda.is_initialized()


# AC: @lora-loaders ac-4
@pytest.mark.parametrize("cls,stem,key", CASES)
@pytest.mark.parametrize(
    "alpha",
    [
        torch.ones(2),
        torch.tensor(float("nan")),
        torch.tensor(float("inf")),
        torch.tensor(float("-inf")),
        torch.tensor(True),
    ],
)
def test_invalid_alpha_atomic(tmp_path, cls, stem, key, alpha):
    data = pair(stem)
    data[stem + ".alpha"] = alpha
    reject_atomically(tmp_path, cls, stem, key, data, "(?i)alpha")


# AC: @lora-loaders ac-4
@pytest.mark.parametrize("cls,stem,key", CASES)
@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_nonfloating_factors_still_rejected(tmp_path, cls, stem, key, dtype, direction):
    data = pair(stem)
    name = stem + f".lora_{direction}.weight"
    data[name] = data[name].to(dtype)
    reject_atomically(tmp_path, cls, stem, key, data, "floating factors")


# AC: @lora-loaders ac-4
@pytest.mark.parametrize("side", ["orphan", "alpha-alias"])
def test_krea_alpha_side_group_atomic(tmp_path, side):
    cls, stem, key = CASES[-1]
    data = pair(stem)
    alpha_stem = (
        "diffusion_model.blocks.1.attn.wq"
        if side == "orphan"
        else "transformer.transformer_blocks.0.attn.to_q"
    )
    data[alpha_stem + ".alpha"] = torch.tensor(2.0)
    reject_atomically(tmp_path, cls, stem, key, data, "(?i)orphan|alias")


# AC: @lora-loaders ac-2
@pytest.mark.parametrize("alpha", [torch.tensor(17), torch.tensor(17.0)])
def test_krea_full_lokr_alpha_is_valid_but_not_scaling(tmp_path, alpha):
    cls, stem, key = CASES[-1]
    w1 = torch.arange(4.0).reshape(2, 2)
    w2 = torch.arange(4.0, 8.0).reshape(2, 2)
    path = tmp_path / "lokr.safetensors"
    save_file({stem + ".lokr_w1": w1, stem + ".lokr_w2": w2, stem + ".alpha": alpha}, path)
    with cls() as loader:
        loader.load(str(path), strength=0.5, set_id="test")
        torch.testing.assert_close(apply(loader, key, "test"), torch.kron(w1, w2) * 0.5)


# AC: @lora-loaders ac-4
def test_complex_alpha_error():
    # Safetensors cannot serialize complex; exercise the same metadata boundary directly.
    from lib.lora.validation import validate_alpha

    with pytest.raises(ValueError, match="(?i)alpha.*real"):
        validate_alpha(torch.tensor(2 + 1j), "module.alpha")
