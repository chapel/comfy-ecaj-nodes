"""Native Krea 2 full-checkpoint normalization with tiny real CPU files."""

import pytest
import torch
from safetensors.torch import save_file

from lib import model_loader
from lib.model_loader import KeyMismatchError, ModelLoader

# Native header names, including the nine text-adapter norms from the failure.
ADAPTER_NORMS = (
    "txtmlp.0.scale",
    "txtfusion.layerwise_blocks.0.prenorm.scale",
    "txtfusion.layerwise_blocks.0.postnorm.scale",
    "txtfusion.layerwise_blocks.1.prenorm.scale",
    "txtfusion.layerwise_blocks.1.postnorm.scale",
    "txtfusion.refiner_blocks.0.prenorm.scale",
    "txtfusion.refiner_blocks.0.postnorm.scale",
    "txtfusion.refiner_blocks.1.prenorm.scale",
    "txtfusion.refiner_blocks.1.postnorm.scale",
)
NATIVE_KEYS = (
    "blocks.0.mod.lin",
    "blocks.0.attn.wq.weight",
    "blocks.0.attn.qknorm.knorm.scale",
    "blocks.0.attn.qknorm.qnorm.scale",
    "blocks.0.prenorm.scale",
    "blocks.0.postnorm.scale",
    "txtfusion.layerwise_blocks.0.attn.wq.weight",
    "txtfusion.refiner_blocks.0.mlp.down.weight",
    "txtfusion.projector.weight",
    "txtmlp.1.weight",
    "first.weight",
    "first.bias",
    "last.linear.weight",
    "last.modulation.lin",
    "last.norm.scale",
    "tmlp.0.weight",
    "tproj.1.weight",
) + ADAPTER_NORMS
EXCLUDED_KEYS = (
    "first_stage_model.encoder.weight",
    "model.first_stage_model.encoder.weight",
    "conditioner.embedders.weight",
    "model.conditioner.embedders.weight",
    "cond_stage_model.transformer.weight",
    "model.cond_stage_model.transformer.weight",
    "encoder.weight",
    "decoder.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
    "text_encoder.weight",
    "unrelated.weight",
    "blocks_extra.weight",
    "txtfusion_extra.weight",
    "optimizer.blocks.0.attn.wq.weight",
)


def _native_tensors():
    # Distinct per-key values make wrong-file lookup/order bugs observable.
    return {
        key: torch.full((2,), index + 0.25, dtype=torch.float32, device="cpu")
        for index, key in enumerate(NATIVE_KEYS)
    }


# AC: @full-model-loader ac-2
# AC: @full-model-loader ac-5
# AC: @full-model-loader ac-8
# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
@pytest.mark.parametrize(
    "prefix",
    ["", "model.diffusion_model.", "diffusion_model.", "model.transformer.", "transformer."],
)
def test_krea2_checkpoint_maps_values_and_request_order(tmp_path, prefix):
    source = _native_tensors()
    tensors = {prefix + key: value for key, value in source.items()}
    tensors.update({key: torch.full((2,), -999.0) for key in EXCLUDED_KEYS})
    path = tmp_path / "krea2.safetensors"
    save_file(tensors, str(path))

    with ModelLoader(str(path)) as loader:
        assert loader.arch == "krea2"
        assert loader.affected_keys == frozenset("diffusion_model." + key for key in source)
        requested = list(reversed(NATIVE_KEYS)) + [ADAPTER_NORMS[0]]
        actual = loader.get_weights(["diffusion_model." + key for key in requested])
        assert len(actual) == len(requested)
        for key, tensor in zip(requested, actual, strict=True):
            assert tensor.device.type == "cpu"
            assert torch.equal(tensor, source[key])


# AC: @full-model-loader ac-7
@pytest.mark.parametrize("missing", ADAPTER_NORMS)
def test_native_checkpoint_missing_norm_still_raises(tmp_path, missing):
    tensors = _native_tensors()
    del tensors[missing]
    path = tmp_path / "missing.safetensors"
    save_file(tensors, str(path))
    with ModelLoader(str(path)) as loader:
        assert loader.arch == "krea2"
        with pytest.raises(KeyMismatchError) as exc_info:
            loader.get_weights(["diffusion_model.first.weight", "diffusion_model." + missing])
        message = str(exc_info.value)
        assert "missing 1 key(s)" in message
        assert "diffusion_model." + missing in message


# AC: @full-model-loader ac-5
# AC: @full-model-loader ac-8
@pytest.mark.parametrize(
    "keys",
    [
        ("blocks.0.attn.wq.weight", "first.weight", "txtmlp.0.scale"),
        ("layers.0.attention.qkv.weight", "noise_refiner.0.attn.weight"),
        ("input_blocks.0.weight", "middle_block.0.weight", "output_blocks.0.weight"),
        ("double_blocks.0.img_attn.qkv.weight",),
    ],
)
def test_bare_keys_require_complete_krea2_signature(tmp_path, keys):
    path = tmp_path / "not-krea2.safetensors"
    save_file({key: torch.ones(2) for key in keys}, str(path))
    if keys[0].startswith("blocks."):
        with pytest.raises(KeyMismatchError, match="no usable"):
            ModelLoader(str(path))
    else:
        with ModelLoader(str(path)) as loader:
            assert loader.arch in {"zimage", "sdxl", "flux"}
            assert loader.arch != "krea2"
            assert loader.affected_keys == {"diffusion_model." + key for key in keys}
            torch.testing.assert_close(
                loader.get_weights(["diffusion_model." + keys[0]])[0], torch.ones(2)
            )


# AC: @full-model-loader ac-1
# AC: @full-model-loader ac-8
def test_native_checkpoint_detection_reads_only_header(tmp_path, monkeypatch):
    path = tmp_path / "header.safetensors"
    save_file(_native_tensors(), str(path))
    real_safe_open = model_loader.safe_open
    reads = []

    class HeaderOnlyHandle:
        def __init__(self, *args, **kwargs):
            self.handle = real_safe_open(*args, **kwargs)

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, key):
            reads.append(key)
            raise AssertionError("opening/detecting must not read tensors")

    monkeypatch.setattr(model_loader, "safe_open", HeaderOnlyHandle)
    with ModelLoader(str(path)) as loader:
        assert loader.arch == "krea2"
        assert loader.affected_keys == frozenset("diffusion_model." + key for key in NATIVE_KEYS)
        assert loader.loaded_bytes == 0
        assert reads == []
