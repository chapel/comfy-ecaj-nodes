"""Cross-lane proofs: parsed source layouts reach the shared executor."""

import pytest
import torch
from safetensors.torch import save_file

from lib.gpu_ops import apply_lora_batch_gpu
from lib.lora.flux import FluxLoader


# AC: @lora-loaders ac-2
# AC: @batched-executor ac-2
@pytest.mark.parametrize("strength", [0.5, -1.0])
def test_expanded_flux_diffusers_mlp_reaches_sliced_execution(tmp_path, strength):
    stem = "transformer.single_transformer_blocks.0.proj_mlp"
    key = "diffusion_model.single_blocks.0.linear1.weight"
    up = torch.arange(32.0).reshape(16, 2) / 32
    down = torch.arange(8.0).reshape(2, 4) / 8
    path = tmp_path / "expanded-mlp.safetensors"
    save_file({stem + ".lora_up.weight": up, stem + ".lora_down.weight": down}, path)
    loader = FluxLoader()
    try:
        loader.load(str(path), strength=strength)
        specs = loader.get_delta_specs([key], {key: 0})
        assert len(specs) == 1
        assert specs[0].offset == (12, 16)
        loader.validate_compatible_keys({key}, {key: (28, 4)})
        base = torch.arange(112.0).reshape(1, 28, 4)
        expected = base.clone()
        expected[:, 12:28] += strength * (up @ down)
        actual = apply_lora_batch_gpu([key], base, specs, "cpu", torch.float32)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, :12], base[:, :12], rtol=0, atol=0)
        assert not torch.cuda.is_initialized()
    finally:
        loader.cleanup()
    assert loader.loaded_bytes == 0
