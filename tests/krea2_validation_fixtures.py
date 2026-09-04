"""Shared CPU-safe fixtures for Krea 2 validation tests."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

KREA2_VALIDATION_KEYS = (
    "diffusion_model.first.weight",
    "diffusion_model.pe_embedder.weight",
    "diffusion_model.blocks.0.mod.lin",
    "diffusion_model.blocks.0.attn.wq.weight",
    "diffusion_model.blocks.0.attn.wk.weight",
    "diffusion_model.blocks.0.attn.wv.weight",
    "diffusion_model.blocks.0.attn.wo.weight",
    "diffusion_model.blocks.0.mlp.gate.weight",
    "diffusion_model.blocks.1.attn.wq.weight",
    "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
    "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.weight",
    "diffusion_model.txtfusion.layerwise_blocks.1.attn.wo.weight",
    "diffusion_model.txtfusion.projector.weight",
    "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight",
    "diffusion_model.txtfusion.refiner_blocks.1.attn.wo.weight",
    "diffusion_model.txtmlp.3.weight",
    "diffusion_model.tmlp.0.weight",
    "diffusion_model.tmlp.2.weight",
    "diffusion_model.tproj.1.weight",
    "diffusion_model.tproj.1.bias",
    "diffusion_model.last.linear.weight",
)

KREA2_VALIDATION_SHAPES = {
    "diffusion_model.blocks.0.attn.wq.weight": (4, 3),
    "diffusion_model.blocks.0.attn.wk.weight": (4, 3),
    "diffusion_model.blocks.0.attn.wv.weight": (4, 3),
    "diffusion_model.blocks.0.attn.wo.weight": (4, 3),
    "diffusion_model.blocks.0.mlp.gate.weight": (4, 3),
    "diffusion_model.blocks.1.attn.wq.weight": (4, 3),
    "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight": (4, 3),
    "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.weight": (4, 3),
    "diffusion_model.txtfusion.layerwise_blocks.1.attn.wo.weight": (4, 3),
    "diffusion_model.txtfusion.projector.weight": (4, 3),
    "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight": (3, 5),
    "diffusion_model.txtfusion.refiner_blocks.1.attn.wo.weight": (4, 3),
    "diffusion_model.first.weight": (7, 6),
    "diffusion_model.txtmlp.3.weight": (7, 7),
    "diffusion_model.tmlp.0.weight": (4, 3),
    "diffusion_model.tmlp.2.weight": (5, 4),
    "diffusion_model.tproj.1.weight": (6, 5),
    "diffusion_model.tproj.1.bias": (6,),
    "diffusion_model.last.linear.weight": (8, 7),
    "diffusion_model.pe_embedder.weight": (2, 2),
    "diffusion_model.blocks.0.mod.lin": (2, 2),
}

KREA2_BLOCK_CONTROL_KEYS = (
    "diffusion_model.blocks.0.attn.wq.weight",
    "diffusion_model.blocks.1.attn.wq.weight",
    "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight",
    "diffusion_model.txtfusion.refiner_blocks.1.attn.wo.weight",
    "diffusion_model.tproj.1.weight",
    "diffusion_model.blocks.0.mod.lin",
    "diffusion_model.unknown.weight",
)


def krea2_state_tensors(
    *,
    dtype: torch.dtype = torch.float32,
    mixed_dtype_key: str | None = None,
    mixed_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Build a small Krea-like state dict with deterministic shapes."""
    tensors: dict[str, torch.Tensor] = {}
    for key in KREA2_VALIDATION_KEYS:
        shape = KREA2_VALIDATION_SHAPES[key]
        tensor_dtype = mixed_dtype if key == mixed_dtype_key else dtype
        tensors[key] = torch.ones(shape, dtype=tensor_dtype)
    return tensors


def diffusers_krea2_lora_tensors() -> dict[str, torch.Tensor]:
    """Supported public Diffusers-style Krea 2 LoRA fixture."""
    return {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        "transformer.text_fusion.refiner_blocks.0.ff.down.lora_A.weight": torch.ones(2, 5),
        "transformer.text_fusion.refiner_blocks.0.ff.down.lora_B.weight": torch.ones(3, 2),
        "transformer.text_fusion.layerwise_blocks.1.attn.to_out.0.lora_A.weight": torch.ones(
            2,
            3,
        ),
        "transformer.text_fusion.layerwise_blocks.1.attn.to_out.0.lora_B.weight": torch.ones(
            4,
            2,
        ),
        "transformer.time_embed.linear_1.lora_A.weight": torch.ones(2, 3),
        "transformer.time_embed.linear_1.lora_B.weight": torch.ones(4, 2),
        "transformer.time_embed.linear_2.lora_A.weight": torch.ones(2, 4),
        "transformer.time_embed.linear_2.lora_B.weight": torch.ones(5, 2),
        "transformer.time_mod_proj.lora_A.weight": torch.ones(2, 5),
        "transformer.time_mod_proj.lora_B.weight": torch.ones(6, 2),
    }


def native_krea2_lora_tensors() -> dict[str, torch.Tensor]:
    """Supported public native/Comfy-style Krea 2 LoRA fixture."""
    return {
        "diffusion_model.blocks.0.attn.wk.lora_down.weight": torch.ones(2, 3),
        "diffusion_model.blocks.0.attn.wk.lora_up.weight": torch.ones(4, 2),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_down.weight": torch.ones(
            2,
            3,
        ),
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_up.weight": torch.ones(4, 2),
        "diffusion_model.tproj.1.diff_b": torch.arange(6, dtype=torch.float32),
    }


def unsupported_krea2_lora_tensors() -> dict[str, torch.Tensor]:
    """Unsupported and incomplete groups for complete-or-rejected diagnostics."""
    return {
        "transformer.unknown_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": torch.ones(2, 3),
    }


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return str(path)
