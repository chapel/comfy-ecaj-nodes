"""Native/Diffusers module mappings from Comfy's utils.py (no model imports)."""

import re

# Comfy utils.UNET_MAP_RESNET, inverted to Diffusers -> native modules.
_SDXL_RESNET = {
    "conv1": "in_layers.2",
    "time_emb_proj": "emb_layers.1",
    "conv2": "out_layers.3",
    "conv_shortcut": "skip_connection",
    "norm1": "in_layers.0",
    "norm2": "out_layers.0",
}
_SDXL_BASIC = {
    "conv_in": "input_blocks.0.0",
    "conv_out": "out.2",
    "conv_norm_out": "out.0",
    "time_embedding.linear_1": "time_embed.0",
    "time_embedding.linear_2": "time_embed.2",
    "add_embedding.linear_1": "label_emb.0.0",
    "add_embedding.linear_2": "label_emb.0.2",
    "class_embedding.linear_1": "label_emb.0.0",
    "class_embedding.linear_2": "label_emb.0.2",
}


def sdxl_module(path):
    """Map standard SDXL UNet (two resnets per down block) Diffusers modules."""
    if path in _SDXL_BASIC:
        return _SDXL_BASIC[path]
    match = re.fullmatch(
        r"(down_blocks|up_blocks)\.(\d+)\.(attentions|resnets)\.(\d+)\.(.+)", path
    )
    if match:
        side, block, kind, layer, module = match.groups()
        block, layer = int(block), int(layer)
        if side == "down_blocks":
            root = f"input_blocks.{1 + 3 * block + layer}"
        else:
            root = f"output_blocks.{3 * block + layer}"
        if kind == "attentions":
            return root + ".1." + module
        return root + ".0." + _SDXL_RESNET.get(module, module)
    match = re.fullmatch(r"mid_block\.(attentions|resnets)\.(\d+)\.(.+)", path)
    if match:
        kind, layer, module = match.groups()
        return (
            "middle_block.1." + module
            if kind == "attentions"
            else f"middle_block.{2 * int(layer)}." + _SDXL_RESNET.get(module, module)
        )
    match = re.fullmatch(r"down_blocks\.(\d+)\.downsamplers\.0\.conv", path)
    if match:
        return f"input_blocks.{3 * (int(match[1]) + 1)}.0.op"
    match = re.fullmatch(r"up_blocks\.(\d+)\.upsamplers\.0\.conv", path)
    if match:
        block = int(match[1])
        return f"output_blocks.{3 * block + 2}.{2 if block < 2 else 1}.conv"
    return path


_FLUX_DOUBLE = {
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "norm1.linear": "img_mod.lin",
    "norm1_context.linear": "txt_mod.lin",
    "ff.net.0.proj": "img_mlp.0",
    "ff.net.2": "img_mlp.2",
    "ff_context.net.0.proj": "txt_mlp.0",
    "ff_context.net.2": "txt_mlp.2",
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
    "attn.norm_q": "img_attn.norm.query_norm",
    "attn.norm_k": "img_attn.norm.key_norm",
    "attn.norm_added_q": "txt_attn.norm.query_norm",
    "attn.norm_added_k": "txt_attn.norm.key_norm",
}
for _p in "qkv":
    _FLUX_DOUBLE[f"attn.to_{_p}"] = f"img_attn.to_{_p}"
    _FLUX_DOUBLE[f"attn.add_{_p}_proj"] = f"txt_attn.to_{_p}"
_FLUX_SINGLE = {
    "norm.linear": "modulation.lin",
    "proj_out": "linear2",
    "attn.norm_q": "norm.query_norm",
    "attn.norm_k": "norm.key_norm",
    "attn.to_qkv_mlp_proj": "linear1",
    "attn.to_out": "linear2",
    "proj_mlp": "proj_mlp",
    **{f"attn.to_{p}": f"to_{p}" for p in "qkv"},
}
_FLUX_BASIC = {
    "proj_out": "final_layer.linear",
    "x_embedder": "img_in",
    "context_embedder": "txt_in",
}
for _source, _target in [("timestep", "time"), ("text", "vector"), ("guidance", "guidance")]:
    for _index, _layer in [(1, "in"), (2, "out")]:
        _FLUX_BASIC[f"time_text_embed.{_source}_embedder.linear_{_index}"] = (
            f"{_target}_in.{_layer}_layer"
        )


def flux_module(path):
    """Comfy utils.flux_to_diffusers mapping, preserving unfused slice names."""
    for source, target, mapping in [
        ("transformer_blocks", "double_blocks", _FLUX_DOUBLE),
        ("single_transformer_blocks", "single_blocks", _FLUX_SINGLE),
    ]:
        match = re.fullmatch(source + r"[._](\d+)[._](.+)", path)
        if not match:
            continue
        index, module = match.groups()
        for alias, native in mapping.items():
            if module in (alias, alias.replace(".", "_")):
                return f"{target}.{index}.{native}"
        raise ValueError(f"Unsupported Flux Diffusers module: {path}")
    return _FLUX_BASIC.get(path, path)
