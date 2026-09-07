"""Isolated, optional real Comfy contract. No checkpoints, GPU or service calls."""

# ruff: noqa: E402  -- CPU CLI parsing must precede all Comfy runtime imports.
import importlib
import os
import sys
import types
from pathlib import Path

comfy_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(comfy_root))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
from comfy.cli_args import args

assert args.cpu  # MUST precede model_management / sd imports

import torch
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy.weight_adapter.lora import LoRAAdapter

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("ecaj_contract")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
model_node = importlib.import_module("ecaj_contract.nodes.exit")
clip_node = importlib.import_module("ecaj_contract.nodes.clip_exit")
weights_module = importlib.import_module("ecaj_contract.nodes.effective_weights")
recipe_module = importlib.import_module("ecaj_contract.lib.recipe")
CPU = torch.device("cpu")

for kind in ("MODEL", "CLIP"):
    module = torch.nn.Module()
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.tensor([[10.0]])))
    layer.register_parameter("untouched", torch.nn.Parameter(torch.tensor([[20.0]])))
    prefix = "diffusion_model" if kind == "MODEL" else "clip_l"
    module.add_module(prefix, layer)
    key, other = prefix + ".weight", prefix + ".untouched"
    patcher = ModelPatcher(module, CPU, CPU)
    adapter = LoRAAdapter.load(
        "tiny",
        {
            "tiny.lora_up.weight": torch.tensor([[2.0]]),
            "tiny.lora_down.weight": torch.tensor([[3.0]]),
        },
        alpha=1.0,
        dora_scale=None,
    )
    patcher.add_patches({key: adapter, other: adapter})
    state = weights_module.EffectiveWeights(patcher)
    assert state.raw[key].item() == 10
    assert state[key].item() == 16
    assert not patcher.backup
    if kind == "CLIP":
        source = CLIP(no_init=True)
        source.patcher = patcher
        source.cond_stage_model = module
        source.tokenizer = None
        source.layer_idx = None
        source.tokenizer_options = {}
        source.use_clip_schedule = False
        source.apply_hooks_to_conds = False
        node_module = clip_node
    else:
        source = patcher
        node_module = model_node

    # Execute the production node through its actual chunker and installer.
    # Only file analysis and the evaluator are controlled: +1 isolates the
    # effective-input boundary, not a claim about nonlinear WIDEN arithmetic.
    base = recipe_module.RecipeBase(
        model_patcher=source, arch="sdxl", domain="clip" if kind == "CLIP" else "diffusion"
    )
    target = recipe_module.RecipeLoRA(loras=())
    recipe = recipe_module.RecipeMerge(base=base, target=target, backbone=None, t_factor=1.0)
    loader = types.SimpleNamespace(cleanup=lambda: None, loaded_bytes=0)
    node_module.analyze_recipe = lambda *a, **kw: types.SimpleNamespace(
        loader=loader, affected_keys={key}, set_affected={str(id(target)): {key}}, arch="sdxl"
    )
    node_module.analyze_recipe_models = lambda *a, **kw: types.SimpleNamespace(
        model_loaders={}, model_affected={}, all_model_keys=set()
    )
    node_module.compile_plan = lambda *a: None
    node_module.execute_plan = lambda **kw: kw["base_batch"] + 1
    if kind == "MODEL":
        (result,) = node_module.WIDENExitNode().execute(recipe, enable_cache=False)
        returned_patcher = result
    else:
        (result,) = node_module.WIDENCLIPExitNode().execute(recipe)
        returned_patcher = result.patcher
    assert returned_patcher.patch_weight_to_device(key, CPU, return_weight=True).item() == 17
    assert returned_patcher.patch_weight_to_device(other, CPU, return_weight=True).item() == 26
    assert patcher.patch_weight_to_device(key, CPU, return_weight=True).item() == 16
    assert layer.weight.item() == 10 and not patcher.backup
    assert len(patcher.patches[key]) == 1 and len(returned_patcher.patches[key]) == 2

assert not torch.cuda.is_initialized()
print("MODEL=17 CLIP=17 untouched=26 CUDA_initialized=False")
