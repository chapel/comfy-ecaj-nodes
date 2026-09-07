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
from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
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


def run_full_pipeline():
    """Real tiny SDXL LoRA file -> analyzer -> compiler -> WIDEN -> installer."""
    from safetensors.torch import save_file

    module = torch.nn.Module()
    parent = module
    for component in "diffusion_model.input_blocks.0.0".split("."):
        child = torch.nn.Module()
        parent.add_module(component, child)
        parent = child
    parent.register_parameter("weight", torch.nn.Parameter(torch.full((2, 2), 10.0)))
    key = "diffusion_model.input_blocks.0.0.weight"
    patcher = ModelPatcher(module, CPU, CPU)
    upstream = LoRAAdapter.load(
        "upstream",
        {
            "upstream.lora_up.weight": torch.full((2, 1), 2.0),
            "upstream.lora_down.weight": torch.full((1, 2), 3.0),
        },
        alpha=1.0,
        dora_scale=None,
    )
    patcher.add_patches({key: upstream})
    target_path = Path(os.environ["TMPDIR"]) / "target-contract.safetensors"
    save_file(
        {
            "lora_unet_input_blocks_0_0.lora_up.weight": torch.ones(2, 1),
            "lora_unet_input_blocks_0_0.lora_down.weight": torch.ones(1, 2),
        },
        str(target_path),
    )
    old_resolver = model_node._build_lora_resolver
    model_node._build_lora_resolver = lambda: lambda name: str(target_path)
    try:
        base = recipe_module.RecipeBase(model_patcher=patcher, arch="sdxl")
        target = recipe_module.RecipeLoRA(loras=({"path": target_path.name, "strength": 1.0},))
        recipe = recipe_module.RecipeMerge(base=base, target=target, backbone=None, t_factor=1.0)
        (result,) = model_node.WIDENExitNode().execute(recipe, enable_cache=False)
        effective = result.patch_weight_to_device(key, CPU, return_weight=True)
        torch.testing.assert_close(effective, torch.full((2, 2), 17.0))
        torch.testing.assert_close(parent.weight, torch.full((2, 2), 10.0))
        torch.testing.assert_close(
            patcher.patch_weight_to_device(key, CPU, return_weight=True), torch.full((2, 2), 16.0)
        )
        assert len(patcher.patches[key]) == 1 and len(result.patches[key]) == 2
        print("REAL_WIDEN_SDXL=17 (analyzer/compiler/evaluator unmodified)")
    finally:
        model_node._build_lora_resolver = old_resolver


def run_contract():
    for kind in ("MODEL", "CLIP"):
        module = torch.nn.Module()
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.tensor([[10.0]])))
        layer.register_parameter("untouched", torch.nn.Parameter(torch.tensor([[20.0]])))
        prefix = "diffusion_model" if kind == "MODEL" else "clip_l"
        module.add_module(prefix, layer)
        key, other = prefix + ".weight", prefix + ".untouched"
        # Comfy's Dynamic factory itself reroutes CPU to standard ModelPatcher.
        patcher = ModelPatcherDynamic(module, CPU, CPU)
        assert type(patcher) is ModelPatcher
        sampling = object()
        if kind == "MODEL":
            patcher.add_object_patch("model_sampling", sampling)
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
        node_module.compile_plan = lambda *a: types.SimpleNamespace(ops=())
        node_module.execute_plan = lambda **kw: kw["base_batch"] + 1
        if kind == "MODEL":
            (result,) = node_module.WIDENExitNode().execute(recipe, enable_cache=False)
            returned_patcher = result
            assert returned_patcher.object_patches["model_sampling"] is sampling
        else:
            (result,) = node_module.WIDENCLIPExitNode().execute(recipe)
            returned_patcher = result.patcher
        assert returned_patcher.patch_weight_to_device(key, CPU, return_weight=True).item() == 17
        assert returned_patcher.patch_weight_to_device(other, CPU, return_weight=True).item() == 26
        assert patcher.patch_weight_to_device(key, CPU, return_weight=True).item() == 16
        assert layer.weight.item() == 10 and not patcher.backup
        assert len(patcher.patches[key]) == 1 and len(returned_patcher.patches[key]) == 2

        if kind == "MODEL":
            from safetensors.torch import load_file

            # Actual streaming save using real upstream patches: affected and
            # unaffected keys must both include LoRA. Only the Comfy artifact
            # loader boundary is replaced (tiny tensors are not a real model).
            static_source = patcher.clone()
            static_source.object_patches = {}
            saved_base = recipe_module.RecipeBase(model_patcher=static_source, arch="sdxl")
            saved_recipe = recipe_module.RecipeMerge(
                base=saved_base, target=target, backbone=None, t_factor=1.0
            )
            artifact = Path(os.environ["TMPDIR"]) / "effective-contract.safetensors"
            model_node._resolve_save_path = lambda *a, **kw: str(artifact)
            artifact_return = object()
            model_node._load_diffusion_model_artifact = lambda path: artifact_return
            (returned,) = model_node.WIDENExitNode().execute(
                saved_recipe, save_model=True, model_name="effective-contract", enable_cache=False
            )
            assert returned is artifact_return
            saved = load_file(str(artifact))
            assert saved["model." + key].item() == 17
            assert saved["model." + other].item() == 26
            assert layer.weight.item() == 10 and not patcher.backup

    assert not torch.cuda.is_initialized()
    print("MODEL=17 CLIP=17 untouched=26 CUDA_initialized=False")


run_full_pipeline()
run_contract()
# Release tiny patchers while Comfy callback globals are still available.
import gc

gc.collect()
