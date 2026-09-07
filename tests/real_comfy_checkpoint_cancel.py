"""Isolated CPU checkpoint publication proof; never load a real checkpoint."""

# ruff: noqa: E402 -- CPU CLI parsing must precede Comfy runtime imports.
import gc
import importlib
import itertools
import os
import runpy
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
from comfy.cli_args import args

assert args.cpu
import comfy.model_management as mm
import comfy.utils
import torch
from comfy.model_patcher import ModelPatcher

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("ecaj_checkpoint_contract")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
node = importlib.import_module("ecaj_checkpoint_contract.nodes.exit")
recipes = importlib.import_module("ecaj_checkpoint_contract.lib.recipe")
assert (
    node.throw_exception_if_processing_interrupted is mm.throw_exception_if_processing_interrupted
)
case = runpy.run_path(str(Path(__file__).with_name("test_checkpoint_cancel.py")))[
    "checkpoint_case"
]


def run():
    count = 0
    for merged, cache, cancel in itertools.product((False, True), repeat=3):
        module = torch.nn.Module()
        layer = torch.nn.Module()
        layer.register_parameter("w", torch.nn.Parameter(torch.tensor([2.0])))
        module.add_module("diffusion_model", layer)
        patcher = ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))
        mm.interrupt_current_processing(False)
        # Real progress must also observe the isolated process's cancellation.
        comfy.utils.set_progress_bar_global_hook(
            lambda *a, **kw: mm.throw_exception_if_processing_interrupted()
        )
        try:
            case(
                node,
                recipes,
                patcher,
                Path(os.environ["TMPDIR"]) / "output.safetensors",
                merged,
                cache,
                cancel,
                lambda: mm.interrupt_current_processing(True),
                mm.throw_exception_if_processing_interrupted,
            )
            count += 1
        finally:
            mm.interrupt_current_processing(False)
            comfy.utils.set_progress_bar_global_hook(None)
            del patcher
            gc.collect()
    assert count == 8
    assert not torch.cuda.is_initialized()
    print(f"CHECKPOINT_MATRIX={count} CUDA_initialized=False")


run()
