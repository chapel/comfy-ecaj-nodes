"""Checkout registration smoke, independent of pytest's import alias bridge.

This verifies discovery surfaces, not a running ComfyUI or model execution.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = {
    "WIDENEntry",
    "WIDENCLIPEntry",
    "WIDENCLIPExit",
    "WIDENLoRA",
    "WIDENCompose",
    "WIDENMerge",
    "WIDENExit",
    "WIDENBlockConfigSDXL",
    "WIDENBlockConfigSDXLCLIP",
    "WIDENBlockConfigZImage",
    "WIDENBlockConfigQwen",
    "WIDENBlockConfigFlux",
    "WIDENBlockConfigKrea2",
    "WIDENModelInput",
    "WIDENDiffusionModelInput",
    "WIDENCLIPLoRA",
    "WIDENCLIPCompose",
    "WIDENCLIPMerge",
    "WIDENCLIPModelInput",
}
SMOKE = """
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

# Only the file-dropdown host API is needed for registration/INPUT_TYPES.
folder_paths = ModuleType("folder_paths")
folder_paths.get_filename_list = lambda folder: []
sys.modules["folder_paths"] = folder_paths
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "ecaj_registration_smoke", root / "__init__.py",
    submodule_search_locations=[str(root)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
classes = package.NODE_CLASS_MAPPINGS
expected = set(sys.argv[2:])
assert set(classes) == expected, (set(classes) ^ expected)
assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == expected
for node_id, cls in classes.items():
    assert isinstance(cls, type), node_id
    assert cls.CATEGORY.startswith("ecaj/") and cls.CATEGORY == cls.CATEGORY.lower(), node_id
    assert isinstance(cls.FUNCTION, str), node_id
    assert callable(getattr(cls, cls.FUNCTION, None)), (node_id, cls.FUNCTION)
    assert isinstance(cls.RETURN_TYPES, tuple), node_id
    inputs = cls.INPUT_TYPES()
    assert isinstance(inputs, dict) and isinstance(inputs["required"], dict), node_id
    assert isinstance(package.NODE_DISPLAY_NAME_MAPPINGS[node_id], str), node_id
assert not torch.cuda.is_initialized()
print("registered", len(classes))
"""


def run_smoke(root):
    return subprocess.run(
        [sys.executable, "-I", "-c", SMOKE, str(root), *sorted(EXPECTED_IDS)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


# AC: @comfyui-packaging ac-1, ac-2, ac-4 (CPU discovery surface only)
def test_real_checkout_registration():
    result = run_smoke(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "registered 19"


@pytest.mark.parametrize("fault", ["missing_id", "missing_import", "missing_function"])
def test_registration_smoke_rejects_broken_root(tmp_path, fault):
    """Mutation sensitivity: alter actual root source, not an invented registry."""
    source = (ROOT / "__init__.py").read_text()
    if fault == "missing_id":
        source = source.replace(
            '        "WIDENBlockConfigKrea2": WIDENBlockConfigKrea2Node,\n', ""
        )
    elif fault == "missing_import":
        source = source.replace(
            "    from .nodes.block_config_krea2 import WIDENBlockConfigKrea2Node\n", ""
        )
    else:
        source += '\nWIDENBlockConfigKrea2Node.FUNCTION = "missing_function"\n'
    assert source != (ROOT / "__init__.py").read_text()
    (tmp_path / "__init__.py").write_text(source)
    for name in ("nodes", "lib"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    result = run_smoke(tmp_path)
    assert result.returncode != 0
    assert "WIDENBlockConfigKrea2" in result.stderr, result.stderr
