"""Normal CPU contract tests plus opt-in isolated real Comfy lane."""

import gc
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch

from nodes.effective_weights import EffectiveWeights, SelectedWeights


class StaticPatcher:
    """Faithful return_weight contract double; never mutates resident weights."""

    def __init__(self):
        self.raw = {f"w{i}": torch.full((2, 2), float(i)) for i in range(8)}
        self.patches = {
            k: [(1.0, ("diff", (torch.ones(2, 2),)), 1.0, None, None)] for k in self.raw
        }
        self.calls = []
        self.refs = []

    def model_state_dict(self):
        return dict(self.raw)

    def patch_weight_to_device(self, key, device_to=None, return_weight=False):
        assert device_to == torch.device("cpu") and return_weight
        self.calls.append(key)
        value = self.raw[key] + self.patches[key][0][1][1][0]
        self.refs.append(weakref.ref(value))
        return value


# AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
def test_metadata_and_group_creation_do_not_materialize_weights():
    patcher = StaticPatcher()
    state = EffectiveWeights(patcher)
    group = SelectedWeights(state, state.keys())
    assert len(group) == 8
    assert {k: tuple(v.shape) for k, v in state.raw.items()} == {k: (2, 2) for k in patcher.raw}
    assert patcher.calls == []
    for key in group:
        value = group[key]
        torch.testing.assert_close(value, patcher.raw[key] + 1)
        del value
        gc.collect()
        assert all(ref() is None for ref in patcher.refs)
    assert patcher.calls == list(patcher.raw)


# AC: @exit-node ac-7
@pytest.mark.parametrize(
    "field",
    ["hook_patches", "forced_hooks", "weight_wrapper_patches", "injections", "object_patches"],
)
def test_unsupported_behavior_fails_without_weight_access(field):
    patcher = StaticPatcher()
    setattr(patcher, field, {"active": object()})
    with pytest.raises(ValueError, match=field):
        EffectiveWeights(patcher)
    assert not patcher.calls


# AC: @exit-node ac-7
def test_opaque_patch_functions_and_nested_patch_payloads_fail():
    for record in [
        (1, ("diff", (torch.ones(2, 2),)), 1, None, lambda x: x),
        (1, [object()], 1, None, None),
        (1, ("unknown", (torch.ones(2, 2),)), 1, None, None),
    ]:
        patcher = StaticPatcher()
        patcher.patches = {"w0": [record]}
        with pytest.raises(ValueError, match="Unsupported"):
            EffectiveWeights(patcher)
        assert not patcher.calls


# AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
def test_dynamic_memory_mode_with_static_patch_contract_is_supported():
    patcher = StaticPatcher()
    patcher.is_dynamic = lambda: True
    torch.testing.assert_close(EffectiveWeights(patcher)["w0"], torch.ones(2, 2))


# AC: @exit-node ac-7
# AC: @clip-exit-node ac-4
def test_real_comfy_cpu_contract_optional(tmp_path):
    root = os.environ.get("ECAJ_COMFY_ROOT")
    if not root:
        pytest.skip("Set ECAJ_COMFY_ROOT to run isolated real Comfy CPU patch contract")
    script = Path(__file__).with_name("real_comfy_effective_contract.py")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
    }
    completed = subprocess.run(
        [sys.executable, str(script), root], env=env, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MODEL=17 CLIP=17 untouched=26 CUDA_initialized=False" in completed.stdout
