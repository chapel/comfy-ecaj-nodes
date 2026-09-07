"""CPU regressions for save ownership, cancellation and effective inputs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from lib.recipe import CheckpointComponents, RecipeBase, RecipeLoRA, RecipeMerge
from nodes import exit as exit_module


@pytest.fixture
def save_setup(monkeypatch, tmp_path, mock_model_patcher):
    target = tmp_path / "output.safetensors"
    monkeypatch.setattr(exit_module, "_resolve_save_path", lambda *a, **kw: str(target))
    monkeypatch.setattr(exit_module, "compute_lora_stats", lambda *a: {})
    monkeypatch.setattr(exit_module, "validate_checkpoint_components", lambda *a, **kw: None)
    return target, RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")


def merge(base):
    return RecipeMerge(base=base, target=RecipeLoRA(loras=()), backbone=None, t_factor=1.0)


# AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
@pytest.mark.parametrize("merged", [False, True])
def test_cache_disabled_cannot_overwrite_user_file(save_setup, merged):
    target, base = save_setup
    save_file({"user_original": torch.ones(1)}, str(target))
    original = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        exit_module.WIDENExitNode().execute(
            merge(base) if merged else base,
            save_model=True,
            model_name="output",
            enable_cache=False,
        )
    assert target.read_bytes() == original


# AC: @streaming-materialization-progress ac-failure-status-not-success
@pytest.mark.parametrize("tick", [1, 2, 6])
def test_interrupt_aborts_before_publication(save_setup, monkeypatch, tick):
    target, base = save_setup

    class Interrupted(Exception):
        pass

    class Bar:
        def __init__(self, total):
            self.calls = 0

        def update(self, value):
            self.calls += 1
            if self.calls == tick:
                raise Interrupted("cancelled")

    monkeypatch.setattr(exit_module, "ProgressBar", Bar)
    with pytest.raises(Interrupted):
        exit_module.WIDENExitNode().execute(
            base, save_model=True, model_name="output", enable_cache=False
        )
    assert not target.exists()
    assert not list(target.parent.glob(".ecaj_tmp_*"))


# AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
@pytest.mark.parametrize("fail", [False, True])
def test_noop_checkpoint_releases_before_artifact_reload(save_setup, monkeypatch, fail):
    target, base = save_setup
    base = RecipeBase(
        model_patcher=base.model_patcher,
        arch=base.arch,
        checkpoint_components=CheckpointComponents(clip=object(), vae=object()),
    )
    events = []

    def save(path, model, **kwargs):
        events.append(("save", model))
        if fail:
            raise RuntimeError("save failed")

    monkeypatch.setattr(exit_module, "save_comfy_checkpoint", save)
    monkeypatch.setattr(
        exit_module, "_release_temporary_checkpoint_model", lambda m: events.append(("release", m))
    )
    returned = object()

    def reload(path):
        assert [e[0] for e in events] == ["save", "release"]
        return returned

    monkeypatch.setattr(exit_module, "_load_checkpoint_artifact", reload)
    if fail:
        with pytest.raises(RuntimeError, match="save failed"):
            exit_module.WIDENExitNode().execute(
                base, save_model=True, model_name="output", enable_cache=False
            )
    else:
        assert exit_module.WIDENExitNode().execute(
            base, save_model=True, model_name="output", enable_cache=False
        ) == (returned,)
    assert [e[0] for e in events] == ["save", "release"]
    assert events[0][1] is events[1][1]


# AC: @memory-management ac-13
@pytest.mark.parametrize("mode", ["patch", "diffusion", "checkpoint"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_second_acquisition_failure_cleans_first(save_setup, monkeypatch, mode, error_type):
    _, base = save_setup
    if mode == "checkpoint":
        base = RecipeBase(
            model_patcher=base.model_patcher,
            arch=base.arch,
            checkpoint_components=CheckpointComponents(clip=object(), vae=object()),
        )
    loader = SimpleNamespace(cleanup=Mock())
    monkeypatch.setattr(
        exit_module,
        "analyze_recipe",
        lambda *a, **kw: SimpleNamespace(loader=loader, affected_keys=set()),
    )

    def fail(*a, **kw):
        raise error_type("second acquisition")

    monkeypatch.setattr(exit_module, "analyze_recipe_models", fail)
    with pytest.raises(error_type, match="second acquisition"):
        exit_module.WIDENExitNode().execute(
            merge(base), save_model=mode != "patch", model_name="output", enable_cache=False
        )
    loader.cleanup.assert_called_once()


# AC: @full-saved-model-output ac-complete-artifact
@pytest.mark.parametrize("merged", [False, True])
def test_diffusion_save_preserves_upstream_effective_weights(save_setup, monkeypatch, merged):
    from safetensors.torch import load_file

    target, base = save_setup
    patcher = base.model_patcher
    key = next(iter(patcher._state_dict))
    raw = patcher._state_dict[key].clone()
    delta = torch.full_like(raw, 3.0)
    patcher.add_patches({key: ("diff", (delta,))})
    calls = []

    def effective(name, device_to=None, return_weight=False):
        assert return_weight and device_to == torch.device("cpu")
        calls.append(name)
        return patcher._state_dict[name] + delta

    monkeypatch.setattr(patcher, "patch_weight_to_device", effective, raising=False)
    exit_module.WIDENExitNode().execute(
        merge(base) if merged else base, save_model=True, model_name="output", enable_cache=False
    )
    saved = load_file(str(target))
    torch.testing.assert_close(saved["model." + key], raw + delta)
    torch.testing.assert_close(patcher._state_dict[key], raw)
    assert calls == [key]  # no patch materialization during identity/shape preparation


# AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
def test_checkpoint_last_group_is_dead_at_reload(save_setup, monkeypatch):
    import gc
    import weakref

    _, base = save_setup
    base = RecipeBase(
        model_patcher=base.model_patcher,
        arch=base.arch,
        checkpoint_components=CheckpointComponents(clip=object(), vae=object()),
    )
    recipe = merge(base)
    key = next(iter(base.model_patcher._state_dict))
    loader = SimpleNamespace(cleanup=lambda: None, loaded_bytes=0)
    monkeypatch.setattr(
        exit_module,
        "analyze_recipe",
        lambda *a, **kw: SimpleNamespace(
            loader=loader,
            arch="sdxl",
            affected_keys={key},
            set_affected={str(id(recipe.target)): {key}},
        ),
    )
    monkeypatch.setattr(
        exit_module,
        "analyze_recipe_models",
        lambda *a, **kw: SimpleNamespace(
            model_loaders={}, model_affected={}, all_model_keys=set()
        ),
    )
    monkeypatch.setattr(exit_module, "compile_plan", lambda *a: None)
    refs = []

    def evaluate(**kw):
        value = torch.full((4, 4), 7.0)
        refs.append(weakref.ref(value))
        return {key: value}

    monkeypatch.setattr(exit_module, "chunked_evaluation", evaluate)
    monkeypatch.setattr(exit_module, "save_comfy_checkpoint", lambda *a, **kw: None)
    returned = object()

    def reload(path):
        gc.collect()
        assert refs and refs[0]() is None
        return returned

    monkeypatch.setattr(exit_module, "_load_checkpoint_artifact", reload)
    assert exit_module.WIDENExitNode().execute(
        recipe, save_model=True, model_name="output", enable_cache=False
    ) == (returned,)


# AC: @full-saved-model-output ac-return-loaded-model
# AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
@pytest.mark.parametrize("mixed", [False, True])
def test_selected_artifact_return_has_artifact_weights_and_dtypes(tmp_path, monkeypatch, mixed):
    from safetensors.torch import load_file

    path = tmp_path / "artifact.safetensors"
    tensors = {
        "model.diffusion_model.weight": torch.full((2, 2), 20.0),
        "model.diffusion_model.bias": torch.full(
            (2,), 7.0, dtype=torch.float16 if mixed else torch.float32
        ),
    }
    save_file(tensors, str(path))
    calls = []

    def comfy_loader(filename, model_options=None):
        calls.append((filename, model_options))
        # Comfy boundary stand-in: real CPU storage owned by a new torch module,
        # not a copied source patcher or an ecaj patch payload.
        module = torch.nn.Module()
        for name, tensor in load_file(filename).items():
            module.register_parameter(name.rsplit(".", 1)[-1], torch.nn.Parameter(tensor))
        return SimpleNamespace(model=module, patches={}, model_state_dict=module.state_dict)

    monkeypatch.setattr(exit_module, "_comfy_load_diffusion_model", comfy_loader)
    result = exit_module._load_diffusion_model_artifact(str(path))
    assert calls == [(str(path), {})]
    assert result.patches == {}
    for key, value in tensors.items():
        actual = result.model_state_dict()[key.rsplit(".", 1)[-1]]
        torch.testing.assert_close(actual, value)
        assert actual.dtype == value.dtype


# AC: @exit-model-persistence ac-6
def test_checkpoint_companion_only_changes_invalidate_artifact(save_setup, monkeypatch):
    from safetensors.torch import load_file

    from tests.test_effective_weights_contract import StaticPatcher

    path, base = save_setup
    clip_patcher = StaticPatcher()
    clip = SimpleNamespace(patcher=clip_patcher)
    vae_state = {"decoder.weight": torch.full((2, 2), 8.0)}
    vae = SimpleNamespace(get_sd=lambda: vae_state)
    recipe = RecipeBase(
        model_patcher=base.model_patcher,
        arch="sdxl",
        checkpoint_components=CheckpointComponents(clip=clip, vae=vae),
    )
    writes = []

    def comfy_save(filename, model, *, clip, vae, metadata, **kw):
        from nodes.effective_weights import EffectiveWeights

        tensors = {"model." + k: v for k, v in model.model_state_dict().items()}
        tensors["conditioner.embedders.0.weight"] = EffectiveWeights(clip.patcher)["w0"]
        tensors["first_stage_model.decoder.weight"] = vae.get_sd()["decoder.weight"]
        save_file(tensors, filename, metadata=metadata)
        writes.append(filename)

    monkeypatch.setattr("comfy.sd.save_checkpoint", comfy_save)
    monkeypatch.setattr(exit_module, "_load_checkpoint_artifact", lambda path: object())
    node = exit_module.WIDENExitNode()
    node.execute(recipe, save_model=True, model_name="output", enable_cache=True)
    first = path.read_bytes()
    node.execute(recipe, save_model=True, model_name="output", enable_cache=True)
    assert len(writes) == 1 and path.read_bytes() == first
    clip_patcher.patches["w0"][0][1][1][0].add_(2)
    node.execute(recipe, save_model=True, model_name="output", enable_cache=True)
    assert len(writes) == 2 and path.read_bytes() != first
    torch.testing.assert_close(
        load_file(str(path))["conditioner.embedders.0.weight"], torch.full((2, 2), 3.0)
    )
    second = path.read_bytes()
    vae_state["decoder.weight"].add_(1)
    node.execute(recipe, save_model=True, model_name="output", enable_cache=True)
    assert len(writes) == 3 and path.read_bytes() != second
    torch.testing.assert_close(
        load_file(str(path))["first_stage_model.decoder.weight"], torch.full((2, 2), 9.0)
    )


# AC: @exit-model-persistence ac-bounded-identity-preparation
@pytest.mark.parametrize("mode", ["noop", "diffusion", "checkpoint", "patch"])
def test_disabled_cache_never_scans_input_content(save_setup, monkeypatch, mode):
    _, base = save_setup
    if mode == "checkpoint":
        base = RecipeBase(
            model_patcher=base.model_patcher,
            arch="sdxl",
            checkpoint_components=CheckpointComponents(clip=object(), vae=object()),
        )
        monkeypatch.setattr(exit_module, "save_comfy_checkpoint", lambda *a, **kw: None)
        monkeypatch.setattr(exit_module, "_load_checkpoint_artifact", lambda *a: object())

    def forbidden(*a, **kw):
        raise AssertionError("cache-disabled full scan")

    monkeypatch.setattr(exit_module, "compute_base_identity", forbidden)
    monkeypatch.setattr(exit_module, "compute_lora_stats", forbidden)
    result = exit_module.WIDENExitNode().execute(
        base if mode == "noop" else merge(base),
        save_model=mode != "patch",
        model_name="output",
        enable_cache=False,
    )
    assert len(result) == 1 and result[0] is not None


# AC: @accurate-ram-preflight ac-2
@pytest.mark.parametrize("mode", ["patch", "diffusion", "checkpoint", "clip"])
def test_node_preflight_caps_tiny_actual_group(save_setup, monkeypatch, mode):
    from nodes import clip_exit

    _, base = save_setup
    module = clip_exit if mode == "clip" else exit_module
    if mode == "clip":
        clip = SimpleNamespace(
            patcher=base.model_patcher,
            clone=lambda: SimpleNamespace(add_patches=lambda *a, **kw: None),
        )
        base = RecipeBase(model_patcher=clip, arch="sdxl", domain="clip")
        key = next(iter(clip.patcher._state_dict))
    else:
        key = next(iter(base.model_patcher._state_dict))
    if mode == "checkpoint":
        base = RecipeBase(
            model_patcher=base.model_patcher,
            arch="sdxl",
            checkpoint_components=CheckpointComponents(clip=object(), vae=object()),
        )
        monkeypatch.setattr(module, "save_comfy_checkpoint", lambda *a, **kw: None)
        monkeypatch.setattr(module, "_load_checkpoint_artifact", lambda *a: object())
    recipe = merge(base)
    loader = SimpleNamespace(cleanup=lambda: None, loaded_bytes=0)
    monkeypatch.setattr(
        module,
        "analyze_recipe",
        lambda *a, **kw: SimpleNamespace(
            loader=loader,
            arch="sdxl",
            affected_keys={key},
            set_affected={str(id(recipe.target)): {key}},
        ),
    )
    monkeypatch.setattr(
        module,
        "analyze_recipe_models",
        lambda *a, **kw: SimpleNamespace(
            model_loaders={}, model_affected={}, all_model_keys=set()
        ),
    )
    monkeypatch.setattr(module, "compile_plan", lambda *a: SimpleNamespace(ops=()))
    monkeypatch.setattr(module, "execute_plan", lambda **kw: kw["base_batch"] + 1)
    preflight = Mock()
    monkeypatch.setattr(module, "check_ram_preflight", preflight)
    if mode == "clip":
        module.WIDENCLIPExitNode().execute(recipe)
    else:
        module.WIDENExitNode().execute(
            recipe, save_model=mode != "patch", model_name="output", enable_cache=False
        )
    preflight.assert_called_once()
    assert preflight.call_args.kwargs["worst_chunk_bytes"] == 64  # one actual 4x4 fp32 key
