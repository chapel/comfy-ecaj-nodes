"""Checkpoint publication boundary, with tiny real safetensors artifacts."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file


def checkpoint_case(node, recipes, patcher, target, merged, cache, cancel, request, check):
    """Shared assertions for stub-runtime tests and the isolated real Comfy run."""
    import comfy.sd

    old = {
        "model.diffusion_model.w": torch.tensor([1.0]),
        "conditioner.x": torch.tensor([1.0]),
        "first_stage_model.x": torch.tensor([1.0]),
    }
    metadata = node.build_metadata(
        "{}",
        "old-recipe",
        [],
        None,
        output_mode="full",
        artifact_kind="checkpoint",
        source_model_kind="checkpoint",
        checkpoint_components=True,
    )
    save_file(old, str(target), metadata=metadata)
    node._classify_temp_artifact(str(target), expected_kind="checkpoint")
    before = target.read_bytes()
    base = recipes.RecipeBase(
        model_patcher=patcher,
        arch="sdxl",
        checkpoint_components=recipes.CheckpointComponents(clip=object(), vae=object()),
    )
    recipe = (
        recipes.RecipeMerge(
            base=base, target=recipes.RecipeLoRA(loras=()), backbone=None, t_factor=1.0
        )
        if merged
        else base
    )
    events, saved_models, released_models, raised = [], [], [], []
    release = node._release_temporary_checkpoint_model
    returned = object()
    caught = None
    result = None

    def serialize(filename, model, **kwargs):
        events.append("serialize")
        saved_models.append(model)
        if merged:
            key = next(iter(patcher.model_state_dict()))
            kind, (value,) = model.patches[key][-1][1]
            assert kind == "set"
            torch.testing.assert_close(value, torch.full_like(value, 7.0))
        save_file({k: v + 1 for k, v in old.items()}, filename, metadata=kwargs["metadata"])
        if cancel:
            request()

    def checked():
        try:
            check()
        except BaseException as exc:
            raised.append(exc)
            raise

    def released(model):
        events.append("release")
        released_models.append(model)
        release(model)

    def reload(filename):
        events.append("reload")
        assert released_models == saved_models
        assert filename == str(target)
        assert load_file(filename)["model.diffusion_model.w"].item() == 2
        return returned

    # Keep the real publisher, classifier, fsync, replace, clone/install and
    # release. Substitute only heavy serialization/reload and merge evaluation.
    with (
        patch.object(node, "_resolve_save_path", lambda *a, **kw: str(target)),
        patch.object(node, "throw_exception_if_processing_interrupted", checked, create=True),
        patch.object(comfy.sd, "save_checkpoint", serialize),
        patch.object(node, "_release_temporary_checkpoint_model", released),
        patch.object(node, "_load_checkpoint_artifact", reload),
        patch.object(node.os, "replace", wraps=node.os.replace) as replace,
    ):
        if merged:
            key = next(iter(patcher.model_state_dict()))
            loader = SimpleNamespace(cleanup=lambda: None, loaded_bytes=0)
            with (
                patch.object(
                    node,
                    "analyze_recipe",
                    lambda *a, **kw: SimpleNamespace(
                        loader=loader,
                        arch="sdxl",
                        affected_keys={key},
                        set_affected={str(id(recipe.target)): {key}},
                    ),
                ),
                patch.object(node, "compile_plan", lambda *a: None),
                patch.object(
                    node,
                    "chunked_evaluation",
                    lambda **kw: {key: torch.full_like(patcher.model_state_dict()[key], 7.0)},
                ),
            ):
                try:
                    result = node.WIDENExitNode().execute(
                        recipe, save_model=True, model_name="output", enable_cache=cache
                    )
                except BaseException as exc:
                    caught = exc
        else:
            try:
                result = node.WIDENExitNode().execute(
                    recipe, save_model=True, model_name="output", enable_cache=cache
                )
            except BaseException as exc:
                caught = exc

    assert events[:2] == ["serialize", "release"], (events, caught)
    assert len(saved_models) == 1  # cache-enabled case must actually miss
    assert released_models == saved_models
    assert saved_models[0] is not patcher
    assert replace.call_count == (0 if cancel else 1)
    assert not list(target.parent.glob(".ecaj_tmp_*"))
    if cancel:
        assert target.read_bytes() == before, "cancelled checkpoint replaced the existing artifact"
        assert load_file(str(target))["model.diffusion_model.w"].item() == 1
        assert events == ["serialize", "release"]  # no reload on cancellation
        assert len(raised) == 1 and caught is raised[0]
        assert result is None
    else:
        assert caught is None, caught
        assert not raised
        assert result == (returned,)
        assert events == ["serialize", "release", "reload"]
        assert target.read_bytes() != before
        node._classify_temp_artifact(str(target), expected_kind="checkpoint")


# AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
# AC: @saved-model-artifact-safety ac-no-partial-publication
# AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
@pytest.mark.parametrize("merged", [False, True], ids=["noop", "merged"])
@pytest.mark.parametrize("cache", [False, True], ids=["cache-off", "cache-on"])
@pytest.mark.parametrize("cancel", [False, True], ids=["publish", "cancel"])
def test_checkpoint_publication(tmp_path, mock_model_patcher, merged, cache, cancel):
    from lib import recipe
    from nodes import exit as node

    interrupt = KeyboardInterrupt("cancel during checkpoint serialization")
    pending = False

    def request():
        nonlocal pending
        pending = True

    def check():
        if pending:
            raise interrupt

    checkpoint_case(
        node,
        recipe,
        mock_model_patcher,
        tmp_path / "output.safetensors",
        merged,
        cache,
        cancel,
        request,
        check,
    )


# AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
# AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
def test_real_checkpoint_cancellation_optional(tmp_path):
    root = os.environ.get("ECAJ_COMFY_ROOT")
    if not root:
        pytest.skip("Set ECAJ_COMFY_ROOT for real Comfy CPU interruption checks")
    script = Path(__file__).with_name("real_comfy_checkpoint_cancel.py")
    completed = subprocess.run(
        [sys.executable, str(script), root],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "TMPDIR": str(tmp_path),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CHECKPOINT_MATRIX=8 CUDA_initialized=False" in completed.stdout
