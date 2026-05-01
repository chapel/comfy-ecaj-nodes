"""Tests for diffusion-model saved output round trip.

AC coverage for:
- @full-saved-model-output ac-diffusion-model-source-kind-round-trip
- @full-saved-model-output ac-diffusion-model-companion-separation
- @full-saved-model-output ac-complete-artifact (diffusion-model branch)
- @full-saved-model-output ac-return-loaded-model (diffusion-model branch)
- @full-saved-model-output ac-cache-reuses-artifact (diffusion-model branch)
- @full-saved-model-output ac-cache-reuse-is-artifact-backed (diffusion-model branch)
- @streaming-full-model-materialization ac-incomplete-write-not-reused
- @streaming-full-model-materialization ac-raw-byte-dtype-preservation
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lib.batch_groups import OpSignature
from lib.persistence import check_full_model_cache
from lib.recipe import RecipeBase, RecipeLoRA, RecipeMerge
from lib.recipe_eval import EvalPlan
from nodes.exit import (
    WIDENExitNode,
    _classify_temp_artifact,
    _load_diffusion_model_artifact,
    _resolve_diffusion_models_path,
    _resolve_save_path,
    _to_external_diffusion_key,
)
from tests.conftest import MockModelPatcher

_DIFFUSION_PREFIX = "diffusion_model."
_EXTERNAL_DIFFUSION_PREFIX = "model.diffusion_model."


# ===========================================================================
# Helpers
# ===========================================================================


def _make_diffusion_only_patcher(arch_keys: tuple[str, ...]):
    """Build a MockModelPatcher whose state dict matches the given arch keys."""
    return MockModelPatcher(keys=arch_keys)


def _make_full_mode_mocks(mock_patcher, keys_to_process):
    mock_loader = MagicMock()
    mock_loader.affected_keys = set(keys_to_process)
    mock_loader.loaded_bytes = 0
    mock_loader.cleanup = MagicMock()

    mock_analyze = MagicMock(
        model_patcher=mock_patcher,
        arch="sdxl",
        loader=mock_loader,
        # set_id_map keying expects int-castable keys
        set_affected={"1": set(keys_to_process)},
        affected_keys=set(keys_to_process),
    )
    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()
    plan = EvalPlan(ops=(), result_reg=0, dead_after=())
    return mock_analyze, mock_model_analysis, mock_loader, plan


def _run_diffusion_save(
    recipe,
    mock_patcher,
    keys,
    save_path,
    *,
    enable_cache=True,
    chunked_eval_override=None,
    comfy_load_side_effect=None,
):
    """Run Exit node in full saved model mode for a diffusion-only recipe.

    By default the Comfy diffusion-model loader is stubbed to return a fresh
    MagicMock so the post-save return path succeeds.  Tests that need to
    assert on the loader call pass `comfy_load_side_effect` to capture
    arguments or return a sentinel.
    """
    mock_analyze, mock_model_analysis, mock_loader, plan = _make_full_mode_mocks(
        mock_patcher, keys,
    )
    merged = chunked_eval_override or {k: torch.randn(4, 4) for k in keys}
    sig = OpSignature(shape=(4, 4), ndim=2)

    def streaming_eval_side_effect(**kwargs):
        called_keys = kwargs.get("keys", [])
        write_fn = kwargs.get("write_fn")
        for k in called_keys:
            if k in merged:
                write_fn(k, merged[k])

    if comfy_load_side_effect is None:
        def comfy_load_side_effect(*a, **kw):
            return MagicMock(name="comfy_loaded_default")

    base_patches = {
        "nodes.exit.analyze_recipe": mock_analyze,
        "nodes.exit.analyze_recipe_models": mock_model_analysis,
        "nodes.exit.compile_plan": plan,
        "nodes.exit.compile_batch_groups": {sig: keys} if keys else {},
        "nodes.exit.compute_base_identity": "base_id",
        "nodes.exit.compute_lora_stats": {},
        "nodes.exit.validate_model_name": "model.safetensors",
        # Diffusion-only saves route through _resolve_save_path which dispatches
        # to _resolve_diffusion_models_path (not _resolve_checkpoints_path) for
        # non-checkpoint recipes.  Patch the router so diffusion-only tests
        # never reach the checkpoint resolver.
        "nodes.exit._resolve_save_path": save_path,
        "nodes.exit.validate_checkpoint_components": None,
        "nodes.exit.check_full_model_cache": False,
        "nodes.exit.check_ram_preflight": None,
        "nodes.exit.ProgressBar": None,
    }

    ctxs = []
    for target, value in base_patches.items():
        ctxs.append(patch(target, return_value=value))
    ctxs.append(patch(
        "nodes.exit.streaming_evaluation_to_sink",
        side_effect=streaming_eval_side_effect,
    ))
    ctxs.append(patch(
        "nodes.exit._comfy_load_diffusion_model",
        side_effect=comfy_load_side_effect,
    ))

    started = []
    try:
        for c in ctxs:
            c.start()
            started.append(c)
        node = WIDENExitNode()
        result = node.execute(
            recipe, save_model=True, model_name="model",
            enable_cache=enable_cache,
        )
    finally:
        for c in started:
            c.stop()
    return result


# ===========================================================================
# AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
# Diffusion-model recipe writes external Comfy-loadable key layout
# ===========================================================================


class TestDiffusionModelArtifactKeyLayout:
    """Saved artifacts from diffusion-model recipes use the external
    model.diffusion_model.* prefix Comfy's diffusion-model loader expects."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_artifact_uses_external_diffusion_model_prefix(
        self, mock_model_patcher, tmp_path,
    ):
        """Diffusion-only save writes keys with model.diffusion_model.* prefix,
        not bare diffusion_model.*, so Comfy's diffusion loader can detect them."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "diff.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())
        _run_diffusion_save(merge, mock_model_patcher, keys, save_path)

        with safe_open(save_path, framework="pt") as f:
            artifact_keys = set(f.keys())

        # No bare diffusion_model.* keys (internal format leak)
        for k in artifact_keys:
            assert not (
                k.startswith(_DIFFUSION_PREFIX)
                and not k.startswith(_EXTERNAL_DIFFUSION_PREFIX)
            ), f"Key {k!r} uses internal-only diffusion_model.* prefix"

        # Every internal key has a corresponding external key in the artifact
        expected_external = {
            _to_external_diffusion_key(k) for k in keys
        }
        assert artifact_keys == expected_external, (
            f"Expected external keys {expected_external}, got {artifact_keys}"
        )

    # AC: @full-saved-model-output ac-diffusion-model-companion-separation
    def test_artifact_excludes_companion_components(
        self, mock_model_patcher, tmp_path,
    ):
        """Diffusion-only save artifact contains no conditioner.* or
        first_stage_model.* keys — companion components stay outside."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "diff.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())
        _run_diffusion_save(merge, mock_model_patcher, keys, save_path)

        with safe_open(save_path, framework="pt") as f:
            artifact_keys = set(f.keys())

        for k in artifact_keys:
            assert not k.startswith("conditioner."), (
                f"Companion key {k!r} leaked into diffusion-only artifact"
            )
            assert not k.startswith("first_stage_model."), (
                f"Companion key {k!r} leaked into diffusion-only artifact"
            )
            assert not k.startswith("cond_stage_model."), (
                f"Companion key {k!r} leaked into diffusion-only artifact"
            )

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_artifact_metadata_records_source_model_kind(
        self, mock_model_patcher, tmp_path,
    ):
        """Saved artifact metadata declares source_model_kind=diffusion_model
        and artifact_kind=diffusion."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "diff.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())
        _run_diffusion_save(merge, mock_model_patcher, keys, save_path)

        with safe_open(save_path, framework="pt") as f:
            metadata = f.metadata()

        assert metadata is not None
        assert metadata.get("__ecaj_artifact_kind__") == "diffusion"
        assert metadata.get("__ecaj_source_model_kind__") == "diffusion_model"


# ===========================================================================
# AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
# Representative architectures: Flux, Z-Image, Qwen
# ===========================================================================


_FLUX_KEYS = (
    "diffusion_model.double_blocks.0.img_attn.qkv.weight",
    "diffusion_model.double_blocks.1.img_attn.qkv.weight",
    "diffusion_model.single_blocks.0.linear1.weight",
    "diffusion_model.img_in.weight",
)
_ZIMAGE_KEYS = (
    "diffusion_model.layers.0.attention.qkv.weight",
    "diffusion_model.layers.1.attention.qkv.weight",
    "diffusion_model.noise_refiner.0.attn.weight",
    "diffusion_model.context_refiner.0.attn.weight",
)
_QWEN_KEYS = (
    "diffusion_model.transformer_blocks.0.attn2.to_k.weight",
    "diffusion_model.transformer_blocks.1.attn2.to_k.weight",
    "diffusion_model.x_embedder.proj.weight",
)


@pytest.mark.parametrize(
    "arch_name,arch_keys",
    [("flux", _FLUX_KEYS), ("zimage", _ZIMAGE_KEYS), ("qwen", _QWEN_KEYS)],
)
class TestDiffusionTransformerKeyLayout:
    """Representative diffusion-transformer key layouts round-trip through
    the external prefix."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_external_prefix_for_arch(self, arch_name, arch_keys, tmp_path):
        """Each representative arch state dict ends up with model.diffusion_model.*
        keys that match the source bare-key suffix."""
        patcher = _make_diffusion_only_patcher(arch_keys)
        base = RecipeBase(
            model_patcher=patcher, arch=arch_name, checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / f"{arch_name}.safetensors")
        keys = list(patcher.model_state_dict().keys())
        _run_diffusion_save(merge, patcher, keys, save_path)

        with safe_open(save_path, framework="pt") as f:
            artifact_keys = set(f.keys())

        for k in arch_keys:
            assert _to_external_diffusion_key(k) in artifact_keys
            # bare suffix shows up after stripping model.diffusion_model.
            bare = k[len(_DIFFUSION_PREFIX):]
            assert any(
                ak == _EXTERNAL_DIFFUSION_PREFIX + bare for ak in artifact_keys
            )


# ===========================================================================
# AC: @full-saved-model-output ac-return-loaded-model
# Returned MODEL comes from comfy.sd.load_diffusion_model
# ===========================================================================


class TestReturnedModelUsesComfyLoader:
    """The MODEL returned by Exit after a diffusion-model save (cache miss
    or hit) is produced by comfy.sd.load_diffusion_model, not by an
    in-memory deepcopy of the WIDEN merge payload."""

    # AC: @full-saved-model-output ac-return-loaded-model
    def test_post_save_returns_comfy_loaded_model(self, mock_model_patcher, tmp_path):
        """After a fresh diffusion-only save, the returned MODEL is the
        result of comfy.sd.load_diffusion_model called with the saved path."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "diff.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())

        sentinel_model = MagicMock(name="comfy_loaded_model")
        loader_calls: list[str] = []

        def fake_load(path, model_options=None, **kwargs):
            loader_calls.append(path)
            return sentinel_model

        result = _run_diffusion_save(
            merge, mock_model_patcher, keys, save_path,
            comfy_load_side_effect=fake_load,
        )

        assert result == (sentinel_model,)
        assert loader_calls == [save_path]

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    # AC: @full-saved-model-output ac-return-loaded-model
    def test_cache_hit_uses_comfy_loader(self, mock_model_patcher, tmp_path):
        """A cache hit returns a Comfy-loaded MODEL — comfy.sd.load_diffusion_model
        is invoked, no in-memory merge runs."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "cached.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())

        # Pre-populate a valid diffusion-model cache artifact
        external_tensors = {
            _to_external_diffusion_key(k): torch.ones(4, 4) * 9.0 for k in keys
        }
        dep_fps = json.dumps({}, sort_keys=True, separators=(",", ":"))
        save_file(
            external_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "match",
                "__ecaj_affected_keys__": json.dumps(keys[:1]),
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
                "__ecaj_base_identity__": "base_id",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )

        sentinel_model = MagicMock(name="cached_comfy_model")

        with (
            patch("nodes.exit.validate_model_name", return_value="cached.safetensors"),
            # Diffusion-only cache hit routes through _resolve_save_path,
            # which dispatches to _resolve_diffusion_models_path.  Patching
            # the router keeps this test off the real folder_paths config.
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="match"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch(
                "nodes.exit._comfy_load_diffusion_model", return_value=sentinel_model,
            ) as mock_load,
        ):
            (result,) = WIDENExitNode().execute(
                merge, save_model=True, model_name="cached",
            )

        # Cache hit: no GPU pipeline ran
        mock_analyze.assert_not_called()
        # Comfy loader produced the returned MODEL
        mock_load.assert_called_once()
        assert mock_load.call_args.args[0] == save_path
        assert result is sentinel_model


# ===========================================================================
# AC: @full-saved-model-output ac-cache-reuses-artifact
# AC: cache rejects mismatched source kind / artifact kind / metadata
# ===========================================================================


class TestDiffusionCacheValidation:
    """Diffusion-model cache hits require matching artifact kind, source model
    kind, base identity, and dependency fingerprints."""

    # AC: @full-saved-model-output ac-cache-reuses-artifact
    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_diffusion_cache_hit_requires_source_model_kind(self, tmp_path):
        """check_full_model_cache rejects a 'diffusion' artifact whose
        source_model_kind metadata is missing or mismatched."""
        save_path = str(tmp_path / "model.safetensors")
        external_keys = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        }
        manifest = {
            "model.diffusion_model.input_blocks.0.0.weight": (
                torch.float32, (4, 4),
            ),
        }
        dep_fps = json.dumps({})

        # Missing source_model_kind
        save_file(
            external_keys,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "h",
                "__ecaj_affected_keys__": "[]",
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_base_identity__": "b",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )
        assert check_full_model_cache(
            save_path, "h", expected_manifest=manifest,
            expected_artifact_kind="diffusion",
            expected_base_identity="b",
            expected_dependency_fingerprints=dep_fps,
            expected_source_model_kind="diffusion_model",
        ) is False

        os.remove(save_path)

        # Wrong source_model_kind
        save_file(
            external_keys,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "h",
                "__ecaj_affected_keys__": "[]",
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "checkpoint",
                "__ecaj_base_identity__": "b",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )
        assert check_full_model_cache(
            save_path, "h", expected_manifest=manifest,
            expected_artifact_kind="diffusion",
            expected_base_identity="b",
            expected_dependency_fingerprints=dep_fps,
            expected_source_model_kind="diffusion_model",
        ) is False

    # AC: @full-saved-model-output ac-cache-reuses-artifact
    def test_diffusion_cache_hit_accepts_matching_metadata(self, tmp_path):
        """check_full_model_cache accepts a diffusion artifact whose metadata
        matches all expected fields including source_model_kind."""
        save_path = str(tmp_path / "model.safetensors")
        external_keys = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        }
        manifest = {
            "model.diffusion_model.input_blocks.0.0.weight": (
                torch.float32, (4, 4),
            ),
        }
        dep_fps = json.dumps({})

        save_file(
            external_keys,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "h",
                "__ecaj_affected_keys__": "[]",
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
                "__ecaj_base_identity__": "b",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )
        assert check_full_model_cache(
            save_path, "h", expected_manifest=manifest,
            expected_artifact_kind="diffusion",
            expected_base_identity="b",
            expected_dependency_fingerprints=dep_fps,
            expected_source_model_kind="diffusion_model",
        ) is True

    # AC: @full-saved-model-output ac-cache-reuses-artifact
    def test_diffusion_cache_rejects_checkpoint_artifact_kind(self, tmp_path):
        """A checkpoint-kind artifact must not be accepted as a diffusion-model
        cache hit, even if other metadata matches."""
        save_path = str(tmp_path / "model.safetensors")
        external_keys = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "conditioner.embedders.0.weight": torch.randn(4, 4),
            "first_stage_model.decoder.weight": torch.randn(4, 4),
        }
        dep_fps = json.dumps({})

        save_file(
            external_keys,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "h",
                "__ecaj_affected_keys__": "[]",
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "checkpoint",
                "__ecaj_source_model_kind__": "checkpoint",
                "__ecaj_base_identity__": "b",
                "__ecaj_dependency_fingerprints__": dep_fps,
                "__ecaj_checkpoint_components__": "true",
            },
        )
        assert check_full_model_cache(
            save_path, "h",
            expected_artifact_kind="diffusion",
            expected_base_identity="b",
            expected_dependency_fingerprints=dep_fps,
            expected_source_model_kind="diffusion_model",
        ) is False


# ===========================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Incomplete diffusion-model writes are not published or reused
# ===========================================================================


class TestIncompleteWritesNotReused:
    """Incomplete diffusion-model materialization must not produce a reusable
    cache artifact."""

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_failed_diffusion_write_does_not_publish(
        self, mock_model_patcher, tmp_path,
    ):
        """When streaming evaluation raises mid-stream, no artifact survives
        at save_path and no temp file is left behind."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "fail.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())

        mock_analyze, mock_model_analysis, mock_loader, plan = _make_full_mode_mocks(
            mock_model_patcher, keys,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        def boom(**kwargs):
            raise RuntimeError("fail mid-stream")

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=boom),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="fail.safetensors"),
            # Diffusion-only failure path still uses _resolve_save_path —
            # patch the router so the test never touches the real
            # diffusion_models / unet folder configuration.
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            with pytest.raises(RuntimeError, match="fail mid-stream"):
                WIDENExitNode().execute(
                    merge, save_model=True, model_name="fail",
                )

        assert not os.path.exists(save_path)
        leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert leftovers == []


# ===========================================================================
# AC: @streaming-full-model-materialization ac-raw-byte-dtype-preservation
# BF16 raw bytes preserved across the new key remap
# ===========================================================================


class TestBF16RoundTripWithExternalKeys:
    """BF16 tensors retain bitwise identity across the diffusion-model save
    path even though keys are remapped to model.diffusion_model.*."""

    # AC: @streaming-full-model-materialization ac-raw-byte-dtype-preservation
    def test_bf16_bytes_preserved(self, tmp_path):
        """BF16 inputs end up in the artifact under model.diffusion_model.*
        with original dtype, shape, and bitwise-equivalent bytes."""
        bf16_tensor = torch.randn(4, 4, dtype=torch.float32).to(torch.bfloat16)
        sd = {"diffusion_model.bf16_weight": bf16_tensor}

        patcher = MockModelPatcher.__new__(MockModelPatcher)
        patcher._state_dict = sd
        patcher.model = MagicMock()
        patcher.model.diffusion_model = MagicMock()
        patcher.model.diffusion_model.state_dict = MagicMock(return_value={
            k.removeprefix(_DIFFUSION_PREFIX): v for k, v in sd.items()
        })
        patcher.patches = {}
        patcher.patches_uuid = uuid.uuid4()

        def _clone():
            c = MockModelPatcher.__new__(MockModelPatcher)
            c._state_dict = patcher._state_dict
            c.model = patcher.model
            c.patches = {}
            c.patches_uuid = patcher.patches_uuid
            return c
        patcher.clone = _clone

        base = RecipeBase(
            model_patcher=patcher, arch="sdxl", checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "bf16.safetensors")
        keys = list(sd.keys())
        merged = {keys[0]: bf16_tensor.clone()}
        _run_diffusion_save(
            merge, patcher, keys, save_path, chunked_eval_override=merged,
        )

        with safe_open(save_path, framework="pt") as f:
            external_key = "model.diffusion_model.bf16_weight"
            loaded = f.get_tensor(external_key)

        assert loaded.dtype == torch.bfloat16
        assert tuple(loaded.shape) == (4, 4)
        # Bitwise equivalence: byte-for-byte
        assert (
            loaded.view(torch.uint8).reshape(-1).tolist()
            == bf16_tensor.view(torch.uint8).reshape(-1).tolist()
        )


# ===========================================================================
# AC: @full-saved-model-output (branching: diffusion vs checkpoint)
# ===========================================================================


class TestRoutingByRecipeSourceKind:
    """Diffusion-model recipes route to the diffusion-model save branch.
    Checkpoint-style recipes continue routing to the checkpoint branch."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_diffusion_only_recipe_does_not_invoke_save_comfy_checkpoint(
        self, mock_model_patcher, tmp_path,
    ):
        """A diffusion-only recipe must NOT call save_comfy_checkpoint —
        checkpoint save semantics belong to checkpoint-style recipes."""
        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "model.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())

        with patch("nodes.exit.save_comfy_checkpoint") as mock_save_ckpt:
            _run_diffusion_save(merge, mock_model_patcher, keys, save_path)

        mock_save_ckpt.assert_not_called()


# ===========================================================================
# Helper unit tests
# ===========================================================================


class TestKeyRemapHelper:
    """_to_external_diffusion_key transforms internal keys to the external
    Comfy-loadable layout."""

    def test_internal_diffusion_key_to_external(self):
        assert _to_external_diffusion_key("diffusion_model.foo.bar") == (
            "model.diffusion_model.foo.bar"
        )

    def test_already_external_unchanged(self):
        assert _to_external_diffusion_key(
            "model.diffusion_model.foo.bar"
        ) == "model.diffusion_model.foo.bar"

    def test_unknown_prefix_unchanged(self):
        assert _to_external_diffusion_key("conditioner.foo") == "conditioner.foo"


class TestClassifyTempArtifactDiffusion:
    """_classify_temp_artifact rejects diffusion artifacts that contain
    internal-format diffusion_model.* keys (no external prefix)."""

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_internal_format_rejected_for_diffusion_kind(self, tmp_path):
        """A temp artifact tagged kind=diffusion that contains bare
        diffusion_model.* keys (no model. prefix) is rejected."""
        path = str(tmp_path / "tmp.safetensors")
        save_file(
            {"diffusion_model.x": torch.randn(2, 2)},
            path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
            },
        )
        with pytest.raises(RuntimeError, match=r"diffusion"):
            _classify_temp_artifact(path, expected_kind="diffusion")

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    def test_external_format_accepted_for_diffusion_kind(self, tmp_path):
        """A temp artifact tagged kind=diffusion with model.diffusion_model.*
        keys is accepted (no exception)."""
        path = str(tmp_path / "tmp.safetensors")
        save_file(
            {"model.diffusion_model.x": torch.randn(2, 2)},
            path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
            },
        )
        # Should not raise
        _classify_temp_artifact(path, expected_kind="diffusion")


class TestComfyLoaderHelper:
    """_load_diffusion_model_artifact delegates to comfy.sd.load_diffusion_model."""

    # AC: @full-saved-model-output ac-return-loaded-model
    def test_calls_comfy_load_diffusion_model(self, tmp_path):
        path = str(tmp_path / "x.safetensors")
        sentinel = MagicMock(name="comfy_model")
        with patch(
            "nodes.exit._comfy_load_diffusion_model", return_value=sentinel,
        ) as mock_loader:
            result = _load_diffusion_model_artifact(path)
        assert result is sentinel
        mock_loader.assert_called_once()
        assert mock_loader.call_args.args[0] == path


# ===========================================================================
# AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
# Diffusion-model save path is resolved through Comfy's diffusion_models
# folder (UNETLoader discovery), not the checkpoints folder.  These tests
# deliberately do NOT patch _resolve_checkpoints_path: a diffusion save that
# silently routes through the checkpoint resolver is the bug they catch.
# ===========================================================================


class TestDiffusionModelsPathResolver:
    """_resolve_diffusion_models_path uses folder_paths('diffusion_models')."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_uses_diffusion_models_folder(self, monkeypatch, tmp_path):
        """Resolver returns a path under the first diffusion_models directory."""
        import folder_paths

        diffusion_dir = tmp_path / "diffusion_models"
        diffusion_dir.mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return [str(diffusion_dir)]
            if folder == "checkpoints":
                return [str(ckpt_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)
        path = _resolve_diffusion_models_path("model.safetensors")
        assert path == os.path.join(str(diffusion_dir), "model.safetensors")
        # Critically, the diffusion resolver MUST NOT publish under the
        # checkpoints folder — Comfy's standalone diffusion-model loader
        # discovers files via folder_paths('diffusion_models').
        assert str(ckpt_dir) not in path

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_falls_back_to_unet_for_legacy_layouts(self, monkeypatch, tmp_path):
        """Older ComfyUI installs use 'unet' for the same content.  When
        'diffusion_models' is not configured, the resolver falls back to
        'unet' so the saved artifact is still discoverable."""
        import folder_paths

        unet_dir = tmp_path / "unet"
        unet_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return []
            if folder == "unet":
                return [str(unet_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)
        path = _resolve_diffusion_models_path("model.safetensors")
        assert path == os.path.join(str(unet_dir), "model.safetensors")

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_raises_when_no_directory_configured(self, monkeypatch):
        """If neither 'diffusion_models' nor 'unet' is configured, the
        resolver raises ValueError so callers cannot accidentally publish to
        the wrong folder."""
        import folder_paths

        monkeypatch.setattr(
            folder_paths, "get_folder_paths", lambda folder: [],
        )
        with pytest.raises(ValueError, match=r"diffusion_models"):
            _resolve_diffusion_models_path("model.safetensors")


class TestSaveKindAwarePathRouter:
    """_resolve_save_path picks the resolver matching the artifact kind."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_diffusion_save_uses_diffusion_models_resolver(
        self, monkeypatch, tmp_path,
    ):
        """is_checkpoint=False routes through the diffusion-models resolver,
        not the checkpoints resolver — this is the bug from review."""
        import folder_paths

        diffusion_dir = tmp_path / "diffusion_models"
        diffusion_dir.mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return [str(diffusion_dir)]
            if folder == "checkpoints":
                return [str(ckpt_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)
        path = _resolve_save_path("model.safetensors", is_checkpoint=False)
        assert path.startswith(str(diffusion_dir) + os.sep)
        assert str(ckpt_dir) not in path

    # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    def test_checkpoint_save_uses_checkpoints_resolver(
        self, monkeypatch, tmp_path,
    ):
        """is_checkpoint=True still routes through the checkpoints resolver."""
        import folder_paths

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        monkeypatch.setattr(
            folder_paths, "get_folder_paths",
            lambda folder: [str(ckpt_dir)] if folder == "checkpoints" else [],
        )
        path = _resolve_save_path("model.safetensors", is_checkpoint=True)
        assert path.startswith(str(ckpt_dir) + os.sep)


class TestDiffusionSaveRoutingThroughFolderPaths:
    """End-to-end: diffusion save_model routes through folder_paths
    'diffusion_models' (not 'checkpoints'), without any test patching the
    checkpoint resolver."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_fresh_save_publishes_under_diffusion_models_folder(
        self, mock_model_patcher, tmp_path, monkeypatch,
    ):
        """A diffusion-only save_model run resolves the artifact path through
        folder_paths('diffusion_models').  We configure folder_paths to
        return distinct directories for 'diffusion_models' and 'checkpoints'
        and assert the artifact lands under diffusion_models."""
        import folder_paths

        diffusion_dir = tmp_path / "diffusion_models"
        diffusion_dir.mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return [str(diffusion_dir)]
            if folder == "checkpoints":
                return [str(ckpt_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)

        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        mock_analyze, mock_model_analysis, mock_loader, plan = _make_full_mode_mocks(
            mock_model_patcher, keys,
        )
        merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval_side_effect(**kwargs):
            for k in kwargs.get("keys", []):
                kwargs["write_fn"](k, merged[k])

        # Note: we do NOT patch _resolve_checkpoints_path or _resolve_save_path.
        # The system under test must route the diffusion save through
        # _resolve_diffusion_models_path → folder_paths('diffusion_models').
        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink",
                  side_effect=streaming_eval_side_effect),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._comfy_load_diffusion_model",
                  return_value=MagicMock(name="loaded")),
        ):
            WIDENExitNode().execute(
                merge, save_model=True, model_name="diffusion_only",
            )

        expected_path = diffusion_dir / "diffusion_only.safetensors"
        assert expected_path.exists(), (
            f"diffusion save did not publish under diffusion_models folder; "
            f"contents: {list(diffusion_dir.iterdir())} / "
            f"{list(ckpt_dir.iterdir())}"
        )
        # Artifact must NOT have been written to the checkpoints folder.
        assert not (ckpt_dir / "diffusion_only.safetensors").exists()

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    # AC: @full-saved-model-output ac-no-op-produces-full-artifact
    def test_noop_diffusion_save_publishes_under_diffusion_models_folder(
        self, mock_model_patcher, tmp_path, monkeypatch,
    ):
        """A no-op diffusion save (RecipeBase, no merge) also publishes under
        the diffusion_models folder, not under checkpoints."""
        import folder_paths

        diffusion_dir = tmp_path / "diffusion_models"
        diffusion_dir.mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return [str(diffusion_dir)]
            if folder == "checkpoints":
                return [str(ckpt_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)

        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )

        with (
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._comfy_load_diffusion_model",
                  return_value=MagicMock(name="loaded")),
        ):
            WIDENExitNode().execute(
                base, save_model=True, model_name="diffusion_noop",
            )

        assert (diffusion_dir / "diffusion_noop.safetensors").exists()
        assert not (ckpt_dir / "diffusion_noop.safetensors").exists()

    # AC: @full-saved-model-output ac-cache-reuses-artifact
    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_diffusion_cache_hit_resolves_through_diffusion_models_folder(
        self, mock_model_patcher, tmp_path, monkeypatch,
    ):
        """Diffusion cache validation must look in the diffusion_models
        folder.  We pre-place a valid artifact there and assert that
        check_full_model_cache is invoked with that path (so cache reuse
        cannot accidentally reach into the checkpoints folder)."""
        import folder_paths

        diffusion_dir = tmp_path / "diffusion_models"
        diffusion_dir.mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        def get_folder_paths(folder: str):
            if folder == "diffusion_models":
                return [str(diffusion_dir)]
            if folder == "checkpoints":
                return [str(ckpt_dir)]
            return []

        monkeypatch.setattr(folder_paths, "get_folder_paths", get_folder_paths)

        base = RecipeBase(
            model_patcher=mock_model_patcher, arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        keys = list(mock_model_patcher.model_state_dict().keys())

        mock_analyze, mock_model_analysis, mock_loader, plan = _make_full_mode_mocks(
            mock_model_patcher, keys,
        )

        sentinel = MagicMock(name="cached_model")
        captured_paths: list[str] = []

        def fake_check(save_path, *args, **kwargs):
            captured_paths.append(save_path)
            return True

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models",
                  return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=plan),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", side_effect=fake_check),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit._comfy_load_diffusion_model", return_value=sentinel),
        ):
            (result,) = WIDENExitNode().execute(
                merge, save_model=True, model_name="cached_artifact",
            )

        assert result is sentinel
        assert len(captured_paths) == 1
        cache_path = captured_paths[0]
        assert cache_path.startswith(str(diffusion_dir) + os.sep), (
            f"Diffusion cache check resolved path under {cache_path!r}; "
            f"expected under {str(diffusion_dir)!r} (diffusion_models folder)"
        )
        assert str(ckpt_dir) not in cache_path
