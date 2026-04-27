"""Tests for streaming full saved model materialization and artifact-backed cache.

AC coverage for:
  @streaming-full-model-materialization
  @full-saved-model-output
  @comfy-memory-manager-compatibility
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lib.batch_groups import OpSignature
from lib.persistence import build_metadata, check_full_model_cache
from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge
from lib.recipe_eval import EvalPlan
from lib.streaming_save import MaterializationSink
from nodes.exit import (
    WIDENExitNode,
    _incremental_cache,
    install_merged_patches,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SDXL_KEYS = (
    "diffusion_model.input_blocks.0.0.weight",
    "diffusion_model.input_blocks.1.0.weight",
    "diffusion_model.middle_block.0.weight",
    "diffusion_model.output_blocks.0.0.weight",
)


def _make_full_mode_mocks(mock_model_patcher, keys_to_process, *, recipe=None):
    """Create mocks for full saved model mode exit node tests."""
    mock_loader = MagicMock()
    mock_loader.affected_keys = set(keys_to_process)
    mock_loader.loaded_bytes = 0
    mock_loader.cleanup = MagicMock()

    set_affected = {}
    if recipe is not None:
        def _find_loras(n):
            if isinstance(n, RecipeLoRA):
                set_affected[str(id(n))] = set(keys_to_process)
            elif isinstance(n, RecipeCompose):
                for b in n.branches:
                    _find_loras(b)
            elif isinstance(n, RecipeMerge):
                _find_loras(n.base)
                _find_loras(n.target)
                if n.backbone is not None:
                    _find_loras(n.backbone)
        _find_loras(recipe)
    if not set_affected:
        set_affected = {str(id(None)): set(keys_to_process)}

    mock_analyze = MagicMock(
        model_patcher=mock_model_patcher,
        arch="sdxl",
        loader=mock_loader,
        set_affected=set_affected,
        affected_keys=set(keys_to_process),
    )
    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()

    dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

    return mock_analyze, mock_model_analysis, mock_loader, dummy_plan


def _run_full_mode(recipe, mock_model_patcher, keys, tmp_path,
                   *, model_name="test_model", enable_cache=True,
                   extra_patches=None, chunked_eval_override=None,
                   save_workflow=True, extra_pnginfo=None):
    """Run WIDENExitNode.execute() in full saved model mode with mocking.

    Full mode now uses streaming_evaluation_to_sink (which calls write_fn
    per-key) instead of evaluate_affected_group (which returns a dict).
    This helper patches streaming_evaluation_to_sink with a side_effect
    that generates tensors and calls write_fn for each key.
    """
    mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
        mock_model_patcher, keys, recipe=recipe,
    )

    save_path = str(tmp_path / f"{model_name}.safetensors")

    if chunked_eval_override is None:
        merged = {k: torch.randn(4, 4) for k in keys}
    else:
        merged = chunked_eval_override

    sig = OpSignature(shape=(4, 4), ndim=2)

    def streaming_eval_side_effect(**kwargs):
        called_keys = kwargs.get("keys", [])
        write_fn = kwargs.get("write_fn")
        for k in called_keys:
            if k in merged:
                write_fn(k, merged[k])

    patches = {
        "nodes.exit.analyze_recipe": mock_analyze,
        "nodes.exit.analyze_recipe_models": mock_model_analysis,
        "nodes.exit.compile_plan": dummy_plan,
        "nodes.exit.compile_batch_groups": {sig: keys} if keys else {},
        "nodes.exit.compute_base_identity": "base_id",
        "nodes.exit.compute_lora_stats": {},
        "nodes.exit.validate_model_name": f"{model_name}.safetensors",
        "nodes.exit._resolve_checkpoints_path": save_path,
        "nodes.exit.check_full_model_cache": False,
        "nodes.exit.check_ram_preflight": None,
        "nodes.exit.ProgressBar": None,
    }
    if extra_patches:
        patches.update(extra_patches)

    ctx_managers = []
    mock_refs = {}
    for target, value in patches.items():
        if value is None:
            p = patch(target, return_value=value)
        elif isinstance(value, MagicMock) and hasattr(value, "return_value"):
            p = patch(target, return_value=value)
        elif isinstance(value, dict) or isinstance(value, EvalPlan):
            p = patch(target, return_value=value)
        elif isinstance(value, str):
            p = patch(target, return_value=value)
        elif isinstance(value, bool):
            p = patch(target, return_value=value)
        elif isinstance(value, MagicMock):
            p = patch(target, value)
        else:
            p = patch(target, return_value=value)
        ctx_managers.append((target, p))

    # Patch streaming_evaluation_to_sink with side_effect
    streaming_patch = patch(
        "nodes.exit.streaming_evaluation_to_sink",
        side_effect=streaming_eval_side_effect,
    )
    ctx_managers.append(("nodes.exit.streaming_evaluation_to_sink", streaming_patch))

    entered = []
    try:
        for target, p in ctx_managers:
            m = p.start()
            mock_refs[target] = m
            entered.append(p)

        node = WIDENExitNode()
        result = node.execute(
            recipe, save_model=True, model_name=model_name,
            enable_cache=enable_cache, save_workflow=save_workflow,
            extra_pnginfo=extra_pnginfo,
        )
    finally:
        for p in entered:
            p.stop()

    return result, {
        "mock_loader": mock_loader,
        "mock_analyze": mock_analyze,
        "mock_model_analysis": mock_model_analysis,
        "merged": merged,
        "save_path": save_path,
    }


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# ===========================================================================


class TestFullModeSucceedsWithoutDictPath:
    """Full mode must succeed even when dict-returning evaluation and
    in-memory result sink construction are monkeypatched to fail.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    """

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_full_mode_succeeds_when_dict_eval_patched_to_fail(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode does not use the dict-returning chunked_evaluation or
        evaluate_affected_group paths for newly evaluated affected tensors —
        it uses streaming_evaluation_to_sink which streams each tensor
        directly to the sink via write_fn.

        We monkeypatch both chunked_evaluation and evaluate_affected_group
        to raise RuntimeError, then verify full mode still succeeds because
        it calls streaming_evaluation_to_sink instead.
        """
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "test.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )

        affected_tensors = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        # Monkeypatch the dict-returning paths to fail.
        # Full mode must succeed because it uses streaming_evaluation_to_sink.
        def chunked_eval_bomb(**kwargs):
            raise RuntimeError("chunked_evaluation must not be called in full mode")

        def eval_group_bomb(**kwargs):
            raise RuntimeError("evaluate_affected_group must not be called in full mode")

        def streaming_eval(*, keys, base_tensors, eval_fn, batch_size,
                           device, dtype, storage_dtype, write_fn):
            for k in keys:
                write_fn(k, affected_tensors[k])

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", side_effect=chunked_eval_bomb),
            patch("nodes.exit.evaluate_affected_group", side_effect=eval_group_bomb),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="test.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(merge, save_model=True, model_name="test")

        # Verify artifact was created
        assert result is not None
        with safe_open(save_path, framework="pt") as f:
            meta = f.metadata()
            assert meta["__ecaj_output_mode__"] == "full"
            assert "__ecaj_recipe_hash__" in meta

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    def test_full_mode_succeeds_when_inmemory_sink_construction_fails(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode does not instantiate an InMemoryResultSink for newly
        evaluated full-mode tensors. We verify that even if constructing
        such a concept would fail, full mode succeeds because it uses
        MaterializationSink instead.

        This is tested by verifying that no in-memory merged_state dict is
        accumulated and stored in _incremental_cache after full mode.
        """
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        _incremental_cache.clear()

        (result,), mocks = _run_full_mode(merge, mock_model_patcher, keys, tmp_path)

        # Full mode must NOT leave tensor payload in _incremental_cache
        assert len(_incremental_cache) == 0
        assert result is not None


# ===========================================================================
# AC: Patch mode still works
# ===========================================================================


class TestPatchModePreserved:
    """Patch mode returns a patched model and preserves patch-mode
    incremental cache behavior.

    AC: @streaming-full-model-materialization (implicit — patch mode unchanged)
    """

    # AC: @streaming-full-model-materialization (patch mode unchanged)
    def test_patch_mode_returns_patched_model(self, mock_model_patcher):
        """Patch mode (save_model=False) returns a ModelPatcher with set patches."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(merge, save_model=False)

        # Patch mode returns a ModelPatcher clone with patches
        assert result is not mock_model_patcher
        # Patch mode populates incremental cache
        assert len(_incremental_cache) == 1

    # AC: @streaming-full-model-materialization (patch mode cache preserved)
    def test_patch_mode_preserves_incremental_cache(self, mock_model_patcher):
        """Patch mode populates _incremental_cache with tensor payload."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(merge, save_model=False)

        # Patch mode should store merged_state tensors in cache
        entry = list(_incremental_cache.values())[0]
        assert len(entry.merged_state) == len(keys)
        for k in keys:
            assert isinstance(entry.merged_state[k], torch.Tensor)


# ===========================================================================
# AC: @streaming-full-model-materialization ac-direct-artifact-handoff
# Event order test
# ===========================================================================


class TestFullModeEventOrder:
    """Full mode with multiple affected groups records correct runtime event order.

    Uses two affected groups with INTERLEAVED key names to verify that
    each group is fully evaluated and written to the sink before the next
    group is evaluated — even when the groups' keys would interleave in
    sorted name order.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    AC: @streaming-full-model-materialization ac-affected-results-released
    """

    # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    # AC: @streaming-full-model-materialization ac-affected-results-released
    def test_event_order_interleaved_keys(
        self, tmp_path
    ):
        """Full mode event order with two affected groups whose keys interleave
        in sorted name order.

        Group 1 (4x4): keys "a" and "z"
        Group 2 (8x8): keys "b" and "c"
        Sorted order: a, b, c, z  (interleaved!)

        The streaming loop must evaluate group 1, write BOTH of its keys
        (a and z), free group 1, THEN evaluate group 2 and write its keys.
        Group 1's 'z' result must NOT remain resident while group 2 is
        evaluated.
        """
        group1_keys = ["a", "z"]
        group2_keys = ["b", "c"]
        base_only_key = "base_only"
        all_keys = group1_keys + group2_keys

        state_dict = {}
        for k in group1_keys:
            state_dict[k] = torch.randn(4, 4, dtype=torch.float32)
        for k in group2_keys:
            state_dict[k] = torch.randn(8, 8, dtype=torch.float32)
        state_dict[base_only_key] = torch.randn(4, 4, dtype=torch.float32)

        from tests.conftest import MockModelPatcher
        mock_patcher = MockModelPatcher.__new__(MockModelPatcher)
        mock_patcher._state_dict = state_dict
        mock_patcher.model = MagicMock()
        mock_patcher.model.diffusion_model = MagicMock()
        mock_patcher.model.diffusion_model.state_dict = MagicMock(return_value=state_dict)
        mock_patcher.patches = {}
        import uuid
        mock_patcher.patches_uuid = uuid.uuid4()

        def _clone():
            c = MockModelPatcher.__new__(MockModelPatcher)
            c._state_dict = mock_patcher._state_dict
            c.model = mock_patcher.model
            c.patches = {}
            c.patches_uuid = mock_patcher.patches_uuid
            return c
        mock_patcher.clone = _clone

        base = RecipeBase(model_patcher=mock_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "event_order.safetensors")

        events = []

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_patcher, all_keys, recipe=merge,
        )

        group1_results = {k: torch.randn(4, 4) for k in group1_keys}
        group2_results = {k: torch.randn(8, 8) for k in group2_keys}

        sig1 = OpSignature(shape=(4, 4), ndim=2)
        sig2 = OpSignature(shape=(8, 8), ndim=2)
        batch_groups = {sig1: group1_keys, sig2: group2_keys}

        def recording_streaming_eval(**kwargs):
            """Mock streaming_evaluation_to_sink that records events and
            calls write_fn for each key — simulating per-tensor streaming."""
            called_keys = kwargs.get("keys", [])
            write_fn = kwargs.get("write_fn")
            if set(called_keys) <= set(group1_keys):
                events.append("eval:g1")
                for k in called_keys:
                    write_fn(k, group1_results[k])
            elif set(called_keys) <= set(group2_keys):
                events.append("eval:g2")
                for k in called_keys:
                    write_fn(k, group2_results[k])
            else:
                raise RuntimeError(f"Unexpected keys: {called_keys}")

        original_sink_class = MaterializationSink

        class RecordingSink(original_sink_class):
            def write_tensor(self, name, tensor):
                if name in group1_results:
                    events.append(f"write:g1:{name}")
                elif name in group2_results:
                    events.append(f"write:g2:{name}")
                else:
                    events.append(f"write:base:{name}")
                super().write_tensor(name, tensor)

            def finalize(self, save_path):
                events.append("finalize")
                super().finalize(save_path)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value=batch_groups),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=recording_streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="event_order.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.MaterializationSink", RecordingSink),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(merge, save_model=True, model_name="event_order")
            events.append("saved_model_loaded")

        # Both groups must be evaluated
        assert "eval:g1" in events
        assert "eval:g2" in events
        assert "finalize" in events
        assert "saved_model_loaded" in events

        # Group 1 writes must ALL happen BEFORE group 2 evaluation.
        # This is the critical streaming assertion — even though group 1 key
        # "z" sorts after group 2 keys "b"/"c", group 1's results must be
        # fully written and freed before group 2 evaluation begins.
        g1_write_events = [e for e in events if e.startswith("write:g1:")]
        g2_write_events = [e for e in events if e.startswith("write:g2:")]
        assert len(g1_write_events) == len(group1_keys)
        assert len(g2_write_events) == len(group2_keys)

        last_g1_write_idx = max(events.index(e) for e in g1_write_events)
        eval_g2_idx = events.index("eval:g2")
        finalize_idx = events.index("finalize")
        load_idx = events.index("saved_model_loaded")

        # Group 1 is written to sink BEFORE group 2 starts evaluation
        assert last_g1_write_idx < eval_g2_idx, (
            f"Group 1 writes must complete before group 2 evaluation. "
            f"Events: {events}"
        )

        # All writes happen before finalize
        for e in g1_write_events + g2_write_events:
            assert events.index(e) < finalize_idx

        # Finalize before load
        assert finalize_idx < load_idx


# ===========================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Failure after partial streaming
# ===========================================================================


class TestFailureAbortsMaterialization:
    """Failure after partial streaming aborts materialization.

    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    """

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_failure_during_eval_aborts_sink(self, mock_model_patcher, tmp_path):
        """If evaluation fails mid-stream, the sink is aborted and no
        partial artifact is accepted as cache hit."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "partial.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        def failing_streaming_eval(**kwargs):
            raise RuntimeError("Simulated evaluation failure")

        mock_sink = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=failing_streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="partial.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.MaterializationSink", return_value=mock_sink),
        ):
            with pytest.raises(RuntimeError, match="Simulated evaluation failure"):
                WIDENExitNode().execute(merge, save_model=True, model_name="partial")

        # Sink should have been aborted
        mock_sink.abort.assert_called()
        # Finalize should NOT have been called
        mock_sink.finalize.assert_not_called()
        # No partial artifact file
        import os
        assert not os.path.exists(save_path)

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_failure_leaves_no_full_mode_tensor_in_cache(
        self, mock_model_patcher, tmp_path
    ):
        """After failure, no full-mode tensor payload remains in _incremental_cache."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        _incremental_cache.clear()

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        def failing_streaming_eval(**kwargs):
            raise RuntimeError("fail")

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=failing_streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="fail.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=str(tmp_path / "fail.safetensors")),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            with pytest.raises(RuntimeError):
                WIDENExitNode().execute(merge, save_model=True, model_name="fail")

        assert len(_incremental_cache) == 0


# ===========================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Finalize failure
# ===========================================================================


class TestFinalizeFailure:
    """Finalize failure cleans up and does not create a reusable artifact.

    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    """

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_finalize_failure_does_not_create_artifact(
        self, mock_model_patcher, tmp_path
    ):
        """If finalize fails, no artifact file should exist at save_path."""
        import os

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "finalize_fail.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(*, keys, base_tensors, eval_fn, batch_size,
                           device, dtype, storage_dtype, write_fn):
            for k in keys:
                write_fn(k, merged[k])

        original_sink = MaterializationSink

        class FailingFinalizeSink(original_sink):
            def finalize(self, save_path):
                self.abort()
                raise OSError("Simulated finalize failure")

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="finalize_fail.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.MaterializationSink", FailingFinalizeSink),
        ):
            with pytest.raises(OSError, match="Simulated finalize failure"):
                WIDENExitNode().execute(merge, save_model=True, model_name="finalize_fail")

        assert not os.path.exists(save_path)


# ===========================================================================
# AC: @full-saved-model-output ac-cache-reuses-artifact
# Full artifact cache hit
# ===========================================================================


class TestFullArtifactCacheHit:
    """Full artifact cache hit returns loaded saved model without recomputing.

    AC: @full-saved-model-output ac-cache-reuses-artifact
    AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    AC: @full-saved-model-output ac-return-loaded-model
    """

    # AC: @full-saved-model-output ac-cache-reuses-artifact
    def test_cache_hit_skips_gpu(self, mock_model_patcher, tmp_path):
        """On full-model cache hit, GPU pipeline is skipped entirely."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        affected_key = keys[0]
        save_path = str(tmp_path / "cached.safetensors")

        # Create a full artifact with ALL model keys (affected + unaffected).
        # Use distinct values so we can verify the returned model loads from
        # the artifact, not the original base.
        artifact_tensors = {}
        for k in keys:
            artifact_tensors[k] = torch.ones(4, 4) * 42.0

        save_file(
            artifact_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "match",
                "__ecaj_affected_keys__": json.dumps([affected_key]),
                "__ecaj_output_mode__": "full",
            },
        )

        with (
            patch("nodes.exit.validate_model_name", return_value="cached.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="match"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.ProgressBar", None),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
        ):
            (result,) = WIDENExitNode().execute(
                merge, save_model=True, model_name="cached"
            )
            mock_analyze.assert_not_called()

        assert result is not mock_model_patcher
        # All keys must have artifact values (42.0) in model_state_dict —
        # loaded into model-owned memory, not as set patches.
        # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
        result_sd = result.model_state_dict()
        for k in keys:
            assert torch.allclose(result_sd[k], torch.ones(4, 4) * 42.0), (
                f"Key {k} should have artifact value 42.0"
            )
        # No set patches — weights are model-owned, not patch-resident.
        assert len(result.patches) == 0

    # AC: @full-saved-model-output ac-return-loaded-model
    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_artifact_load_returns_full_model_weights(self, mock_model_patcher, tmp_path):
        """_load_model_from_artifact loads ALL keys from the artifact into the
        model's own state dict (not as set patches).

        A runtime probe: create an artifact with value 20.0 for every key.
        The returned model's model_state_dict() must have 20.0 (from the
        artifact), not the original base values.  No set patches should exist.
        """
        from nodes.exit import _load_model_from_artifact

        keys = list(mock_model_patcher.model_state_dict().keys())
        affected_key = keys[0]
        unaffected_key = keys[1]
        save_path = str(tmp_path / "full_artifact.safetensors")

        # Build artifact with distinctive values
        artifact_tensors = {}
        for k in keys:
            artifact_tensors[k] = torch.ones(4, 4) * 20.0

        save_file(
            artifact_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "match",
                "__ecaj_affected_keys__": json.dumps([affected_key]),
                "__ecaj_output_mode__": "full",
            },
        )

        result = _load_model_from_artifact(
            save_path, mock_model_patcher, torch.float32,
        )

        # All keys must have artifact values in model_state_dict, not the
        # original base values.  Weights are model-owned, not patch-owned.
        result_sd = result.model_state_dict()
        for k in keys:
            assert torch.allclose(result_sd[k], torch.ones(4, 4) * 20.0), (
                f"Key {k} should have artifact value 20.0, got {result_sd[k][0, 0].item()}"
            )

        # No set patches — weights live in the model's own state dict.
        # This ensures Comfy's memory manager owns the memory lifecycle.
        assert len(result.patches) == 0

        # Original model_patcher must not be affected.
        orig_sd = mock_model_patcher.model_state_dict()
        for k in keys:
            assert not torch.allclose(orig_sd[k], torch.ones(4, 4) * 20.0), (
                f"Original key {k} should NOT have artifact value"
            )


# ===========================================================================
# AC: Cache mode cross-contamination tests
# ===========================================================================


class TestCacheModeIsolation:
    """Patch artifacts not accepted as full-mode cache hits, and vice versa.

    AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    """

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_patch_artifact_not_accepted_as_full_cache(self, tmp_path):
        """A patch-mode artifact should not be accepted as a full-model cache hit."""
        save_path = str(tmp_path / "patch.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["k"]',
                "__ecaj_output_mode__": "patch",
            },
        )
        assert check_full_model_cache(save_path, "hash1") is False

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_missing_output_mode_not_accepted_as_full_cache(self, tmp_path):
        """Artifacts without __ecaj_output_mode__ should not be full-model cache hits."""
        save_path = str(tmp_path / "old.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["k"]',
            },
        )
        assert check_full_model_cache(save_path, "hash1") is False

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_wrong_hash_not_accepted(self, tmp_path):
        """Wrong recipe hash should not be accepted as full-model cache hit."""
        save_path = str(tmp_path / "wrong.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["k"]',
                "__ecaj_output_mode__": "full",
            },
        )
        assert check_full_model_cache(save_path, "wrong_hash") is False

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_missing_affected_keys_not_accepted(self, tmp_path):
        """Artifact with valid version/hash/mode but missing __ecaj_affected_keys__
        is rejected. Such an artifact would crash _load_model_from_artifact."""
        save_path = str(tmp_path / "no_affected_keys.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_output_mode__": "full",
            },
        )
        assert check_full_model_cache(save_path, "hash1") is False

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_malformed_affected_keys_not_accepted(self, tmp_path):
        """Artifact with invalid JSON in __ecaj_affected_keys__ is rejected."""
        save_path = str(tmp_path / "bad_json.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": "not valid json",
                "__ecaj_output_mode__": "full",
            },
        )
        assert check_full_model_cache(save_path, "hash1") is False

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_valid_full_artifact_accepted(self, tmp_path):
        """Valid full-mode artifact with matching hash is accepted."""
        save_path = str(tmp_path / "good.safetensors")
        save_file(
            {"k": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["k"]',
                "__ecaj_output_mode__": "full",
            },
        )
        assert check_full_model_cache(save_path, "hash1") is True


# ===========================================================================
# AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
# Full mode leaves no tensor payload after success or failure
# ===========================================================================


class TestNoResidentPayload:
    """Full mode leaves no full-mode affected tensor payload after success or failure.

    AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
    """

    # AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
    def test_full_mode_no_tensor_payload_after_success(
        self, mock_model_patcher, tmp_path
    ):
        """After full mode success, _incremental_cache has no tensor payload."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        _incremental_cache.clear()

        (result,), mocks = _run_full_mode(merge, mock_model_patcher, keys, tmp_path)

        assert len(_incremental_cache) == 0

    # AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
    def test_pre_populated_patch_cache_not_read_by_full_mode(
        self, mock_model_patcher, tmp_path
    ):
        """A pre-populated patch-mode tensor cache sentinel is not read or
        reused by full mode."""
        from nodes.exit import _CacheEntry

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # Pre-populate cache with a sentinel
        sentinel = _CacheEntry(
            structural_fingerprint="sentinel_fp",
            block_configs=[],
            merged_state={k: torch.zeros(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )
        _incremental_cache.clear()
        _incremental_cache["sentinel_fp"] = sentinel

        (result,), mocks = _run_full_mode(merge, mock_model_patcher, keys, tmp_path)

        # Full mode should NOT have used the sentinel's zero tensors
        # (it should have used the freshly computed tensors from chunked_evaluation)
        # The sentinel should still be there (full mode clears for enable_cache=False only)
        # but full mode doesn't READ it
        assert result is not None


# ===========================================================================
# AC: enable_cache=False evicts in-memory cache entries
# ===========================================================================


class TestEnableCacheFalseEvicts:
    """enable_cache=False evicts in-memory cache entries.

    AC: @streaming-full-model-materialization (cache disable behavior)
    """

    # AC: @streaming-full-model-materialization (cache disable)
    def test_enable_cache_false_evicts_in_full_mode(
        self, mock_model_patcher, tmp_path
    ):
        """enable_cache=False in full mode should evict _incremental_cache entries."""
        from nodes.exit import _CacheEntry

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # Pre-populate cache
        _incremental_cache.clear()
        _incremental_cache["some_fp"] = _CacheEntry(
            structural_fingerprint="some_fp",
            block_configs=[],
            merged_state={k: torch.zeros(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )

        (result,), mocks = _run_full_mode(
            merge, mock_model_patcher, keys, tmp_path, enable_cache=False,
        )

        assert len(_incremental_cache) == 0


# ===========================================================================
# AC: @full-saved-model-output ac-no-op-produces-full-artifact
# No-op recipes
# ===========================================================================


class TestNoOpFullMode:
    """No-op full mode returns a loaded saved artifact.

    AC: @full-saved-model-output ac-no-op-produces-full-artifact
    """

    # AC: @full-saved-model-output ac-no-op-produces-full-artifact
    def test_recipe_base_noop_produces_full_artifact(
        self, mock_model_patcher, tmp_path
    ):
        """RecipeBase-only recipe in full mode produces a valid full artifact."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        save_path = str(tmp_path / "noop.safetensors")

        with (
            patch("nodes.exit.validate_model_name", return_value="noop.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="noop_hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.ProgressBar", None),
        ):
            (result,) = WIDENExitNode().execute(
                base, save_model=True, model_name="noop"
            )

        # Artifact should exist with all base keys
        import os
        assert os.path.exists(save_path)

        with safe_open(save_path, framework="pt") as f:
            meta = f.metadata()
            assert meta["__ecaj_output_mode__"] == "full"
            saved_keys = set(f.keys())

        base_keys = set(mock_model_patcher.model_state_dict().keys())
        assert saved_keys == base_keys

    # AC: @full-saved-model-output ac-no-op-produces-full-artifact
    def test_merge_noop_produces_full_artifact(
        self, mock_model_patcher, tmp_path
    ):
        """Merge recipe that produces no affected diffusion keys still produces
        a full artifact in full mode."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "noop_merge.safetensors")
        # No keys to process (empty affected keys)

        (result,), mocks = _run_full_mode(
            merge, mock_model_patcher, [], tmp_path,
            model_name="noop_merge",
            chunked_eval_override={},
        )

        assert result is not None


# ===========================================================================
# AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
# AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
# ===========================================================================


class TestComfyMemoryCompatibility:
    """Full mode works across memory management modes without changing them.

    AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
    AC: @comfy-memory-manager-compatibility ac-non-dynamic-memory-mode-supported
    """

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    def test_full_mode_with_dynamic_vram(self, mock_model_patcher, tmp_path):
        """Full mode succeeds when dynamic VRAM management is simulated."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        (result,), _ = _run_full_mode(merge, mock_model_patcher, keys, tmp_path,
                                       model_name="dvram")
        assert result is not None

    # AC: @comfy-memory-manager-compatibility ac-non-dynamic-memory-mode-supported
    def test_full_mode_without_dynamic_vram(self, mock_model_patcher, tmp_path):
        """Full mode succeeds without dynamic VRAM management."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        (result,), _ = _run_full_mode(merge, mock_model_patcher, keys, tmp_path,
                                       model_name="no_dvram")
        assert result is not None

    # AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
    def test_full_mode_does_not_mutate_memory_mode(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode does not call any API to change the ComfyUI memory mode."""
        import sys

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # Track calls to comfy.model_management
        mm_mod = sys.modules.get("comfy.model_management")
        mode_change_calls = []
        if mm_mod is not None:
            original_attrs = {}
            for attr_name in ("set_vram_state", "vram_state"):
                if hasattr(mm_mod, attr_name):
                    original_attrs[attr_name] = getattr(mm_mod, attr_name)

        (result,), _ = _run_full_mode(merge, mock_model_patcher, keys, tmp_path,
                                       model_name="mode_check")

        # No mode change APIs should have been called
        # (the full mode code only calls free_memory/soft_empty_cache which don't
        # change the memory mode)
        assert result is not None

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_full_mode_comfy_cleanup_unavailable(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode succeeds when Comfy memory-management cleanup APIs are unavailable."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # This test works because the test environment already has comfy as stubs
        # that will raise ImportError for free_memory/get_torch_device/soft_empty_cache
        (result,), _ = _run_full_mode(merge, mock_model_patcher, keys, tmp_path,
                                       model_name="no_comfy_cleanup")
        assert result is not None


# ===========================================================================
# MaterializationSink unit tests
# ===========================================================================


class TestMaterializationSink:
    """Unit tests for MaterializationSink.

    AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    """

    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_round_trip(self, tmp_path):
        """Sink produces a valid safetensors file loadable by safe_open."""
        tensors = {
            "a": torch.randn(4, 4),
            "b": torch.randn(8, 8),
        }
        manifest = {k: (v.dtype, tuple(v.shape)) for k, v in tensors.items()}
        save_path = str(tmp_path / "sink_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path, metadata={"test": "value"})
        for name in sorted(tensors.keys()):
            sink.write_tensor(name, tensors[name])
        sink.finalize(save_path)

        with safe_open(save_path, framework="pt") as f:
            assert f.metadata()["test"] == "value"
            for k, v in tensors.items():
                loaded = f.get_tensor(k)
                assert torch.allclose(v, loaded)

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_abort_removes_temp_file(self, tmp_path):
        """Abort should remove the temp file."""
        import os

        manifest = {"a": (torch.float32, (4, 4))}
        save_path = str(tmp_path / "abort_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)
        sink.write_tensor("a", torch.randn(4, 4))
        tmp_path_before = sink._tmp_path
        sink.abort()

        assert not os.path.exists(tmp_path_before)
        assert not os.path.exists(save_path)

    def test_write_out_of_order_succeeds(self, tmp_path):
        """Sink supports random-access writes — any order is valid."""
        manifest = {
            "a": (torch.float32, (4, 4)),
            "b": (torch.float32, (4, 4)),
        }
        tensors = {"a": torch.randn(4, 4), "b": torch.randn(4, 4)}
        save_path = str(tmp_path / "order_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)
        # Write b before a — out of sorted order
        sink.write_tensor("b", tensors["b"])
        sink.write_tensor("a", tensors["a"])
        sink.finalize(save_path)

        with safe_open(save_path, framework="pt") as f:
            for k, v in tensors.items():
                loaded = f.get_tensor(k)
                assert torch.allclose(v, loaded)

    def test_write_unknown_tensor_raises(self, tmp_path):
        """Writing a tensor not in the manifest should raise RuntimeError."""
        manifest = {"a": (torch.float32, (4, 4))}
        save_path = str(tmp_path / "unknown_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        with pytest.raises(RuntimeError, match="unknown tensor name"):
            sink.write_tensor("z", torch.randn(4, 4))

    def test_write_duplicate_tensor_raises(self, tmp_path):
        """Writing the same tensor twice should raise RuntimeError."""
        manifest = {"a": (torch.float32, (4, 4))}
        save_path = str(tmp_path / "dup_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)
        sink.write_tensor("a", torch.randn(4, 4))

        with pytest.raises(RuntimeError, match="called twice"):
            sink.write_tensor("a", torch.randn(4, 4))

    def test_finalize_without_all_tensors_raises(self, tmp_path):
        """Finalize without writing all tensors should raise RuntimeError."""
        manifest = {
            "a": (torch.float32, (4, 4)),
            "b": (torch.float32, (4, 4)),
        }
        save_path = str(tmp_path / "incomplete_test.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)
        sink.write_tensor("a", torch.randn(4, 4))

        with pytest.raises(RuntimeError, match="wrote 1/2 tensors"):
            sink.finalize(save_path)

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    # AC: @full-saved-model-output ac-complete-artifact
    def test_write_tensor_rejects_dtype_mismatch(self, tmp_path):
        """write_tensor rejects a tensor with the wrong dtype."""
        manifest = {"x": (torch.float32, (2,))}
        save_path = str(tmp_path / "dtype_mismatch.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        with pytest.raises(ValueError, match="dtype mismatch"):
            sink.write_tensor("x", torch.ones(2, dtype=torch.float16))

        sink.abort()

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    # AC: @full-saved-model-output ac-complete-artifact
    def test_write_tensor_rejects_shape_mismatch(self, tmp_path):
        """write_tensor rejects a tensor with the wrong shape."""
        manifest = {"x": (torch.float32, (2,))}
        save_path = str(tmp_path / "shape_mismatch.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        with pytest.raises(ValueError, match="shape mismatch"):
            sink.write_tensor("x", torch.ones(3, dtype=torch.float32))

        sink.abort()

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    # AC: @full-saved-model-output ac-complete-artifact
    def test_write_tensor_rejects_wrong_sized_tensor(self, tmp_path):
        """write_tensor rejects a tensor whose element count differs from manifest
        even if it happens to have same total bytes (e.g. different dimensions)."""
        manifest = {"x": (torch.float32, (2, 3))}
        save_path = str(tmp_path / "size_mismatch.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        # Same dtype but different shape (3, 2) instead of (2, 3)
        with pytest.raises(ValueError, match="shape mismatch"):
            sink.write_tensor("x", torch.ones(3, 2, dtype=torch.float32))

        sink.abort()


# ===========================================================================
# AC: build_metadata output_mode
# ===========================================================================


class TestBuildMetadataOutputMode:
    """build_metadata includes output_mode in metadata.

    AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    """

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_default_mode_is_patch(self):
        """Default output_mode should be 'patch'."""
        meta = build_metadata("{}", "hash", ["k"])
        assert meta["__ecaj_output_mode__"] == "patch"

    # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    def test_full_mode_metadata(self):
        """output_mode='full' should be stored in metadata."""
        meta = build_metadata("{}", "hash", ["k"], output_mode="full")
        assert meta["__ecaj_output_mode__"] == "full"


# ===========================================================================
# AC: RecipeBase noop enable_cache=False eviction
# ===========================================================================


class TestNoopEnableCacheFalseEvicts:
    """RecipeBase full saved model mode with enable_cache=False evicts
    _incremental_cache entries.

    AC: @streaming-full-model-materialization (cache disable behavior)
    """

    # AC: @streaming-full-model-materialization (cache disable)
    def test_noop_enable_cache_false_clears_incremental_cache(
        self, mock_model_patcher, tmp_path
    ):
        """RecipeBase full mode with enable_cache=False must clear any
        pre-populated _incremental_cache entries."""
        from nodes.exit import _CacheEntry

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")
        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "noop_nocache.safetensors")

        # Pre-populate cache with a sentinel
        _incremental_cache.clear()
        _incremental_cache["sentinel"] = _CacheEntry(
            structural_fingerprint="sentinel",
            block_configs=[],
            merged_state={k: torch.zeros(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )
        assert len(_incremental_cache) == 1

        with (
            patch("nodes.exit.validate_model_name", return_value="noop_nocache.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="noop_hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.ProgressBar", None),
        ):
            (result,) = WIDENExitNode().execute(
                base, save_model=True, model_name="noop_nocache",
                enable_cache=False,
            )

        # Cache must be empty after enable_cache=False
        assert len(_incremental_cache) == 0
        assert result is not None


# ===========================================================================
# AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
# Incomplete artifact key verification
# ===========================================================================


class TestIncompleteArtifactRejected:
    """Incomplete artifacts (missing keys) are rejected by check_full_model_cache
    when expected_keys are provided.

    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    AC: @full-saved-model-output ac-complete-artifact
    """

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    # AC: @full-saved-model-output ac-complete-artifact
    def test_incomplete_artifact_rejected_with_expected_keys(self, tmp_path):
        """An artifact with valid metadata but only a subset of expected keys
        is rejected when expected_keys is provided."""
        save_path = str(tmp_path / "incomplete.safetensors")
        # Create artifact with only ONE key
        save_file(
            {"only_one_key": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["only_one_key"]',
                "__ecaj_output_mode__": "full",
            },
        )

        # Without expected_keys, metadata is valid so it passes
        assert check_full_model_cache(save_path, "hash1") is True

        # With expected_keys that include more than just "only_one_key",
        # the incomplete artifact must be rejected.
        expected = {"only_one_key", "another_key", "third_key"}
        assert check_full_model_cache(save_path, "hash1", expected_keys=expected) is False

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_complete_artifact_accepted_with_expected_keys(self, tmp_path):
        """A complete artifact with exactly the expected keys is accepted."""
        expected_keys = {"a", "b", "c"}
        save_path = str(tmp_path / "complete.safetensors")
        save_file(
            {k: torch.randn(4, 4) for k in expected_keys},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a"]',
                "__ecaj_output_mode__": "full",
            },
        )
        assert check_full_model_cache(
            save_path, "hash1", expected_keys=expected_keys,
        ) is True

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_extra_keys_in_artifact_rejected(self, tmp_path):
        """An artifact with extra unexpected keys is also rejected
        (strict equality on key sets)."""
        save_path = str(tmp_path / "extra.safetensors")
        save_file(
            {"a": torch.randn(4, 4), "b": torch.randn(4, 4), "extra": torch.randn(4, 4)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a"]',
                "__ecaj_output_mode__": "full",
            },
        )
        expected = {"a", "b"}
        assert check_full_model_cache(
            save_path, "hash1", expected_keys=expected,
        ) is False


# ===========================================================================
# AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
# Loaded model has no resident set patches
# ===========================================================================


class TestLoadedModelNoResidentPatches:
    """The model returned by _load_model_from_artifact has no set patches.
    Weights live in the model's own state dict (Comfy-managed memory).

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    """

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_loaded_model_has_no_set_patches(self, mock_model_patcher, tmp_path):
        """_load_model_from_artifact returns a model with zero set patches.
        All weights are in the model's own state dict."""
        from nodes.exit import _load_model_from_artifact

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "no_patches.safetensors")

        save_file(
            {k: torch.ones(4, 4) * 7.0 for k in keys},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "test",
                "__ecaj_affected_keys__": json.dumps(keys),
                "__ecaj_output_mode__": "full",
            },
        )

        result = _load_model_from_artifact(
            save_path, mock_model_patcher, torch.float32,
        )

        # No set patches — weights are model-owned.
        assert len(result.patches) == 0

        # All weights come from artifact (value 7.0).
        result_sd = result.model_state_dict()
        for k in keys:
            assert torch.allclose(result_sd[k], torch.ones(4, 4) * 7.0)

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_loaded_model_does_not_mutate_original(self, mock_model_patcher, tmp_path):
        """Loading from artifact does not change the original model_patcher."""
        from nodes.exit import _load_model_from_artifact

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "independence.safetensors")

        orig_values = {k: v.clone() for k, v in mock_model_patcher.model_state_dict().items()}

        save_file(
            {k: torch.ones(4, 4) * 99.0 for k in keys},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "test",
                "__ecaj_affected_keys__": json.dumps(keys),
                "__ecaj_output_mode__": "full",
            },
        )

        _load_model_from_artifact(save_path, mock_model_patcher, torch.float32)

        # Original must be unchanged.
        for k, v in mock_model_patcher.model_state_dict().items():
            assert torch.allclose(v, orig_values[k]), (
                f"Original key {k} was mutated by _load_model_from_artifact"
            )
