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
)
from tests.conftest import make_checkpoint_components

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
        """Full mode does not use the dict-returning chunked_evaluation path
        for newly evaluated affected tensors — it uses
        streaming_evaluation_to_sink which streams each tensor directly to
        the sink via write_fn.

        We monkeypatch chunked_evaluation to raise RuntimeError, then verify
        full mode still succeeds because it calls
        streaming_evaluation_to_sink instead.
        """
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "test.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )

        affected_tensors = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        # Monkeypatch the dict-returning path to fail.
        # Full mode must succeed because it uses streaming_evaluation_to_sink.
        def chunked_eval_bomb(**kwargs):
            raise RuntimeError("chunked_evaluation must not be called in full mode")

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
        """Full mode succeeds even when in-memory result sink construction
        (dict-returning chunked_evaluation) is monkeypatched to fail.

        This proves full mode never constructs an in-memory result dict to
        accumulate affected tensors — it streams directly to the
        MaterializationSink.  We also wrap MaterializationSink to confirm
        that the actual sink constructor IS called (showing the streaming
        sink replaces the in-memory sink path).
        """
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "sink_test.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )

        affected_tensors = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        _incremental_cache.clear()

        # Monkeypatch the dict-returning evaluation path to fail.  Full mode
        # must succeed because it uses streaming_evaluation_to_sink instead.
        def dict_eval_bomb(**kwargs):
            raise RuntimeError("in-memory result accumulation must not be used")

        def streaming_eval(*, keys, base_tensors, eval_fn, batch_size,
                           device, dtype, storage_dtype, write_fn):
            for k in keys:
                write_fn(k, affected_tensors[k])

        # Track that the MaterializationSink (the streaming sink) is actually
        # constructed and used — this is the replacement for the in-memory
        # result dict, and it must succeed when the dict-returning path fails.
        sink_constructed = []
        original_sink = MaterializationSink

        class TrackingSink(original_sink):
            def __init__(self):
                sink_constructed.append(True)
                super().__init__()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", side_effect=dict_eval_bomb),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.MaterializationSink", TrackingSink),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="sink_test.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(merge, save_model=True, model_name="sink_test")

        # MaterializationSink (streaming sink) was constructed
        assert len(sink_constructed) > 0, (
            "MaterializationSink was never constructed — full mode must use the streaming sink"
        )

        # Full mode must NOT leave tensor payload in _incremental_cache
        assert len(_incremental_cache) == 0
        assert result is not None

        # Verify artifact was created with full-mode metadata
        with safe_open(save_path, framework="pt") as f:
            meta = f.metadata()
            assert meta["__ecaj_output_mode__"] == "full"


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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
    # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    def test_event_order_interleaved_keys(
        self, tmp_path
    ):
        """Full mode event order with two affected groups whose keys interleave
        in sorted name order, plus an unaffected base key.

        Group 1 (4x4): keys "a" and "z"
        Group 2 (8x8): keys "b" and "c"
        Base-only: "base_only" (4x4, not affected)
        Sorted order: a, b, base_only, c, z  (interleaved!)

        Required event order:
        1. base_only written (base weights first)
        2. group 1 evaluated → group 1 keys written
        3. group 2 evaluated → group 2 keys written
        4. finalize
        5. saved model loaded
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

        base = RecipeBase(model_patcher=mock_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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

        # --- Structural assertions ---
        assert "eval:g1" in events
        assert "eval:g2" in events
        assert "finalize" in events
        assert "saved_model_loaded" in events

        # Base key must have been written
        base_write_events = [e for e in events if e.startswith("write:base:")]
        assert len(base_write_events) == 1, (
            f"Expected exactly 1 base write event, got {base_write_events}"
        )
        assert f"write:base:{base_only_key}" in events

        g1_write_events = [e for e in events if e.startswith("write:g1:")]
        g2_write_events = [e for e in events if e.startswith("write:g2:")]
        assert len(g1_write_events) == len(group1_keys)
        assert len(g2_write_events) == len(group2_keys)

        # --- Order assertions ---
        base_write_idx = events.index(f"write:base:{base_only_key}")
        eval_g1_idx = events.index("eval:g1")
        last_g1_write_idx = max(events.index(e) for e in g1_write_events)
        eval_g2_idx = events.index("eval:g2")
        finalize_idx = events.index("finalize")
        load_idx = events.index("saved_model_loaded")

        # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
        # Base weights written before any affected group evaluation begins
        assert base_write_idx < eval_g1_idx, (
            f"Base key must be written before group 1 evaluation. Events: {events}"
        )

        # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
        # Group 1 is written to sink BEFORE group 2 starts evaluation
        assert last_g1_write_idx < eval_g2_idx, (
            f"Group 1 writes must complete before group 2 evaluation. "
            f"Events: {events}"
        )

        # All writes happen before finalize
        for e in g1_write_events + g2_write_events + base_write_events:
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
    def test_failure_during_eval_aborts_sink(self, tmp_path):
        """Failure AFTER partial streaming: the first group writes tensors to
        the real MaterializationSink, then the second group's evaluation
        raises.  The test verifies:
        1. The sink is aborted (no finalized artifact at save_path).
        2. The saved model is not loaded (RuntimeError propagates).
        3. The partial artifact is not accepted as a cache hit.
        4. No full-mode tensor payload in _incremental_cache.
        """
        import os
        import uuid as _uuid

        group1_keys = ["a", "b"]
        group2_keys = ["c", "d"]
        all_keys = group1_keys + group2_keys

        # Different shapes so they form different OpSignature groups
        state_dict = {}
        for k in group1_keys:
            state_dict[k] = torch.randn(4, 4, dtype=torch.float32)
        for k in group2_keys:
            state_dict[k] = torch.randn(8, 8, dtype=torch.float32)

        from tests.conftest import MockModelPatcher
        mock_patcher = MockModelPatcher.__new__(MockModelPatcher)
        mock_patcher._state_dict = state_dict
        mock_patcher.model = MagicMock()
        mock_patcher.model.diffusion_model = MagicMock()
        mock_patcher.model.diffusion_model.state_dict = MagicMock(return_value=state_dict)
        mock_patcher.patches = {}
        mock_patcher.patches_uuid = _uuid.uuid4()

        def _clone():
            c = MockModelPatcher.__new__(MockModelPatcher)
            c._state_dict = mock_patcher._state_dict
            c.model = mock_patcher.model
            c.patches = {}
            c.patches_uuid = mock_patcher.patches_uuid
            return c
        mock_patcher.clone = _clone

        base = RecipeBase(model_patcher=mock_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        save_path = str(tmp_path / "partial.safetensors")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_full_mode_mocks(
            mock_patcher, all_keys, recipe=merge,
        )
        sig1 = OpSignature(shape=(4, 4), ndim=2)
        sig2 = OpSignature(shape=(8, 8), ndim=2)
        batch_groups = {sig1: group1_keys, sig2: group2_keys}

        group1_results = {k: torch.randn(4, 4) for k in group1_keys}
        tensors_written = []

        call_count = [0]

        def partial_streaming_eval(**kwargs):
            """First call succeeds (writes tensors), second call raises."""
            called_keys = kwargs.get("keys", [])
            write_fn = kwargs.get("write_fn")
            call_count[0] += 1
            if call_count[0] == 1:
                # First group: write tensors successfully
                for k in called_keys:
                    write_fn(k, group1_results[k])
                    tensors_written.append(k)
            else:
                # Second group: fail after first group was already written
                raise RuntimeError("Simulated mid-stream failure")

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value=batch_groups),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=partial_streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="partial.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            with pytest.raises(RuntimeError, match="Simulated mid-stream failure"):
                WIDENExitNode().execute(merge, save_model=True, model_name="partial")

        # At least one tensor was written before failure
        assert len(tensors_written) > 0, "Test must write tensors before failing"

        # No artifact file at save_path (abort cleaned up)
        assert not os.path.exists(save_path), "Partial artifact must not exist"

        # No full-mode tensor payload in cache
        assert len(_incremental_cache) == 0

        # --- Later-run cache validation ---
        # Simulate a partial artifact surviving at save_path (e.g., abort
        # raced with rename, or temp file was manually moved).  The partial
        # artifact has only group 1's keys, not all 4 keys the model needs.
        # check_full_model_cache must reject it as a cache hit.
        partial_tensors = {k: torch.randn(4, 4) for k in group1_keys}
        save_file(
            partial_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "test_hash",
                "__ecaj_affected_keys__": json.dumps(group1_keys),
                "__ecaj_output_mode__": "full",
            },
        )
        # The partial artifact is missing group2_keys and has wrong shapes
        # for a full model.  check_full_model_cache must reject it when
        # given the full expected manifest.
        full_manifest = {}
        for k in group1_keys:
            full_manifest[k] = (torch.float32, (4, 4))
        for k in group2_keys:
            full_manifest[k] = (torch.float32, (8, 8))
        assert check_full_model_cache(
            save_path, "test_hash", expected_manifest=full_manifest,
        ) is False, "Partial artifact must not be accepted as a full-model cache hit"

    # AC: @streaming-full-model-materialization ac-failed-materialization-releases-resident-payload
    def test_failure_leaves_no_full_mode_tensor_in_cache(
        self, mock_model_patcher, tmp_path
    ):
        """After failure, no full-mode tensor payload remains in _incremental_cache."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
            patch(
                "nodes.exit._resolve_checkpoints_path",
                return_value=str(tmp_path / "fail.safetensors"),
            ),
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
        """If finalize fails, the node's error handling must abort the sink
        and leave no reusable artifact or full-mode cache payload.

        The FailingFinalizeSink raises WITHOUT calling self.abort() — the
        node's except handler (sink.abort()) is the only cleanup path.  If
        the node's error handling is removed, this test would leave a temp
        file or partial artifact on disk.
        """
        import os

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
                # Raise WITHOUT self-cleanup.  The node's except path must
                # call sink.abort() to clean up the temp file.
                raise OSError("Simulated finalize failure")

        _incremental_cache.clear()

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

        # No artifact at save_path — the node's abort cleaned up
        assert not os.path.exists(save_path), (
            "Finalize failure must not leave artifact at save_path"
        )

        # No full-mode tensor payload in cache
        assert len(_incremental_cache) == 0, (
            "Finalize failure must not leave tensor payload in _incremental_cache"
        )

        # No temp files left behind in the directory
        temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert len(temp_files) == 0, (
            f"Temp files left behind after finalize failure: {temp_files}"
        )


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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
        """A pre-populated patch-mode tensor cache sentinel (all zeros) is not
        read or reused by full mode. The returned artifact must contain the
        freshly computed tensors (non-zero), not the sentinel zeros.

        Also tests cross-mode isolation: a full-model artifact must not be
        treated as a patch-mode tensor payload."""
        from nodes.exit import _CacheEntry

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # Pre-populate cache with a sentinel: ALL ZEROS
        sentinel = _CacheEntry(
            structural_fingerprint="sentinel_fp",
            block_configs=[],
            merged_state={k: torch.zeros(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )
        _incremental_cache.clear()
        _incremental_cache["sentinel_fp"] = sentinel

        # Use distinctive non-zero tensors for the fresh computation
        fresh_value = 42.0
        fresh_tensors = {k: torch.ones(4, 4) * fresh_value for k in keys}

        (result,), mocks = _run_full_mode(
            merge, mock_model_patcher, keys, tmp_path,
            chunked_eval_override=fresh_tensors,
        )

        # The returned model's weights must come from the artifact (fresh
        # computation, value 42.0), NOT from the sentinel (zeros).
        result_sd = result.model_state_dict()
        for k in keys:
            assert not torch.allclose(result_sd[k], torch.zeros(4, 4)), (
                f"Key {k} has sentinel zeros — full mode used the patch cache!"
            )

        # The artifact file itself must contain the fresh values, not zeros
        save_path = mocks["save_path"]
        with safe_open(save_path, framework="pt") as f:
            for k in keys:
                loaded = f.get_tensor(k)
                assert torch.allclose(loaded, torch.ones(4, 4) * fresh_value), (
                    f"Artifact key {k} should have fresh value {fresh_value}"
                )

    # AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
    def test_full_artifact_not_treated_as_patch_tensor_payload(
        self, mock_model_patcher, tmp_path
    ):
        """A full-model artifact is not treated as a patch-mode tensor payload.

        Phase 1: Run full mode — _incremental_cache must be empty afterward.
        Phase 2: Run patch mode — patch mode must compute fresh results
        (not reuse the full-mode artifact as a tensor payload) and populate
        _incremental_cache with its own tensor payload.
        """
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        _incremental_cache.clear()

        # Phase 1: full mode
        (result,), mocks = _run_full_mode(merge, mock_model_patcher, keys, tmp_path)

        # Full mode must NOT populate _incremental_cache
        assert len(_incremental_cache) == 0, (
            "Full mode must not leave tensor payload in _incremental_cache"
        )

        # The artifact is marked as full mode
        save_path = mocks["save_path"]
        with safe_open(save_path, framework="pt") as f:
            meta = f.metadata()
            assert meta["__ecaj_output_mode__"] == "full"

        # Phase 2: patch mode — must compute fresh, not reuse full artifact
        mock_analyze, mock_model_analysis, _, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher, keys, recipe=merge,
        )
        patch_merged = {k: torch.randn(4, 4) for k in keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        chunked_eval_called = []

        def patch_chunked_eval(**kwargs):
            chunked_eval_called.append(True)
            return patch_merged

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", side_effect=patch_chunked_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            node = WIDENExitNode()
            (patch_result,) = node.execute(merge, save_model=False)

        # Patch mode must have called chunked_evaluation (its own eval path)
        assert len(chunked_eval_called) > 0, (
            "Patch mode must use chunked_evaluation, not the full-mode artifact"
        )

        # Patch mode DOES populate _incremental_cache (its tensor payload)
        assert len(_incremental_cache) == 1, (
            "Patch mode must populate _incremental_cache with tensor payload"
        )


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

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

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

    def _run_with_vram_state(self, mock_model_patcher, tmp_path, model_name,
                             vram_state_value, *, cleanup_available=True):
        """Run full mode with a simulated Comfy memory management state.

        Installs a mock comfy.model_management module with the given
        vram_state and tracks whether any mode-changing API was called.

        Returns (result, mode_change_calls, mm, save_path).
        """
        import sys
        import types

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())

        # Build a mock comfy.model_management module
        mm = types.ModuleType("comfy.model_management")
        mm.vram_state = vram_state_value  # type: ignore[attr-defined]

        mode_change_calls = []

        def mock_set_vram_state(state):
            mode_change_calls.append(("set_vram_state", state))

        mm.set_vram_state = mock_set_vram_state  # type: ignore[attr-defined]

        if cleanup_available:
            mm.free_memory = MagicMock()  # type: ignore[attr-defined]
            mm.get_torch_device = MagicMock(return_value="cpu")  # type: ignore[attr-defined]
            mm.soft_empty_cache = MagicMock()  # type: ignore[attr-defined]
        else:
            # Simulate cleanup APIs being unavailable
            pass  # no free_memory/get_torch_device/soft_empty_cache

        _incremental_cache.clear()

        # Install mock module
        old_mm = sys.modules.get("comfy.model_management")
        sys.modules["comfy.model_management"] = mm

        try:
            (result,), mocks = _run_full_mode(
                merge, mock_model_patcher, keys, tmp_path,
                model_name=model_name,
            )
        finally:
            # Restore
            if old_mm is not None:
                sys.modules["comfy.model_management"] = old_mm
            else:
                sys.modules.pop("comfy.model_management", None)

        return result, mode_change_calls, mm, mocks["save_path"]

    def _assert_streaming_behavior(self, result, save_path):
        """Common assertions proving full mode streamed to artifact and left
        no resident payload — shared across all memory-mode tests."""
        import os

        # Result is a loaded model from the artifact
        assert result is not None

        # Artifact was written to disk (streaming succeeded)
        assert os.path.exists(save_path), (
            "Full-mode artifact must exist after streaming"
        )
        with safe_open(save_path, framework="pt") as f:
            meta = f.metadata()
            assert meta["__ecaj_output_mode__"] == "full"

        # No full-mode resident payload in _incremental_cache
        assert len(_incremental_cache) == 0, (
            "Full mode must not leave tensor payload in _incremental_cache"
        )

    # AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
    def test_full_mode_with_dynamic_vram(self, mock_model_patcher, tmp_path):
        """Full mode streams affected tensors and leaves no resident payload
        when Dynamic VRAM is the active memory mode.  The exit node must not
        call set_vram_state to opt out."""
        result, mode_change_calls, mm, save_path = self._run_with_vram_state(
            mock_model_patcher, tmp_path, "dvram",
            vram_state_value="NORMAL_VRAM",
        )
        self._assert_streaming_behavior(result, save_path)
        assert len(mode_change_calls) == 0, (
            f"Full mode must not change VRAM state. Calls: {mode_change_calls}"
        )

    # AC: @comfy-memory-manager-compatibility ac-non-dynamic-memory-mode-supported
    def test_full_mode_without_dynamic_vram(self, mock_model_patcher, tmp_path):
        """Full mode streams and leaves no resident payload when Dynamic VRAM
        is NOT active (e.g. HIGH_VRAM)."""
        result, mode_change_calls, mm, save_path = self._run_with_vram_state(
            mock_model_patcher, tmp_path, "no_dvram",
            vram_state_value="HIGH_VRAM",
        )
        self._assert_streaming_behavior(result, save_path)
        assert len(mode_change_calls) == 0, (
            f"Full mode must not change VRAM state. Calls: {mode_change_calls}"
        )

    # AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
    def test_full_mode_does_not_mutate_memory_mode(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode streams, leaves no resident payload, and does not mutate
        the ComfyUI memory mode.  vram_state before and after must match."""
        result, mode_change_calls, mm, save_path = self._run_with_vram_state(
            mock_model_patcher, tmp_path, "mode_check",
            vram_state_value="LOW_VRAM",
        )
        self._assert_streaming_behavior(result, save_path)
        assert len(mode_change_calls) == 0, (
            f"set_vram_state was called: {mode_change_calls}"
        )
        assert mm.vram_state == "LOW_VRAM", (
            f"vram_state was mutated from LOW_VRAM to {mm.vram_state}"
        )

    # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
    def test_full_mode_comfy_cleanup_unavailable(
        self, mock_model_patcher, tmp_path
    ):
        """Full mode streams, leaves no resident payload, and succeeds when
        Comfy memory-management cleanup APIs are unavailable."""
        result, mode_change_calls, mm, save_path = self._run_with_vram_state(
            mock_model_patcher, tmp_path, "no_comfy_cleanup",
            vram_state_value="NORMAL_VRAM",
            cleanup_available=False,
        )
        self._assert_streaming_behavior(result, save_path)


# ===========================================================================
# MaterializationSink unit tests
# ===========================================================================


class TestMaterializationSink:
    """Unit tests for the MaterializationSink helper.

    These are helper-only tests that verify the sink API in isolation.
    Product-path streaming AC coverage lives on runtime tests that execute
    the full saved model mode through WIDENExitNode (see TestFullModeEventOrder,
    TestFailureAbortsMaterialization, TestFinalizeFailure, etc.).
    """

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

    def test_write_tensor_rejects_dtype_mismatch(self, tmp_path):
        """write_tensor rejects a tensor with the wrong dtype."""
        manifest = {"x": (torch.float32, (2,))}
        save_path = str(tmp_path / "dtype_mismatch.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        with pytest.raises(ValueError, match="dtype mismatch"):
            sink.write_tensor("x", torch.ones(2, dtype=torch.float16))

        sink.abort()

    def test_write_tensor_rejects_shape_mismatch(self, tmp_path):
        """write_tensor rejects a tensor with the wrong shape."""
        manifest = {"x": (torch.float32, (2,))}
        save_path = str(tmp_path / "shape_mismatch.safetensors")

        sink = MaterializationSink()
        sink.open(manifest, save_path)

        with pytest.raises(ValueError, match="shape mismatch"):
            sink.write_tensor("x", torch.ones(3, dtype=torch.float32))

        sink.abort()

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

        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
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
    when expected_manifest is provided.

    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    AC: @full-saved-model-output ac-complete-artifact
    """

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    # AC: @full-saved-model-output ac-complete-artifact
    def test_incomplete_artifact_rejected_with_expected_manifest(self, tmp_path):
        """An artifact with valid metadata but only a subset of expected keys
        is rejected when expected_manifest is provided."""
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

        # Without expected_manifest, metadata is valid so it passes
        assert check_full_model_cache(save_path, "hash1") is True

        # With expected_manifest that include more than just "only_one_key",
        # the incomplete artifact must be rejected.
        expected_manifest = {
            "only_one_key": (torch.float32, (4, 4)),
            "another_key": (torch.float32, (4, 4)),
            "third_key": (torch.float32, (4, 4)),
        }
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
        ) is False

    # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
    def test_complete_artifact_accepted_with_expected_manifest(self, tmp_path):
        """A complete artifact with exactly the expected keys/shapes/dtypes is accepted."""
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
        expected_manifest = {k: (torch.float32, (4, 4)) for k in expected_keys}
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
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
        expected_manifest = {
            "a": (torch.float32, (4, 4)),
            "b": (torch.float32, (4, 4)),
        }
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
        ) is False


# ===========================================================================
# AC: @exit-model-persistence ac-9
# Non-ecaj / corrupt file raises ValueError (not silently overwritten)
# ===========================================================================


class TestNonEcajFileRaisesOnFullModeCache:
    """check_full_model_cache raises ValueError for non-ecaj or corrupt files.

    AC: @exit-model-persistence ac-9
    """

    # AC: @exit-model-persistence ac-9
    def test_non_safetensors_file_raises(self, tmp_path):
        """A pre-existing non-safetensors file at save_path must raise ValueError,
        not return False (which would cause the caller to overwrite it)."""
        save_path = str(tmp_path / "user_model.safetensors")
        # Write a plain text file — not a valid safetensors file.
        with open(save_path, "w") as f:
            f.write("this is not a safetensors file")

        with pytest.raises(ValueError, match="not a valid safetensors file"):
            check_full_model_cache(save_path, "any_hash")

    # AC: @exit-model-persistence ac-9
    def test_safetensors_without_ecaj_metadata_raises(self, tmp_path):
        """A valid safetensors file without ecaj metadata must raise ValueError."""
        save_path = str(tmp_path / "foreign.safetensors")
        # Save a valid safetensors file with NO ecaj metadata.
        save_file(
            {"layer.weight": torch.randn(4, 4)},
            save_path,
            metadata={},
        )

        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_full_model_cache(save_path, "any_hash")

    # AC: @exit-model-persistence ac-9
    def test_safetensors_with_no_metadata_raises(self, tmp_path):
        """A safetensors file with None metadata must raise ValueError."""
        save_path = str(tmp_path / "no_meta.safetensors")
        save_file(
            {"x": torch.randn(2, 2)},
            save_path,
        )

        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_full_model_cache(save_path, "any_hash")

    # AC: @exit-model-persistence ac-9
    def test_full_mode_does_not_overwrite_non_ecaj_file(
        self, mock_model_patcher, tmp_path,
    ):
        """Runtime probe: a pre-existing non-safetensors file at the save path
        must cause the full-mode execution to raise, NOT silently overwrite."""
        base = RecipeBase(model_patcher=mock_model_patcher, arch="sdxl",
                          checkpoint_components=make_checkpoint_components())
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "user_file.safetensors")

        # Create a non-safetensors file the user cares about.
        with open(save_path, "w") as f:
            f.write("important user data — must not be overwritten")

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = (
            _make_full_mode_mocks(mock_model_patcher, keys, recipe=merge)
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(**kwargs):
            write_fn = kwargs.get("write_fn")
            for k in kwargs.get("keys", []):
                write_fn(k, torch.randn(4, 4))

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="user_file.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path", return_value=save_path),
            # Do NOT mock check_full_model_cache — let the real one run.
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit.ProgressBar", None),
        ):
            with pytest.raises(ValueError, match="not a valid safetensors"):
                WIDENExitNode().execute(merge, save_model=True, model_name="user_file")

        # The user's file must not have been overwritten.
        with open(save_path) as f:
            assert f.read() == "important user data — must not be overwritten"


# ===========================================================================
# AC: @full-saved-model-output ac-complete-artifact
# Artifact tensor shape/dtype validation
# ===========================================================================


class TestArtifactShapeDtypeValidation:
    """check_full_model_cache validates tensor shapes and dtypes against the
    expected manifest, not just key names.

    AC: @full-saved-model-output ac-complete-artifact
    """

    # AC: @full-saved-model-output ac-complete-artifact
    def test_wrong_shape_rejected(self, tmp_path):
        """An artifact with correct key name but wrong tensor shape is rejected."""
        save_path = str(tmp_path / "wrong_shape.safetensors")
        save_file(
            {"a": torch.randn(3, 3)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a"]',
                "__ecaj_output_mode__": "full",
            },
        )
        # Manifest expects (4, 4) but artifact has (3, 3).
        expected_manifest = {"a": (torch.float32, (4, 4))}
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
        ) is False

    # AC: @full-saved-model-output ac-complete-artifact
    def test_wrong_dtype_rejected(self, tmp_path):
        """An artifact with correct key name and shape but wrong dtype is rejected."""
        save_path = str(tmp_path / "wrong_dtype.safetensors")
        save_file(
            {"a": torch.randn(4, 4, dtype=torch.float16)},
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a"]',
                "__ecaj_output_mode__": "full",
            },
        )
        # Manifest expects float32 but artifact has float16.
        expected_manifest = {"a": (torch.float32, (4, 4))}
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
        ) is False

    # AC: @full-saved-model-output ac-complete-artifact
    def test_correct_shape_and_dtype_accepted(self, tmp_path):
        """An artifact with matching key names, shapes, and dtypes is accepted."""
        save_path = str(tmp_path / "correct.safetensors")
        save_file(
            {
                "a": torch.randn(4, 4, dtype=torch.float32),
                "b": torch.randn(8, 8, dtype=torch.float16),
            },
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a", "b"]',
                "__ecaj_output_mode__": "full",
            },
        )
        expected_manifest = {
            "a": (torch.float32, (4, 4)),
            "b": (torch.float16, (8, 8)),
        }
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
        ) is True

    # AC: @full-saved-model-output ac-complete-artifact
    def test_multi_key_one_wrong_shape(self, tmp_path):
        """If one of multiple keys has the wrong shape, the entire artifact
        is rejected."""
        save_path = str(tmp_path / "partial_mismatch.safetensors")
        save_file(
            {
                "a": torch.randn(4, 4),
                "b": torch.randn(3, 3),  # Wrong: manifest expects (8, 8)
            },
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "hash1",
                "__ecaj_affected_keys__": '["a", "b"]',
                "__ecaj_output_mode__": "full",
            },
        )
        expected_manifest = {
            "a": (torch.float32, (4, 4)),
            "b": (torch.float32, (8, 8)),
        }
        assert check_full_model_cache(
            save_path, "hash1", expected_manifest=expected_manifest,
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


# ===========================================================================
# AC: @full-saved-model-output ac-return-loaded-model
# Mixed-dtype artifact preservation
# ===========================================================================


class TestMixedDtypeArtifactPreservation:
    """_load_model_from_artifact preserves per-key dtypes from the artifact
    instead of coercing everything to a single storage_dtype.

    AC: @full-saved-model-output ac-return-loaded-model
    """

    # AC: @full-saved-model-output ac-return-loaded-model
    def test_mixed_dtype_artifact_preserved(self, tmp_path):
        """A mixed-dtype artifact (some keys float32, some float16) must
        have its per-key dtypes preserved after loading — not coerced to
        a single dtype."""
        import uuid as _uuid

        from nodes.exit import _load_model_from_artifact
        from tests.conftest import MockModelPatcher

        # Create a model patcher with mixed dtypes
        state_dict = {
            "diffusion_model.layer1.weight": torch.randn(4, 4, dtype=torch.float32),
            "diffusion_model.layer2.weight": torch.randn(4, 4, dtype=torch.float16),
        }
        mock_patcher = MockModelPatcher.__new__(MockModelPatcher)
        mock_patcher._state_dict = state_dict
        mock_patcher.model = MagicMock()
        mock_patcher.model.diffusion_model = MagicMock()
        mock_patcher.model.diffusion_model.state_dict = MagicMock(return_value={
            k.removeprefix("diffusion_model."): v for k, v in state_dict.items()
        })
        mock_patcher.model.diffusion_model.load_state_dict = MagicMock(
            side_effect=TypeError("fallback"),
        )
        mock_patcher.patches = {}
        mock_patcher.patches_uuid = _uuid.uuid4()

        def _clone():
            c = MockModelPatcher.__new__(MockModelPatcher)
            c._state_dict = dict(mock_patcher._state_dict)
            c.model = mock_patcher.model
            c.patches = {}
            c.patches_uuid = mock_patcher.patches_uuid
            return c
        mock_patcher.clone = _clone

        # Create artifact with mixed dtypes
        save_path = str(tmp_path / "mixed.safetensors")
        artifact_tensors = {
            "diffusion_model.layer1.weight": torch.ones(4, 4, dtype=torch.float32) * 10.0,
            "diffusion_model.layer2.weight": torch.ones(
                4, 4, dtype=torch.float16,
            ) * 5.0,
        }
        save_file(
            artifact_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "test",
                "__ecaj_affected_keys__": json.dumps(list(artifact_tensors.keys())),
                "__ecaj_output_mode__": "full",
            },
        )

        result = _load_model_from_artifact(
            save_path, mock_patcher, torch.float32,
        )

        result_sd = result.model_state_dict()
        # float32 key must be float32
        assert result_sd["diffusion_model.layer1.weight"].dtype == torch.float32
        # float16 key must remain float16, NOT coerced to float32
        assert result_sd["diffusion_model.layer2.weight"].dtype == torch.float16
