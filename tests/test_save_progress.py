"""Tests for streaming saved-model materialization progress reporting.

AC coverage for: @streaming-materialization-progress
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from lib.batch_groups import OpSignature
from lib.recipe import RecipeBase, RecipeLoRA, RecipeMerge
from lib.recipe_eval import EvalPlan
from lib.save_progress import SavedModelProgress
from nodes.exit import WIDENExitNode, _incremental_cache

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingProgressBar:
    """Fake ProgressBar that records every update() call."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.updates: list[int] = []

    def update(self, value: int) -> None:
        self.updates.append(value)


class RaisingProgressBar:
    """ProgressBar fake that explodes on construction — exercises the
    defensive ``except Exception`` in ``SavedModelProgress.__init__``."""

    def __init__(self, total: int) -> None:
        raise RuntimeError("simulated ProgressBar construction failure")


# ---------------------------------------------------------------------------
# Helper unit tests — verify the SavedModelProgress contract directly.
# ---------------------------------------------------------------------------


class TestSavedModelProgressHelper:
    """Direct tests of the SavedModelProgress helper.

    These cover the no-ComfyUI safety path and the per-phase semantics that
    the exit node depends on.
    """

    # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    def test_helper_advances_progress_bar_per_tensor_write(self):
        """tensor_written advances the underlying ProgressBar one unit."""
        bars: list[RecordingProgressBar] = []

        def factory(total: int) -> RecordingProgressBar:
            bar = RecordingProgressBar(total)
            bars.append(bar)
            return bar

        prog = SavedModelProgress(
            total_units=5,
            progress_bar_factory=factory,
        )
        prog.prepare()
        prog.tensor_written("k1")
        prog.tensor_written("k2")

        assert len(bars) == 1
        assert bars[0].total == 5
        assert bars[0].updates == [1, 1, 1]
        assert prog.advanced == 3
        assert prog.phase == "write_tensor"

    # AC: @streaming-materialization-progress ac-finalization-status-visible
    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_helper_emits_finalize_phase_without_marking_published(self):
        """finalize() enters the finalize phase but must not set published.

        Publication is only reported via mark_published() so a failure
        between finalize() and mark_published() (e.g. inside the sink's
        atomic-replace work) leaves published=False.
        """
        prog = SavedModelProgress(
            total_units=3,
            progress_bar_factory=RecordingProgressBar,
        )
        prog.prepare()
        prog.finalize()

        assert prog.phase == "finalize"
        # finalize() advances the bar but must NOT set published.
        assert prog.published is False

        prog.mark_published()
        # mark_published flips the flag without changing the visible phase
        # or advancing the bar — the user-visible status remains "finalize"
        # until the caller transitions to reload.
        assert prog.published is True
        assert prog.phase == "finalize"

        prog.reload()
        phases = [phase for phase, _ in prog.messages]
        assert phases == ["prepare", "finalize", "reload"]
        assert prog.published is True

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_helper_finalize_without_publish_keeps_published_false(self):
        """A finalize() entry with no mark_published() (sink failure) must
        leave the artifact reported as not-published."""
        prog = SavedModelProgress(
            total_units=3,
            progress_bar_factory=RecordingProgressBar,
        )
        prog.prepare()
        prog.finalize()
        # Simulate sink.finalize raising before mark_published is reached.
        prog.failure("RuntimeError")

        assert prog.published is False
        assert prog.failed is True
        phases = [phase for phase, _ in prog.messages]
        assert phases[-1] == "failure"

    # AC: @streaming-materialization-progress ac-cache-reuse-status-visible
    def test_helper_cache_reuse_uses_distinct_phase(self):
        """Cache reuse does not pretend to write tensors."""
        prog = SavedModelProgress(
            total_units=1,
            progress_bar_factory=RecordingProgressBar,
        )
        prog.cache_reuse()
        phases = [phase for phase, _ in prog.messages]
        assert phases == ["cache_reuse"]
        # No tensor write phases were entered.
        assert "write_tensor" not in phases
        assert prog.published is False

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_helper_failure_does_not_report_published(self):
        """Failure must clear published and emit a failure phase."""
        prog = SavedModelProgress(
            total_units=3,
            progress_bar_factory=RecordingProgressBar,
        )
        prog.prepare()
        prog.tensor_written("k1")
        prog.failure("RuntimeError")

        assert prog.published is False
        assert prog.failed is True
        assert prog.phase == "failure"
        assert "failure" in [phase for phase, _ in prog.messages]

    # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    def test_helper_safe_with_no_progress_bar_factory(self):
        """ProgressBar absence (e.g. running outside ComfyUI) is safe."""
        prog = SavedModelProgress(
            total_units=2,
            progress_bar_factory=None,
        )
        prog.prepare()
        prog.tensor_written("k1")
        prog.finalize()
        prog.mark_published()
        prog.reload()
        # The bar is None; the helper still tracks logical advance/phase.
        assert prog.advanced > 0
        assert prog.phase == "reload"
        assert prog.published is True

    def test_helper_swallows_progress_bar_construction_errors(self):
        """A broken ProgressBar must not break a save."""
        prog = SavedModelProgress(
            total_units=2,
            progress_bar_factory=RaisingProgressBar,
        )
        # Exercising every public method must remain safe.
        prog.prepare()
        prog.tensor_written("k1")
        prog.finalize()
        prog.mark_published()
        prog.reload()
        assert prog.published is True


# ---------------------------------------------------------------------------
# Exit-node integration tests — patch the helper factory functions and
# confirm the exit node drives them through the expected lifecycle.
# ---------------------------------------------------------------------------


def _make_full_mode_mocks(mock_model_patcher, keys, *, recipe=None):
    mock_loader = MagicMock()
    mock_loader.affected_keys = set(keys)
    mock_loader.loaded_bytes = 0
    mock_loader.cleanup = MagicMock()

    set_affected = {str(id(recipe) if recipe is not None else None): set(keys)}

    mock_analyze = MagicMock(
        model_patcher=mock_model_patcher,
        arch="sdxl",
        loader=mock_loader,
        set_affected=set_affected,
        affected_keys=set(keys),
    )
    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()

    dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())
    return mock_analyze, mock_model_analysis, dummy_plan


class TestExitNodeProgressNoOpDiffusionSave:
    """Diffusion-only no-op save advances progress for every base tensor.

    AC: @streaming-materialization-progress ac-no-op-save-progress
    AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    AC: @streaming-materialization-progress ac-finalization-status-visible
    """

    # AC: @streaming-materialization-progress ac-no-op-save-progress
    # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    def test_noop_diffusion_save_advances_progress_per_base_write(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noop_progress.safetensors")
        keys = list(mock_model_patcher.model_state_dict().keys())

        recorded: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=RecordingProgressBar,
                artifact_name=artifact_name,
            )
            recorded.append(prog)
            return prog

        with (
            patch("nodes.exit.validate_model_name", return_value="noop_progress.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            WIDENExitNode().execute(base, save_model=True, model_name="noop")

        assert len(recorded) == 1
        prog = recorded[0]
        # Each base tensor advanced progress, plus prepare/finalize/reload.
        write_phase_messages = [
            (phase, msg) for phase, msg in prog.messages if phase == "write_tensor"
        ]
        assert len(write_phase_messages) == 1, (
            f"Expected single write_tensor phase entry, got: {prog.messages}"
        )
        assert prog.advanced == len(keys) + 3
        assert prog.published is True
        assert prog.failed is False

    # AC: @streaming-materialization-progress ac-finalization-status-visible
    def test_noop_diffusion_save_reports_post_write_phases(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noop_phases.safetensors")

        recorded: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            recorded.append(prog)
            return prog

        with (
            patch("nodes.exit.validate_model_name", return_value="noop_phases.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            WIDENExitNode().execute(base, save_model=True, model_name="noop2")

        prog = recorded[0]
        phases = [phase for phase, _ in prog.messages]
        assert phases.count("prepare") == 1
        assert "finalize" in phases
        assert phases[-1] == "reload"


class TestExitNodeProgressDiffusionMergeSave:
    """Normal diffusion merge save advances progress for unaffected and affected
    tensor handoffs.

    AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    AC: @streaming-materialization-progress ac-affected-write-progress
    """

    # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    # AC: @streaming-materialization-progress ac-affected-write-progress
    def test_diffusion_merge_save_advances_for_each_tensor_handoff(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        all_keys = list(mock_model_patcher.model_state_dict().keys())
        # Mark one key affected, the rest base/unaffected.
        affected_keys = all_keys[:1]
        unaffected_keys = all_keys[1:]
        save_path = str(tmp_path / "merge_progress.safetensors")

        mock_analyze, mock_model_analysis, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher,
            affected_keys,
            recipe=merge,
        )

        merged = {k: torch.randn(4, 4) for k in affected_keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(**kwargs):
            for k in kwargs["keys"]:
                kwargs["write_fn"](k, merged[k])

        recorded: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=RecordingProgressBar,
                artifact_name=artifact_name,
            )
            recorded.append(prog)
            return prog

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: affected_keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="merge_progress.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            WIDENExitNode().execute(merge, save_model=True, model_name="merge")

        assert len(recorded) == 1
        prog = recorded[0]

        # Every tensor in the model was reported via tensor_written —
        # whether base/unaffected or affected/streamed.
        write_msgs = [m for phase, m in prog.messages if phase == "write_tensor"]
        # The phase records on first entry; per-tensor advances do not add
        # a new message, but the total advance count must equal the number
        # of tensors plus the three lifecycle phases.
        assert len(write_msgs) == 1
        assert prog.advanced == len(affected_keys) + len(unaffected_keys) + 3
        # The recorded ProgressBar received the same number of update(1)
        # calls as the helper advanced.
        bar = prog._pbar  # type: ignore[attr-defined]
        assert isinstance(bar, RecordingProgressBar)
        assert sum(bar.updates) == prog.advanced

        # Lifecycle: prepare → write_tensor → finalize → reload, no failure.
        phases = [phase for phase, _ in prog.messages]
        assert phases[0] == "prepare"
        assert "finalize" in phases
        assert phases[-1] == "reload"
        assert "failure" not in phases
        assert prog.published is True


class TestExitNodeProgressCacheHit:
    """Full-mode cache hits report a reuse phase without tensor-write progress.

    AC: @streaming-materialization-progress ac-cache-reuse-status-visible
    """

    # AC: @streaming-materialization-progress ac-cache-reuse-status-visible
    def test_cache_hit_reports_cache_reuse_phase_no_tensor_writes(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        keys = list(mock_model_patcher.model_state_dict().keys())
        affected_key = keys[0]
        save_path = str(tmp_path / "cached_progress.safetensors")

        artifact_tensors = {
            "model.diffusion_model." + k.removeprefix("diffusion_model."): torch.ones(4, 4) * 7.0
            for k in keys
        }
        dep_fps = json.dumps({}, sort_keys=True, separators=(",", ":"))
        save_file(
            artifact_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "match",
                "__ecaj_affected_keys__": json.dumps([affected_key]),
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
                "__ecaj_base_identity__": "base_id",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )

        recorded_save: list[SavedModelProgress] = []
        recorded_cache: list[SavedModelProgress] = []

        def capture_save(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            recorded_save.append(prog)
            return prog

        def capture_cache(*, artifact_name: str):
            prog = SavedModelProgress(
                total_units=1,
                progress_bar_factory=RecordingProgressBar,
                artifact_name=artifact_name,
            )
            recorded_cache.append(prog)
            return prog

        with (
            patch("nodes.exit.validate_model_name", return_value="cached_progress.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="match"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.analyze_recipe") as mock_analyze,
            patch(
                "nodes.exit._comfy_load_diffusion_model",
                return_value=MagicMock(name="loaded"),
            ),
            patch("nodes.exit._build_save_progress", side_effect=capture_save),
            patch("nodes.exit._build_cache_reuse_progress", side_effect=capture_cache),
        ):
            WIDENExitNode().execute(merge, save_model=True, model_name="cached")
            mock_analyze.assert_not_called()

        # Save-path progress must NOT be constructed on cache hit (no tensors
        # were ever written to a fresh artifact).
        assert recorded_save == []
        assert len(recorded_cache) == 1
        prog = recorded_cache[0]
        phases = [phase for phase, _ in prog.messages]
        assert phases == ["cache_reuse"]
        # Reuse path must not advance through a tensor-write phase.
        assert "write_tensor" not in phases
        assert prog.published is False

    # AC: @streaming-materialization-progress ac-cache-reuse-status-visible
    def test_noop_cache_hit_reports_cache_reuse_phase(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        """Even a no-op cache hit (RecipeBase only) reports cache_reuse."""
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )

        keys = list(mock_model_patcher.model_state_dict().keys())
        save_path = str(tmp_path / "cached_noop_progress.safetensors")

        artifact_tensors = {
            "model.diffusion_model." + k.removeprefix("diffusion_model."): torch.zeros(4, 4)
            for k in keys
        }
        dep_fps = json.dumps({}, sort_keys=True, separators=(",", ":"))
        save_file(
            artifact_tensors,
            save_path,
            metadata={
                "__ecaj_version__": "1",
                "__ecaj_recipe__": "{}",
                "__ecaj_recipe_hash__": "match",
                "__ecaj_affected_keys__": json.dumps([]),
                "__ecaj_output_mode__": "full",
                "__ecaj_artifact_kind__": "diffusion",
                "__ecaj_source_model_kind__": "diffusion_model",
                "__ecaj_base_identity__": "base_id",
                "__ecaj_dependency_fingerprints__": dep_fps,
            },
        )

        recorded_cache: list[SavedModelProgress] = []

        def capture_cache(*, artifact_name: str):
            prog = SavedModelProgress(
                total_units=1,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            recorded_cache.append(prog)
            return prog

        with (
            patch(
                "nodes.exit.validate_model_name", return_value="cached_noop_progress.safetensors"
            ),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="match"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch(
                "nodes.exit._comfy_load_diffusion_model",
                return_value=MagicMock(name="loaded"),
            ),
            patch("nodes.exit._build_cache_reuse_progress", side_effect=capture_cache),
        ):
            WIDENExitNode().execute(base, save_model=True, model_name="cached_noop")

        assert len(recorded_cache) == 1
        prog = recorded_cache[0]
        phases = [phase for phase, _ in prog.messages]
        assert phases == ["cache_reuse"]
        assert prog.published is False


class TestExitNodeProgressFailure:
    """Failure during materialization must not report a published phase.

    AC: @streaming-materialization-progress ac-failure-status-not-success
    """

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_diffusion_save_failure_reports_failure_not_published(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        all_keys = list(mock_model_patcher.model_state_dict().keys())
        affected_keys = all_keys[:1]
        save_path = str(tmp_path / "fails.safetensors")

        mock_analyze, mock_model_analysis, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher,
            affected_keys,
            recipe=merge,
        )
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(**kwargs):
            raise RuntimeError("simulated mid-stream failure")

        recorded: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=RecordingProgressBar,
                artifact_name=artifact_name,
            )
            recorded.append(prog)
            return prog

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: affected_keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.validate_model_name", return_value="fails.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            with pytest.raises(RuntimeError, match="simulated mid-stream"):
                WIDENExitNode().execute(merge, save_model=True, model_name="fail")

        assert len(recorded) == 1
        prog = recorded[0]
        assert prog.failed is True
        assert prog.published is False
        phases = [phase for phase, _ in prog.messages]
        assert "failure" in phases
        # The artifact must not have been published.
        import os

        assert not os.path.exists(save_path), (
            "Failed materialization must not leave a published artifact"
        )
        # And no leftover temp files.
        temp_files = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert temp_files == []

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_noop_diffusion_save_failure_reports_failure_not_published(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noop_fail.safetensors")

        recorded: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=RecordingProgressBar,
                artifact_name=artifact_name,
            )
            recorded.append(prog)
            return prog

        # Make MaterializationSink.write_tensor raise after the first write
        # to simulate a failure during the no-op base copy.
        from lib.streaming_save import MaterializationSink as RealSink

        class FailingSink(RealSink):
            def __init__(self):
                super().__init__()
                self._writes = 0

            def write_tensor(self, name, tensor):
                self._writes += 1
                if self._writes >= 2:
                    raise OSError("simulated noop write failure")
                return super().write_tensor(name, tensor)

        with (
            patch("nodes.exit.validate_model_name", return_value="noop_fail.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
            patch("nodes.exit.MaterializationSink", FailingSink),
        ):
            with pytest.raises(OSError, match="noop write failure"):
                WIDENExitNode().execute(base, save_model=True, model_name="noop_fail")

        assert len(recorded) == 1
        prog = recorded[0]
        assert prog.failed is True
        assert prog.published is False
        # No published artifact, no leftover temp.
        import os

        assert not os.path.exists(save_path)
        leftover = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert leftover == []


class TestExitNodeProgressBarAbsence:
    """ProgressBar absence outside ComfyUI must remain safe.

    AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    """

    # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    def test_no_progress_bar_does_not_break_save(self, mock_model_patcher, tmp_path):
        """When ProgressBar is None (running outside ComfyUI), the no-op
        diffusion save still produces a valid published artifact."""
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noprogressbar.safetensors")

        with (
            patch("nodes.exit.validate_model_name", return_value="noprogressbar.safetensors"),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.ProgressBar", None),
        ):
            (result,) = WIDENExitNode().execute(
                base,
                save_model=True,
                model_name="noprogressbar",
            )

        assert result is not None
        import os

        assert os.path.exists(save_path)


class TestExitNodeFinalizationOrder:
    """The reported phase must already be 'finalize' while sink.finalize()
    is performing its long validation/fsync/atomic-replace work — not only
    after that work completes.

    AC: @streaming-materialization-progress ac-finalization-status-visible
    AC: @streaming-materialization-progress ac-failure-status-not-success
    """

    # AC: @streaming-materialization-progress ac-finalization-status-visible
    def test_noop_diffusion_save_phase_is_finalize_during_pre_publish_check(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        """During pre_publish_check (which runs inside sink.finalize, between
        fsync and atomic replace), the no-op diffusion save's reported phase
        must be 'finalize' and the artifact must NOT yet be marked
        published.  After sink.finalize returns, mark_published flips the
        flag without disturbing the visible phase.
        """
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noop_finalize_order.safetensors")

        captured: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            captured.append(prog)
            return prog

        observed: dict[str, object] = {}

        from nodes.exit import _classify_temp_artifact as _real_classify

        def spy_classify(p, *, expected_kind):
            prog = captured[0]
            observed["phase_during_pre_publish"] = prog.phase
            observed["published_during_pre_publish"] = prog.published
            return _real_classify(p, expected_kind=expected_kind)

        with (
            patch(
                "nodes.exit.validate_model_name", return_value="noop_finalize_order.safetensors"
            ),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._classify_temp_artifact", side_effect=spy_classify),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            WIDENExitNode().execute(
                base,
                save_model=True,
                model_name="noop_finalize_order",
            )

        # While sink.finalize was still running (validation/fsync done,
        # atomic replace not yet executed), the user-visible phase was
        # already 'finalize' and the artifact was NOT yet marked published.
        assert observed["phase_during_pre_publish"] == "finalize"
        assert observed["published_during_pre_publish"] is False

        # After sink.finalize returned successfully, mark_published flipped.
        prog = captured[0]
        assert prog.published is True
        # The visible phase progressed to 'reload' on the artifact reload.
        assert prog.phase == "reload"

    # AC: @streaming-materialization-progress ac-finalization-status-visible
    def test_diffusion_merge_save_phase_is_finalize_during_pre_publish_check(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        """Same ordering invariant for the normal merge save path:
        progress.finalize() precedes sink.finalize, so the reported phase
        is 'finalize' while validation/fsync/atomic replace is in flight.
        """
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        all_keys = list(mock_model_patcher.model_state_dict().keys())
        affected_keys = all_keys[:1]
        save_path = str(tmp_path / "merge_finalize_order.safetensors")

        mock_analyze, mock_model_analysis, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher,
            affected_keys,
            recipe=merge,
        )
        merged = {k: torch.randn(4, 4) for k in affected_keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(**kwargs):
            for k in kwargs["keys"]:
                kwargs["write_fn"](k, merged[k])

        captured: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            captured.append(prog)
            return prog

        observed: dict[str, object] = {}

        from nodes.exit import _classify_temp_artifact as _real_classify

        def spy_classify(p, *, expected_kind):
            prog = captured[0]
            observed["phase_during_pre_publish"] = prog.phase
            observed["published_during_pre_publish"] = prog.published
            return _real_classify(p, expected_kind=expected_kind)

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: affected_keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch(
                "nodes.exit.validate_model_name", return_value="merge_finalize_order.safetensors"
            ),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._classify_temp_artifact", side_effect=spy_classify),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            WIDENExitNode().execute(
                merge,
                save_model=True,
                model_name="merge_finalize_order",
            )

        assert observed["phase_during_pre_publish"] == "finalize"
        assert observed["published_during_pre_publish"] is False

        prog = captured[0]
        assert prog.published is True

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_noop_diffusion_save_pre_publish_failure_keeps_published_false(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        """A failure during sink.finalize (raised from pre_publish_check)
        must leave the artifact reported as NOT published, even though the
        finalize phase entry already happened — proving mark_published is
        gated on actual publication success rather than mere phase entry.
        """
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        save_path = str(tmp_path / "noop_pre_publish_fail.safetensors")

        captured: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            captured.append(prog)
            return prog

        phase_when_failed: dict[str, object] = {}

        def failing_classify(p, *, expected_kind):
            prog = captured[0]
            phase_when_failed["phase"] = prog.phase
            phase_when_failed["published"] = prog.published
            raise RuntimeError("simulated pre_publish_check rejection")

        with (
            patch(
                "nodes.exit.validate_model_name", return_value="noop_pre_publish_fail.safetensors"
            ),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.compute_recipe_hash", return_value="hash"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.serialize_recipe", return_value="{}"),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit._classify_temp_artifact", side_effect=failing_classify),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            with pytest.raises(RuntimeError, match="pre_publish_check"):
                WIDENExitNode().execute(
                    base,
                    save_model=True,
                    model_name="noop_pre_publish_fail",
                )

        # The phase entry happened before sink.finalize started its work.
        assert phase_when_failed["phase"] == "finalize"
        assert phase_when_failed["published"] is False

        # After the failure: published stays False, failure phase recorded.
        prog = captured[0]
        assert prog.failed is True
        assert prog.published is False
        phases = [phase for phase, _ in prog.messages]
        assert phases[-1] == "failure"

        # And no artifact was published or left behind as a temp file.
        import os

        assert not os.path.exists(save_path)
        leftover = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert leftover == []

    # AC: @streaming-materialization-progress ac-failure-status-not-success
    def test_diffusion_merge_save_pre_publish_failure_keeps_published_false(
        self,
        mock_model_patcher,
        tmp_path,
    ):
        """Same publication-gating invariant on the normal merge save path."""
        base = RecipeBase(
            model_patcher=mock_model_patcher,
            arch="sdxl",
            checkpoint_components=None,
        )
        lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},),
        )
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        all_keys = list(mock_model_patcher.model_state_dict().keys())
        affected_keys = all_keys[:1]
        save_path = str(tmp_path / "merge_pre_publish_fail.safetensors")

        mock_analyze, mock_model_analysis, dummy_plan = _make_full_mode_mocks(
            mock_model_patcher,
            affected_keys,
            recipe=merge,
        )
        merged = {k: torch.randn(4, 4) for k in affected_keys}
        sig = OpSignature(shape=(4, 4), ndim=2)

        def streaming_eval(**kwargs):
            for k in kwargs["keys"]:
                kwargs["write_fn"](k, merged[k])

        captured: list[SavedModelProgress] = []

        def capture_progress(*, manifest_size: int, artifact_name: str):
            prog = SavedModelProgress(
                total_units=manifest_size + 3,
                progress_bar_factory=None,
                artifact_name=artifact_name,
            )
            captured.append(prog)
            return prog

        phase_when_failed: dict[str, object] = {}

        def failing_classify(p, *, expected_kind):
            prog = captured[0]
            phase_when_failed["phase"] = prog.phase
            phase_when_failed["published"] = prog.published
            raise RuntimeError("simulated pre_publish_check rejection")

        _incremental_cache.clear()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: affected_keys}),
            patch("nodes.exit.streaming_evaluation_to_sink", side_effect=streaming_eval),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch(
                "nodes.exit.validate_model_name", return_value="merge_pre_publish_fail.safetensors"
            ),
            patch("nodes.exit._resolve_save_path", return_value=save_path),
            patch("nodes.exit.validate_checkpoint_components"),
            patch("nodes.exit.check_full_model_cache", return_value=False),
            patch("nodes.exit.check_ram_preflight"),
            patch("nodes.exit._classify_temp_artifact", side_effect=failing_classify),
            patch("nodes.exit._build_save_progress", side_effect=capture_progress),
        ):
            with pytest.raises(RuntimeError, match="pre_publish_check"):
                WIDENExitNode().execute(
                    merge,
                    save_model=True,
                    model_name="merge_pre_publish_fail",
                )

        assert phase_when_failed["phase"] == "finalize"
        assert phase_when_failed["published"] is False

        prog = captured[0]
        assert prog.failed is True
        assert prog.published is False

        import os

        assert not os.path.exists(save_path)
        leftover = [f for f in os.listdir(tmp_path) if f.startswith(".ecaj_tmp_")]
        assert leftover == []
