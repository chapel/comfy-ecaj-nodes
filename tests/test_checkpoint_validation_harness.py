"""Tests for the live ComfyUI checkpoint-save validation harness.

These tests validate guard behavior, report formatting, API classification,
and test-discovery exclusion. They NEVER submit real ComfyUI API prompts,
import ComfyUI modules, mutate a ComfyUI installation, start processes,
or write large artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Import harness module without triggering any ComfyUI work
# ---------------------------------------------------------------------------

_HARNESS_PATH = "scripts/manual/real_comfy_checkpoint_validation"


def _import_harness():
    """Import the harness module by path (no ComfyUI dependency)."""
    import importlib.util

    mod_name = "real_comfy_checkpoint_validation"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        f"{_HARNESS_PATH}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _import_harness()


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
# Guard refusal tests — harness refuses before any ComfyUI work
# ===========================================================================


class TestGuardRefusalNoEnvVar:
    """Guard refuses when COMFY_ECAJ_CHECKPOINT_VALIDATION is missing."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_missing_env_var_refuses(self):
        result = harness.check_guards(
            env={},
            argv=[
                "--run-checkpoint-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-model", "test.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("COMFY_ECAJ_CHECKPOINT_VALIDATION" in r for r in result.reasons)

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_wrong_env_var_value_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "0"},
            argv=[
                "--run-checkpoint-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-model", "test.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("COMFY_ECAJ_CHECKPOINT_VALIDATION" in r for r in result.reasons)


class TestGuardRefusalNoFlag:
    """Guard refuses when --run-checkpoint-validation flag is missing."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_missing_flag_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=[
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-model", "test.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--run-checkpoint-validation" in r for r in result.reasons)


class TestGuardRefusalMissingApiUrl:
    """Guard refuses when --comfy-api-url is missing."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_missing_api_url_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=[
                "--run-checkpoint-validation",
                "--source-model", "test.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--comfy-api-url" in r for r in result.reasons)


class TestGuardRefusalMissingModelName:
    """Guard refuses when --source-model is missing."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_missing_source_model_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=[
                "--run-checkpoint-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--source-model" in r for r in result.reasons)


class TestGuardRefusalMissingReportPath:
    """Guard refuses when --report-output is missing."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_missing_report_output_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=[
                "--run-checkpoint-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-model", "test.safetensors",
            ],
        )
        assert not result.passed
        assert any("--report-output" in r for r in result.reasons)


class TestGuardRefusalMultipleMissing:
    """Guard reports ALL missing inputs, not just the first."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_no_inputs_reports_all_reasons(self):
        result = harness.check_guards(env={}, argv=[])
        assert not result.passed
        # Should have reasons for: env var, flag, api-url, source-model, report-output
        assert len(result.reasons) == 5

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_partial_inputs_still_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=["--run-checkpoint-validation", "--comfy-api-url", "http://localhost:8188"],
        )
        assert not result.passed
        assert len(result.reasons) == 2  # missing source-model and report-output


class TestGuardRefusalBeforeComfyImport:
    """Guard failure must not import comfy, submit API prompts, or mutate anything."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_guard_failure_does_not_import_comfy(self):
        """check_guards() never imports comfy regardless of result."""
        original_modules = set(sys.modules.keys())
        result = harness.check_guards(env={}, argv=[])
        assert not result.passed
        new_modules = set(sys.modules.keys()) - original_modules
        comfy_modules = {m for m in new_modules if m.startswith("comfy")}
        assert not comfy_modules, f"Guard failure imported comfy modules: {comfy_modules}"

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_guard_failure_does_not_use_urllib(self):
        """check_guards() never makes HTTP requests."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = harness.check_guards(env={}, argv=[])
            assert not result.passed
            mock_urlopen.assert_not_called()

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    def test_main_returns_1_on_guard_failure(self):
        """main() returns 1 and does not proceed past guards."""
        exit_code = harness.main(argv=[])
        assert exit_code == 1


class TestGuardAllPass:
    """All guards pass when every input is provided."""

    def test_all_guards_pass(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_CHECKPOINT_VALIDATION": "1"},
            argv=[
                "--run-checkpoint-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-model", "test.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert result.passed
        assert result.reasons == ()


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-identifies-environment
# Report records ComfyUI version and active memory-management mode
# ===========================================================================


class TestReportIdentifiesEnvironment:
    """Report records ComfyUI version and memory mode."""

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_report_includes_comfy_version(self):
        report = harness.CheckpointValidationReport(
            comfy_version="0.3.4",
            memory_mode="normal",
        )
        text = harness.format_report(report)
        assert "0.3.4" in text
        assert "ComfyUI Version:" in text

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_report_includes_memory_mode(self):
        report = harness.CheckpointValidationReport(
            comfy_version="0.3.4",
            memory_mode="lowvram",
        )
        text = harness.format_report(report)
        assert "Memory Mode:" in text
        assert "lowvram" in text

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_extract_comfy_version_from_stats(self):
        stats = {"system": {"comfyui_version": "0.3.7"}}
        assert harness.extract_comfy_version(stats) == "0.3.7"

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_extract_memory_mode_from_stats(self):
        stats = {"devices": [{"vram_state": "normal"}]}
        assert harness.extract_memory_mode(stats) == "normal"

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_extract_memory_mode_unknown_when_no_devices(self):
        stats = {"devices": []}
        assert harness.extract_memory_mode(stats) == "unknown"

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_extract_node_classes_from_object_info(self):
        obj_info = {
            "WIDENEntry": {},
            "WIDENExit": {},
            "CheckpointLoaderSimple": {},
            "KSampler": {},
        }
        classes = harness.extract_node_classes(obj_info)
        assert "WIDENEntry" in classes
        assert "WIDENExit" in classes
        assert classes == sorted(classes)

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_report_json_includes_version_and_mode(self):
        report = harness.CheckpointValidationReport(
            comfy_version="0.3.4",
            memory_mode="highvram",
        )
        data = json.loads(report.to_json())
        assert data["comfy_version"] == "0.3.4"
        assert data["memory_mode"] == "highvram"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
# Report records queued save workflow shape and saved artifact classification
# ===========================================================================


class TestReportRecordsSaveOutcome:
    """Report records save workflow shape and artifact classification."""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_includes_terminal_save_result(self):
        report = harness.CheckpointValidationReport(
            terminal_save_result=harness.WorkflowResult(
                name="terminal_exit_save",
                accepted=True,
                prompt_id="abc123",
            ),
        )
        text = harness.format_report(report)
        assert "Terminal Save" in text
        assert "abc123" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_includes_checkpoint_save_result(self):
        report = harness.CheckpointValidationReport(
            checkpoint_save_result=harness.WorkflowResult(
                name="checkpoint_save",
                accepted=True,
                prompt_id="def456",
            ),
            saved_artifact_path="ecaj_checkpoint_validation_save",
            saved_artifact_classification="checkpoint",
        )
        text = harness.format_report(report)
        assert "Checkpoint Save" in text
        assert "def456" in text
        assert "ecaj_checkpoint_validation_save" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_json_includes_save_fields(self):
        report = harness.CheckpointValidationReport(
            terminal_save_result=harness.WorkflowResult(
                name="terminal_exit_save",
                accepted=False,
                error="scheduling: no outputs",
            ),
            saved_artifact_path="/path/to/artifact.safetensors",
            saved_artifact_classification="checkpoint",
        )
        data = json.loads(report.to_json())
        assert "terminal_save_result" in data
        assert data["terminal_save_result"]["accepted"] is False
        assert data["saved_artifact_path"] == "/path/to/artifact.safetensors"
        assert data["saved_artifact_classification"] == "checkpoint"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
# Report records loader and downstream workflow results
# ===========================================================================


class TestReportRecordsLoaderDownstreamOutcome:
    """Report records loader and downstream results for saved artifact."""

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_report_includes_loader_result(self):
        report = harness.CheckpointValidationReport(
            checkpoint_loader_result=harness.WorkflowResult(
                name="checkpoint_loader",
                accepted=True,
                prompt_id="load123",
            ),
        )
        text = harness.format_report(report)
        assert "Checkpoint Loader" in text
        assert "load123" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_report_includes_downstream_result(self):
        report = harness.CheckpointValidationReport(
            downstream_result=harness.WorkflowResult(
                name="downstream_ksampler",
                accepted=True,
                prompt_id="ks789",
            ),
        )
        text = harness.format_report(report)
        assert "Downstream KSampler" in text
        assert "ks789" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_report_json_includes_loader_and_downstream(self):
        report = harness.CheckpointValidationReport(
            checkpoint_loader_result=harness.WorkflowResult(
                name="checkpoint_loader",
                accepted=True,
                prompt_id="load456",
            ),
            downstream_result=harness.WorkflowResult(
                name="downstream_ksampler",
                accepted=False,
                error="ksampler failed",
            ),
        )
        data = json.loads(report.to_json())
        assert data["checkpoint_loader_result"]["accepted"] is True
        assert data["downstream_result"]["accepted"] is False
        assert "ksampler" in data["downstream_result"]["error"]


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
# Report records cache-reuse result
# ===========================================================================


class TestReportRecordsCacheReuseOutcome:
    """Report records cache-reuse result for saved artifact."""

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_report_includes_cache_reuse_result(self):
        report = harness.CheckpointValidationReport(
            cache_reuse_result=harness.WorkflowResult(
                name="cache_reuse",
                accepted=True,
                prompt_id="cache999",
            ),
        )
        text = harness.format_report(report)
        assert "Cache Reuse" in text
        assert "cache999" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_report_json_includes_cache_reuse(self):
        report = harness.CheckpointValidationReport(
            cache_reuse_result=harness.WorkflowResult(
                name="cache_reuse",
                accepted=True,
                prompt_id="cache789",
            ),
        )
        data = json.loads(report.to_json())
        assert data["cache_reuse_result"]["accepted"] is True
        assert data["cache_reuse_result"]["prompt_id"] == "cache789"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
# Memory mode is recorded but never changed for validation
# ===========================================================================


class TestMemoryModeNotChangedForSuccess:
    """Validation records memory mode but never requires changing it."""

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_report_records_memory_mode_as_is(self):
        """Report records the active memory mode without changing it."""
        report = harness.CheckpointValidationReport(
            memory_mode="lowvram",
        )
        data = json.loads(report.to_json())
        assert data["memory_mode"] == "lowvram"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_failure_categories_include_memory_mode(self):
        """Failure categories track memory_mode failures explicitly."""
        report = harness.CheckpointValidationReport(
            failure_categories={"memory_mode": "none"},
        )
        data = json.loads(report.to_json())
        assert "memory_mode" in data["failure_categories"]

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_failure_detects_memory_mode_issue(self):
        """classify_failure categorizes memory/vram errors as memory_mode."""
        result = harness.WorkflowResult(
            name="test",
            accepted=False,
            error="VRAM out of memory during save",
        )
        assert harness.classify_failure(result) == "memory_mode"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_failure_detects_oom(self):
        result = harness.WorkflowResult(
            name="test", accepted=False, error="OOM: cannot allocate"
        )
        assert harness.classify_failure(result) == "memory_mode"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_run_validation_records_memory_mode_without_changing_it(self):
        """run_validation queries memory mode from stats but does not modify it."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "lowvram"}],
        }
        mock_obj_info = {"WIDENEntry": {}, "WIDENExit": {}}

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value=mock_obj_info), \
             patch.object(harness, "submit_workflow", return_value=harness.WorkflowResult(
                 name="test", accepted=True, prompt_id="p1",
             )), \
             patch.object(harness, "check_cache_reuse", return_value=(True, "cached")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )
        assert report.memory_mode == "lowvram"
        assert report.failure_categories.get("memory_mode") == "none"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-primitive-tests-do-not-substitute-for-live-validation
# Default test discovery does not run real ComfyUI validation
# ===========================================================================


class TestPrimitiveTestsDoNotSubstitute:
    """Primitive tests are supporting evidence only; live validation required."""

    # AC: @live-comfy-saved-output-validation
    #     ac-primitive-tests-do-not-substitute-for-live-validation
    def test_harness_not_in_test_paths(self):
        """The harness script lives outside the pytest testpaths (tests/)."""
        assert not _HARNESS_PATH.startswith("tests/")

    # AC: @live-comfy-saved-output-validation
    #     ac-primitive-tests-do-not-substitute-for-live-validation
    def test_harness_has_no_test_functions(self):
        """The harness module contains no test_* functions or Test* classes."""
        members = dir(harness)
        test_items = [m for m in members if m.startswith("test_") or m.startswith("Test")]
        assert test_items == [], f"Harness contains test-like names: {test_items}"

    # AC: @live-comfy-saved-output-validation
    #     ac-primitive-tests-do-not-substitute-for-live-validation
    def test_pytest_collect_does_not_find_harness(self):
        """Running pytest --collect-only does not collect the harness."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
        )
        assert "real_comfy_checkpoint_validation" not in result.stdout

    # AC: @live-comfy-saved-output-validation
    #     ac-primitive-tests-do-not-substitute-for-live-validation
    def test_main_without_guards_does_not_run_validation(self):
        """Invoking main() without proper guards returns 1 (refusal)."""
        exit_code = harness.main(argv=[])
        assert exit_code == 1


# ===========================================================================
# Report schema validation — all required fields present
# ===========================================================================


class TestReportSchema:
    """Report schema includes all required fields."""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_has_all_required_fields(self):
        """Default report dict contains all required fields."""
        report = harness.CheckpointValidationReport()
        data = report.to_dict()
        ok, missing = harness.report_has_required_fields(data)
        assert ok, f"Missing required fields: {missing}"

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_required_fields_list(self):
        """The required fields set includes all spec-mandated fields."""
        required = harness._REQUIRED_REPORT_FIELDS
        assert "comfy_version" in required
        assert "memory_mode" in required
        assert "terminal_save_workflow_shape" in required
        assert "terminal_save_result" in required
        assert "checkpoint_save_workflow_shape" in required
        assert "saved_artifact_path" in required
        assert "saved_artifact_classification" in required
        assert "checkpoint_save_result" in required
        assert "checkpoint_loader_result" in required
        assert "downstream_result" in required
        assert "cache_reuse_result" in required
        assert "cache_reuse_detail" in required
        assert "failure_categories" in required


# ===========================================================================
# Mocked API classifications — unit test each failure category
# ===========================================================================


class TestMockedApiClassifications:
    """Mocked API classifications for distinct failure categories."""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_classify_scheduling_rejection(self):
        """Scheduling/no-output rejection is classified correctly."""
        result = harness.WorkflowResult(
            name="terminal",
            accepted=False,
            error="Prompt has no outputs: scheduler rejected",
        )
        assert harness.classify_failure(result) == "scheduling"

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_classify_save_failure(self):
        """Save failure is classified correctly."""
        result = harness.WorkflowResult(
            name="save",
            accepted=False,
            error="save_model failed: error writing checkpoint",
        )
        assert harness.classify_failure(result) == "save"

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_classify_loader_failure(self):
        """Loader failure is classified correctly."""
        result = harness.WorkflowResult(
            name="loader",
            accepted=False,
            error="checkpoint loader could not load model",
        )
        assert harness.classify_failure(result) == "loader"

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_classify_downstream_failure(self):
        """Downstream KSampler failure is classified correctly."""
        result = harness.WorkflowResult(
            name="downstream",
            accepted=False,
            error="KSampler inference failed",
        )
        assert harness.classify_failure(result) == "downstream"

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_classify_cache_failure(self):
        """Cache-reuse failure is classified correctly."""
        result = harness.WorkflowResult(
            name="cache",
            accepted=False,
            error="cache reuse failed: hash mismatch",
        )
        assert harness.classify_failure(result) == "cache"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_memory_mode_failure(self):
        """Memory-mode/OOM failure is classified correctly."""
        result = harness.WorkflowResult(
            name="test",
            accepted=False,
            error="OOM: insufficient VRAM for operation",
        )
        assert harness.classify_failure(result) == "memory_mode"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_oom_in_ksampler_node_is_memory_mode(self):
        """OOM inside a KSampler node_error is memory_mode, not downstream."""
        result = harness.WorkflowResult(
            name="downstream",
            accepted=False,
            error="node_error(KSampler): OOM: cannot allocate",
        )
        assert harness.classify_failure(result) == "memory_mode"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_cuda_oom_in_save_node_is_memory_mode(self):
        """CUDA out of memory inside a WIDENExit save node_error is memory_mode, not save."""
        result = harness.WorkflowResult(
            name="save",
            accepted=False,
            error="node_error(WIDENExit): save_model failed: CUDA out of memory",
        )
        assert harness.classify_failure(result) == "memory_mode"

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_classify_vram_in_loader_node_is_memory_mode(self):
        """VRAM error inside a loader node_error is memory_mode, not loader."""
        result = harness.WorkflowResult(
            name="loader",
            accepted=False,
            error="node_error(CheckpointLoaderSimple): VRAM allocation failed",
        )
        assert harness.classify_failure(result) == "memory_mode"

    def test_classify_accepted_is_none(self):
        """Accepted workflow has no failure."""
        result = harness.WorkflowResult(
            name="test",
            accepted=True,
            prompt_id="abc",
        )
        assert harness.classify_failure(result) == "none"

    def test_classify_unknown_error(self):
        """Unknown errors are classified as 'unknown'."""
        result = harness.WorkflowResult(
            name="test",
            accepted=False,
            error="something else entirely",
        )
        assert harness.classify_failure(result) == "unknown"

    def test_classify_connection_error(self):
        """Connection errors are classified correctly."""
        result = harness.WorkflowResult(
            name="test",
            accepted=False,
            error="url_error: connection refused",
        )
        assert harness.classify_failure(result) == "connection"


# ===========================================================================
# Mocked run_validation — full pipeline with mocked API
# ===========================================================================


class TestMockedRunValidation:
    """Full validation pipeline with mocked ComfyUI API."""

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_full_validation_with_mocked_api(self):
        """Full validation produces a complete report when API is mocked."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }
        mock_obj_info = {
            "WIDENEntry": {}, "WIDENExit": {},
            "CheckpointLoaderSimple": {}, "KSampler": {},
            "CLIPTextEncode": {}, "EmptyLatentImage": {},
            "VAEDecode": {}, "SaveImage": {},
        }

        call_count = 0

        def mock_submit(base_url, workflow, name):
            nonlocal call_count
            call_count += 1
            return harness.WorkflowResult(
                name=name,
                accepted=True,
                prompt_id=f"prompt_{call_count}",
            )

        def mock_check_cache(base_url, first_pid, reuse_pid, **kwargs):
            return True, f"WIDENExit node 3 cached in reuse prompt ({reuse_pid})"

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value=mock_obj_info), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", side_effect=mock_check_cache):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        # Verify environment
        assert report.comfy_version == "0.3.4"
        assert report.memory_mode == "normal"
        assert "WIDENEntry" in report.node_classes

        # Verify all workflow results recorded
        assert report.terminal_save_result.accepted
        assert report.checkpoint_save_result.accepted
        assert report.checkpoint_loader_result.accepted
        assert report.downstream_result.accepted
        assert report.cache_reuse_result.accepted

        # Verify artifact classification
        assert report.saved_artifact_classification == "checkpoint"
        assert report.saved_artifact_path == "ecaj_checkpoint_validation_save.safetensors"

        # Verify workflow shapes are recorded
        assert report.terminal_save_workflow_shape
        assert report.checkpoint_save_workflow_shape
        assert "WIDENEntry" in str(report.terminal_save_workflow_shape)
        assert "WIDENEntry" in str(report.checkpoint_save_workflow_shape)
        # Verify CLIP+VAE wiring is visible in shapes
        assert report.checkpoint_save_workflow_shape["2"]["inputs"]["clip"] == ["1", 1]
        assert report.checkpoint_save_workflow_shape["2"]["inputs"]["vae"] == ["1", 2]

        # Verify cache reuse detail is recorded
        assert report.cache_reuse_detail
        assert "cached" in report.cache_reuse_detail

        # Verify failure categories present
        assert "scheduling" in report.failure_categories
        assert "save" in report.failure_categories
        assert "loader" in report.failure_categories
        assert "downstream" in report.failure_categories
        assert "cache" in report.failure_categories
        assert "memory_mode" in report.failure_categories

        # Verify JSON roundtrip
        data = json.loads(report.to_json())
        ok, missing = harness.report_has_required_fields(data)
        assert ok, f"Missing required fields: {missing}"

    # AC: @live-comfy-saved-output-validation ac-report-identifies-environment
    def test_validation_with_failed_stats_returns_early(self):
        """If system_stats fails, report records error and returns early."""
        with patch.object(
            harness, "query_system_stats",
            side_effect=Exception("connection refused"),
        ):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )
        assert report.errors
        assert any("system_stats" in e for e in report.errors)
        assert report.comfy_version == ""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_validation_with_rejected_save(self):
        """When checkpoint save is rejected, artifact is not_saved and
        loader/downstream are skipped — not run against the source model."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        submitted_names = []

        def mock_submit(base_url, workflow, name):
            submitted_names.append(name)
            if name == "checkpoint_save":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="Prompt has no outputs",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="source.safetensors",
                report_output="/fake/report.json",
            )

        assert not report.checkpoint_save_result.accepted
        assert report.saved_artifact_classification == "not_saved"
        # Loader and downstream must NOT be submitted when save failed
        assert "downstream_ksampler" not in submitted_names
        # Loader/downstream results must indicate they were skipped
        assert not report.checkpoint_loader_result.accepted
        assert not report.downstream_result.accepted
        assert "skipped" in report.checkpoint_loader_result.error
        assert "skipped" in report.downstream_result.error


# ===========================================================================
# Workflow builder tests — verify workflow shapes
# ===========================================================================


class TestWorkflowBuilders:
    """Workflow builders produce valid ComfyUI API prompt structures."""

    def test_terminal_exit_save_workflow_shape(self):
        wf = harness.build_terminal_exit_save_workflow("test.safetensors")
        assert "1" in wf  # CheckpointLoaderSimple
        assert "2" in wf  # WIDENEntry
        assert "3" in wf  # WIDENExit
        assert wf["3"]["class_type"] == "WIDENExit"
        assert wf["3"]["inputs"]["save_model"] is True

    def test_terminal_exit_save_wires_clip_and_vae(self):
        wf = harness.build_terminal_exit_save_workflow("test.safetensors")
        entry_inputs = wf["2"]["inputs"]
        assert entry_inputs["model"] == ["1", 0]
        assert entry_inputs["clip"] == ["1", 1]
        assert entry_inputs["vae"] == ["1", 2]

    def test_checkpoint_save_workflow_shape(self):
        wf = harness.build_checkpoint_save_workflow("test.safetensors")
        assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
        assert wf["3"]["inputs"]["save_model"] is True

    def test_checkpoint_save_wires_clip_and_vae(self):
        wf = harness.build_checkpoint_save_workflow("test.safetensors")
        entry_inputs = wf["2"]["inputs"]
        assert entry_inputs["model"] == ["1", 0]
        assert entry_inputs["clip"] == ["1", 1]
        assert entry_inputs["vae"] == ["1", 2]

    def test_downstream_workflow_shape(self):
        wf = harness.build_downstream_workflow("test.safetensors")
        class_types = {v["class_type"] for v in wf.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "CLIPTextEncode" in class_types
        assert "EmptyLatentImage" in class_types
        assert "KSampler" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types

    def test_downstream_workflow_safe_defaults(self):
        wf = harness.build_downstream_workflow("test.safetensors")
        # Find the KSampler node
        ks_node = next(v for v in wf.values() if v["class_type"] == "KSampler")
        assert ks_node["inputs"]["steps"] == 1
        assert ks_node["inputs"]["cfg"] == 1.0
        assert ks_node["inputs"]["seed"] == 42
        assert ks_node["inputs"]["sampler_name"] == "euler"
        assert ks_node["inputs"]["scheduler"] == "normal"
        # Find EmptyLatentImage
        lat_node = next(v for v in wf.values() if v["class_type"] == "EmptyLatentImage")
        assert lat_node["inputs"]["width"] == 256
        assert lat_node["inputs"]["height"] == 256
        assert lat_node["inputs"]["batch_size"] == 1

    def test_downstream_workflow_custom_params(self):
        wf = harness.build_downstream_workflow(
            "test.safetensors",
            width=512, height=512, steps=5, cfg=7.5,
            seed=123, sampler_name="dpmpp_2m", scheduler="karras",
            batch_size=2,
        )
        ks_node = next(v for v in wf.values() if v["class_type"] == "KSampler")
        assert ks_node["inputs"]["steps"] == 5
        assert ks_node["inputs"]["cfg"] == 7.5
        lat_node = next(v for v in wf.values() if v["class_type"] == "EmptyLatentImage")
        assert lat_node["inputs"]["width"] == 512
        assert lat_node["inputs"]["batch_size"] == 2

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_checkpoint_save_workflow_disables_cache(self):
        """Checkpoint save workflow disables cache to force a fresh save."""
        save_wf = harness.build_checkpoint_save_workflow("test.safetensors")
        exit_inputs = save_wf["3"]["inputs"]
        assert exit_inputs["enable_cache"] is False
        assert exit_inputs["save_model"] is True

    def test_cache_reuse_workflow_has_enable_cache(self):
        cache_wf = harness.build_cache_reuse_workflow("test.safetensors")
        exit_inputs = cache_wf["3"]["inputs"]
        assert exit_inputs["enable_cache"] is True
        assert exit_inputs["save_model"] is True

    def test_cache_reuse_workflow_wires_clip_and_vae(self):
        cache_wf = harness.build_cache_reuse_workflow("test.safetensors")
        entry_inputs = cache_wf["2"]["inputs"]
        assert entry_inputs["model"] == ["1", 0]
        assert entry_inputs["clip"] == ["1", 1]
        assert entry_inputs["vae"] == ["1", 2]

    def test_cache_reuse_workflow_same_model_name_as_save(self):
        save_wf = harness.build_checkpoint_save_workflow("test.safetensors")
        cache_wf = harness.build_cache_reuse_workflow("test.safetensors")
        assert (
            cache_wf["3"]["inputs"]["model_name"]
            == save_wf["3"]["inputs"]["model_name"]
        )


# ===========================================================================
# Refusal report formatting
# ===========================================================================


class TestRefusalReport:
    """Refusal report includes clear reasons and guarantees."""

    def test_refusal_report_includes_reasons(self):
        guard = harness.GuardResult(
            passed=False,
            reasons=("Missing env var.", "Missing flag."),
        )
        text = harness.format_refusal(guard)
        assert "REFUSED" in text
        assert "Missing env var." in text
        assert "Missing flag." in text
        assert "No ComfyUI API prompts were submitted" in text
        assert "No ComfyUI modules were imported" in text
        assert "No ComfyUI installation was modified" in text
        assert "No processes were started" in text
        assert "No large artifacts were written" in text


# ===========================================================================
# Report JSON roundtrip
# ===========================================================================


class TestReportJsonRoundtrip:
    """Report serializes to JSON and includes all data."""

    def test_full_report_json_roundtrip(self):
        report = harness.CheckpointValidationReport(
            comfy_version="0.3.4",
            memory_mode="normal",
            comfy_api_url="http://localhost:8188",
            source_model="test.safetensors",
            node_classes=["WIDENEntry", "WIDENExit"],
            terminal_save_workflow_shape={"1": {"class_type": "CheckpointLoaderSimple"}},
            terminal_save_result=harness.WorkflowResult(
                name="terminal", accepted=True, prompt_id="t1",
            ),
            checkpoint_save_workflow_shape={"1": {"class_type": "CheckpointLoaderSimple"}},
            checkpoint_save_result=harness.WorkflowResult(
                name="save", accepted=True, prompt_id="s1",
            ),
            saved_artifact_path="/path/artifact.safetensors",
            saved_artifact_classification="checkpoint",
            checkpoint_loader_result=harness.WorkflowResult(
                name="loader", accepted=True, prompt_id="l1",
            ),
            downstream_result=harness.WorkflowResult(
                name="downstream", accepted=True, prompt_id="d1",
            ),
            cache_reuse_result=harness.WorkflowResult(
                name="cache", accepted=True, prompt_id="c1",
            ),
            cache_reuse_detail="WIDENExit cached in reuse prompt",
            failure_categories={"scheduling": "none", "save": "none"},
            duration_seconds=2.5,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["comfy_version"] == "0.3.4"
        assert data["memory_mode"] == "normal"
        assert data["terminal_save_result"]["prompt_id"] == "t1"
        assert data["checkpoint_save_result"]["prompt_id"] == "s1"
        assert data["checkpoint_loader_result"]["prompt_id"] == "l1"
        assert data["downstream_result"]["prompt_id"] == "d1"
        assert data["cache_reuse_result"]["prompt_id"] == "c1"
        terminal_shape = data["terminal_save_workflow_shape"]
        assert terminal_shape["1"]["class_type"] == "CheckpointLoaderSimple"
        ckpt_shape = data["checkpoint_save_workflow_shape"]
        assert ckpt_shape["1"]["class_type"] == "CheckpointLoaderSimple"
        assert data["cache_reuse_detail"] == "WIDENExit cached in reuse prompt"
        assert data["duration_seconds"] == 2.5


# ===========================================================================
# Cache reuse detection tests
# ===========================================================================


class TestCheckCacheReuse:
    """check_cache_reuse correctly identifies cache hit/miss from history."""

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_reused_when_exit_node_not_in_reuse_outputs(self):
        """Cache reuse detected when WIDENExit is absent from reuse outputs."""
        first_history = {
            "outputs": {"3": {"result": "saved"}},
            "status": {"messages": []},
        }
        reuse_history = {
            "outputs": {},
            "status": {"messages": []},
        }

        def mock_query(base_url, prompt_id, **kwargs):
            if prompt_id == "first":
                return first_history
            return reuse_history

        with patch.object(harness, "query_prompt_history", side_effect=mock_query):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is True
        assert "cached" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_reused_via_execution_cached_message(self):
        """Cache reuse detected via execution_cached status message."""
        first_history = {
            "outputs": {"3": {"result": "saved"}},
            "status": {"messages": []},
        }
        reuse_history = {
            "outputs": {"3": {"result": "saved"}},
            "status": {
                "messages": [
                    ["execution_cached", {"nodes": ["1", "2", "3"]}],
                ],
            },
        }

        def mock_query(base_url, prompt_id, **kwargs):
            if prompt_id == "first":
                return first_history
            return reuse_history

        with patch.object(harness, "query_prompt_history", side_effect=mock_query):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is True
        assert "execution_cached" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_not_reused_when_exit_re_executed(self):
        """Cache NOT reused when WIDENExit produces outputs in both prompts."""
        first_history = {
            "outputs": {"3": {"result": "saved"}},
            "status": {"messages": []},
        }
        reuse_history = {
            "outputs": {"3": {"result": "saved_again"}},
            "status": {"messages": []},
        }

        def mock_query(base_url, prompt_id, **kwargs):
            if prompt_id == "first":
                return first_history
            return reuse_history

        with patch.object(harness, "query_prompt_history", side_effect=mock_query):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is False
        assert "NOT reused" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_check_handles_unavailable_history(self):
        """Returns False with detail when history is unavailable."""
        with patch.object(harness, "query_prompt_history", return_value={}):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is False
        assert "unavailable" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_check_handles_exception(self):
        """Returns False with detail when an exception occurs."""
        with patch.object(
            harness, "query_prompt_history",
            side_effect=Exception("connection timeout"),
        ):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is False
        assert "failed" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_reused_via_timing_speedup(self):
        """Cache reuse detected via timing when exit node has no UI outputs."""
        # WIDENExit returns MODEL (not a UI type), so neither prompt has
        # exit node outputs.  Timing comparison detects application-level cache.
        no_exit_outputs_history = {
            "outputs": {},
            "status": {"messages": []},
        }

        with patch.object(
            harness, "query_prompt_history", return_value=no_exit_outputs_history,
        ):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
                first_elapsed=4.0, reuse_elapsed=0.7,
            )
        assert reused is True
        assert "timing" in detail
        assert "speedup=" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_not_reused_when_timing_too_slow(self):
        """Cache NOT reused when timing speedup is below threshold."""
        no_exit_outputs_history = {
            "outputs": {},
            "status": {"messages": []},
        }

        with patch.object(
            harness, "query_prompt_history", return_value=no_exit_outputs_history,
        ):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
                first_elapsed=4.0, reuse_elapsed=3.5,
            )
        assert reused is False
        assert "timing does not indicate" in detail

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_cache_undetermined_without_timing_data(self):
        """Returns False with descriptive detail when no timing data is available."""
        no_exit_outputs_history = {
            "outputs": {},
            "status": {"messages": []},
        }

        with patch.object(
            harness, "query_prompt_history", return_value=no_exit_outputs_history,
        ):
            reused, detail = harness.check_cache_reuse(
                "http://fake:8188", "first", "reuse",
            )
        assert reused is False
        assert "no timing data" in detail


# ===========================================================================
# Downstream loads saved artifact (not source)
# ===========================================================================


class TestDownstreamLoadsSavedArtifact:
    """Downstream workflow must load the saved artifact, not the source model."""

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_downstream_uses_saved_artifact_filename(self):
        """build_downstream_workflow with saved artifact loads it, not source."""
        wf = harness.build_downstream_workflow(
            "ecaj_checkpoint_validation_save.safetensors",
        )
        loader_node = wf["1"]
        assert loader_node["class_type"] == "CheckpointLoaderSimple"
        assert loader_node["inputs"]["ckpt_name"] == "ecaj_checkpoint_validation_save.safetensors"

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_run_validation_passes_saved_artifact_to_downstream(self):
        """run_validation uses saved artifact filename in downstream workflow."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        submitted_workflows = []

        def mock_submit(base_url, workflow, name):
            submitted_workflows.append((name, workflow))
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id=f"p_{name}",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(True, "cached")):
            harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="source.safetensors",
                report_output="/fake/report.json",
            )

        # Find the downstream workflow submission
        downstream_submissions = [
            (name, wf) for name, wf in submitted_workflows
            if name == "downstream_ksampler"
        ]
        assert len(downstream_submissions) == 1
        _, downstream_wf = downstream_submissions[0]
        # The downstream workflow must load the saved artifact, NOT source.safetensors
        loader_ckpt = downstream_wf["1"]["inputs"]["ckpt_name"]
        assert loader_ckpt == "ecaj_checkpoint_validation_save.safetensors"
        assert loader_ckpt != "source.safetensors"


# ===========================================================================
# Workflow shape recording tests
# ===========================================================================


class TestWorkflowShapeRecording:
    """Report records the queued workflow shapes for save prompts."""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_json_includes_terminal_save_workflow_shape(self):
        """Report JSON includes terminal_save_workflow_shape field."""
        shape = harness.build_terminal_exit_save_workflow("test.safetensors")
        report = harness.CheckpointValidationReport(
            terminal_save_workflow_shape=shape,
        )
        data = json.loads(report.to_json())
        assert "terminal_save_workflow_shape" in data
        assert data["terminal_save_workflow_shape"]["2"]["inputs"]["clip"] == ["1", 1]

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_report_json_includes_checkpoint_save_workflow_shape(self):
        """Report JSON includes checkpoint_save_workflow_shape field."""
        shape = harness.build_checkpoint_save_workflow("test.safetensors")
        report = harness.CheckpointValidationReport(
            checkpoint_save_workflow_shape=shape,
        )
        data = json.loads(report.to_json())
        assert "checkpoint_save_workflow_shape" in data
        assert data["checkpoint_save_workflow_shape"]["2"]["inputs"]["vae"] == ["1", 2]

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_workflow_shapes_show_clip_vae_wiring(self):
        """Workflow shapes prove MODEL+CLIP+VAE are wired correctly."""
        save_shape = harness.build_checkpoint_save_workflow("test.safetensors")
        entry_inputs = save_shape["2"]["inputs"]
        assert entry_inputs["model"] == ["1", 0]
        assert entry_inputs["clip"] == ["1", 1]
        assert entry_inputs["vae"] == ["1", 2]


# ===========================================================================
# submit_workflow polls /history for execution completion
# ===========================================================================


class TestSubmitWorkflowHistoryPolling:
    """submit_workflow waits for /history completion and detects runtime errors."""

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_accepted_prompt_with_successful_execution(self):
        """Prompt accepted and /history shows successful completion."""
        mock_prompt_resp = {"prompt_id": "p1"}
        mock_history = {
            "outputs": {"3": {"result": "saved"}},
            "status": {"status_str": "success", "messages": []},
        }

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp), \
             patch.object(harness, "query_prompt_history", return_value=mock_history):
            result = harness.submit_workflow("http://fake:8188", {}, "test_wf")

        assert result.accepted is True
        assert result.prompt_id == "p1"
        assert result.error == ""
        assert result.outputs == {"3": {"result": "saved"}}

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_accepted_prompt_with_runtime_node_error(self):
        """Prompt accepted but node error detected in /history — not accepted."""
        mock_prompt_resp = {"prompt_id": "p1"}
        mock_history = {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [
                    ["execution_error", {
                        "node_type": "WIDENExit",
                        "exception_message": "save_model failed: disk full",
                    }],
                ],
            },
        }

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp), \
             patch.object(harness, "query_prompt_history", return_value=mock_history):
            result = harness.submit_workflow("http://fake:8188", {}, "test_wf")

        assert result.accepted is False
        assert result.prompt_id == "p1"
        assert "node_error(WIDENExit)" in result.error
        assert "save_model failed" in result.error

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_accepted_prompt_with_execution_timeout(self):
        """Prompt accepted but /history never shows completion."""
        mock_prompt_resp = {"prompt_id": "p1"}

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp), \
             patch.object(harness, "query_prompt_history", return_value={}):
            result = harness.submit_workflow("http://fake:8188", {}, "test_wf")

        assert result.accepted is False
        assert result.prompt_id == "p1"
        assert "execution_timeout" in result.error

    # AC: @live-comfy-saved-output-validation ac-report-records-save-outcome
    def test_rejected_prompt_no_prompt_id(self):
        """Prompt rejected at /prompt level (no prompt_id)."""
        mock_prompt_resp = {
            "error": {"message": "invalid workflow"},
            "node_errors": {},
        }

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp):
            result = harness.submit_workflow("http://fake:8188", {}, "test_wf")

        assert result.accepted is False
        assert result.prompt_id == ""

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_runtime_ksampler_error_detected(self):
        """Runtime KSampler error in /history is classified as failure."""
        mock_prompt_resp = {"prompt_id": "p_ks"}
        mock_history = {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [
                    ["execution_error", {
                        "node_type": "KSampler",
                        "exception_message": "KSampler inference failed: NaN in latents",
                    }],
                ],
            },
        }

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp), \
             patch.object(harness, "query_prompt_history", return_value=mock_history):
            result = harness.submit_workflow("http://fake:8188", {}, "downstream")

        assert result.accepted is False
        assert "KSampler" in result.error

    # AC: @live-comfy-saved-output-validation ac-report-records-cache-reuse-outcome
    def test_submit_workflow_captures_elapsed_time(self):
        """submit_workflow records wall-clock elapsed_seconds on the result."""
        mock_prompt_resp = {"prompt_id": "p_elapsed"}
        mock_history = {
            "outputs": {},
            "status": {"status_str": "success", "messages": []},
        }

        with patch.object(harness, "_api_post_prompt", return_value=mock_prompt_resp), \
             patch.object(harness, "query_prompt_history", return_value=mock_history):
            result = harness.submit_workflow("http://fake:8188", {}, "test_wf")

        assert result.elapsed_seconds > 0


class TestExtractNodeErrors:
    """_extract_node_errors_from_history classifies history entries."""

    def test_successful_execution_returns_empty(self):
        history = {
            "outputs": {"3": {}},
            "status": {"status_str": "success", "messages": []},
        }
        assert harness._extract_node_errors_from_history(history) == ""

    def test_error_with_execution_error_message(self):
        history = {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [
                    ["execution_error", {
                        "node_type": "SaveImage",
                        "exception_message": "disk full",
                    }],
                ],
            },
        }
        result = harness._extract_node_errors_from_history(history)
        assert "node_error(SaveImage)" in result
        assert "disk full" in result

    def test_error_without_execution_error_message(self):
        history = {
            "outputs": {},
            "status": {"status_str": "error", "messages": []},
        }
        result = harness._extract_node_errors_from_history(history)
        assert "execution_error" in result

    def test_no_status_returns_empty(self):
        assert harness._extract_node_errors_from_history({}) == ""


# ===========================================================================
# _extract_failing_node_type — parse node class from node_error() strings
# ===========================================================================


class TestExtractFailingNodeType:
    """_extract_failing_node_type parses the failing node class from error strings."""

    def test_ksampler_node_error(self):
        assert harness._extract_failing_node_type(
            "node_error(KSampler): inference failed",
        ) == "KSampler"

    def test_checkpoint_loader_node_error(self):
        assert harness._extract_failing_node_type(
            "node_error(CheckpointLoaderSimple): could not load model",
        ) == "CheckpointLoaderSimple"

    def test_widen_exit_node_error(self):
        assert harness._extract_failing_node_type(
            "node_error(WIDENExit): save failed",
        ) == "WIDENExit"

    def test_non_node_error_returns_empty(self):
        error = "execution_timeout: prompt never completed"
        assert harness._extract_failing_node_type(error) == ""

    def test_empty_string_returns_empty(self):
        assert harness._extract_failing_node_type("") == ""


# ===========================================================================
# Loader/downstream outcome distinction
# ===========================================================================


class TestLoaderDownstreamDistinction:
    """Loader and downstream outcomes must be distinct in run_validation.

    When a downstream node (KSampler, VAEDecode) fails, the checkpoint loader
    result must remain accepted=True — the loader succeeded, only the
    downstream step failed.  Conversely, when CheckpointLoaderSimple fails,
    the loader result is accepted=False and downstream is skipped.
    """

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_ksampler_failure_loader_still_accepted(self):
        """When KSampler fails, checkpoint_loader_result.accepted is True."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            if name == "downstream_ksampler":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="node_error(KSampler): KSampler inference failed",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        # Loader succeeded — the failing node was KSampler, not the loader
        assert report.checkpoint_loader_result.accepted is True
        assert report.failure_categories["loader"] == "none"
        # Downstream failed
        assert report.downstream_result.accepted is False
        assert "KSampler" in report.downstream_result.error
        assert report.failure_categories["downstream"] == "downstream"

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_checkpoint_loader_failure_downstream_skipped(self):
        """When CheckpointLoaderSimple fails, downstream is skipped."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            if name == "downstream_ksampler":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="node_error(CheckpointLoaderSimple): could not load model",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        # Loader failed
        assert report.checkpoint_loader_result.accepted is False
        assert "CheckpointLoaderSimple" in report.checkpoint_loader_result.error
        assert report.failure_categories["loader"] == "loader"
        # Downstream was skipped because the loader failed
        assert report.downstream_result.accepted is False
        assert "skipped" in report.downstream_result.error

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_both_succeed_when_downstream_succeeds(self):
        """When everything succeeds, both loader and downstream are accepted."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id=f"p_{name}",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(True, "cached")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        assert report.checkpoint_loader_result.accepted is True
        assert report.downstream_result.accepted is True
        assert report.failure_categories["loader"] == "none"
        assert report.failure_categories["downstream"] == "none"

    # AC: @live-comfy-saved-output-validation ac-report-records-loader-downstream-outcome
    def test_vaedecode_failure_loader_still_accepted(self):
        """When VAEDecode fails, the loader still succeeded."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            if name == "downstream_ksampler":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="node_error(VAEDecode): decode failed",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        assert report.checkpoint_loader_result.accepted is True
        assert report.failure_categories["loader"] == "none"
        assert report.downstream_result.accepted is False
        assert "VAEDecode" in report.downstream_result.error


# ===========================================================================
# Memory-mode failure propagation
# ===========================================================================


class TestMemoryModeFailurePropagation:
    """memory_mode failure category propagates from per-workflow classifications."""

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_memory_mode_failure_propagated_from_save(self):
        """When save fails with VRAM error, memory_mode category is set."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            if name == "checkpoint_save":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="VRAM out of memory during save",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        # The save workflow was classified as memory_mode failure
        assert report.failure_categories["save"] == "memory_mode"
        # The top-level memory_mode category must NOT be "none"
        assert report.failure_categories["memory_mode"] != "none"
        assert "save" in report.failure_categories["memory_mode"]

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_memory_mode_failure_propagated_from_downstream_oom(self):
        """When downstream KSampler fails with OOM, memory_mode category is set.

        Reproduces the exact reviewer scenario: node_error(KSampler): OOM
        must be classified as memory_mode, not downstream — even though
        'ksampler' appears in the error string.
        """
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        def mock_submit(base_url, workflow, name):
            if name == "downstream_ksampler":
                return harness.WorkflowResult(
                    name=name, accepted=False,
                    error="node_error(KSampler): OOM: cannot allocate",
                )
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id="p1",
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        # The downstream workflow was classified as memory_mode, not downstream
        assert report.failure_categories["downstream"] == "memory_mode"
        # The top-level memory_mode category surfaces the failure
        assert report.failure_categories["memory_mode"] != "none"
        assert "downstream" in report.failure_categories["memory_mode"]

    # AC: @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success
    def test_memory_mode_none_when_no_vram_failures(self):
        """When no workflow has VRAM errors, memory_mode is 'none'."""
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", return_value=harness.WorkflowResult(
                 name="test", accepted=True, prompt_id="p1",
             )), \
             patch.object(harness, "check_cache_reuse", return_value=(True, "cached")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="test.safetensors",
                report_output="/fake/report.json",
            )

        assert report.failure_categories["memory_mode"] == "none"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
# AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
# AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
# New memory-validation inputs must not bypass the existing guards.
# ===========================================================================


class TestNewMemoryArgsDoNotBypassGuards:
    """Adding --comfy-pid / --comfy-log-path / --repeat-cache-miss-runs
    must not satisfy the required guards or trigger any process/log reads."""

    # AC: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in
    # AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
    def test_new_args_alone_do_not_satisfy_guards(self):
        result = harness.check_guards(
            env={},
            argv=[
                "--comfy-pid", "12345",
                "--comfy-log-path", "/fake/comfy.log",
                "--repeat-cache-miss-runs", "3",
                "--memory-accumulation-threshold-mb", "512",
            ],
        )
        assert not result.passed
        # Still missing env var, flag, api url, source model, report output
        assert len(result.reasons) == 5

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_guard_failure_does_not_read_procfs(self, tmp_path):
        """Guard refusal must not read /proc when --comfy-pid is supplied."""
        with patch(
            "builtins.open",
            side_effect=AssertionError("guard must not read /proc"),
        ):
            # The fact that check_guards does NOT call open() proves it does
            # not touch procfs. patch builtins.open globally to catch any
            # opens at this layer.
            result = harness.check_guards(
                env={},
                argv=[
                    "--comfy-pid", "12345",
                    "--comfy-log-path", str(tmp_path / "fake.log"),
                ],
            )
        assert not result.passed

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_main_with_only_new_args_refuses(self):
        """main() with only new memory args still returns 1 (refusal)."""
        exit_code = harness.main(
            argv=[
                "--comfy-pid", "999",
                "--comfy-log-path", "/fake/comfy.log",
                "--repeat-cache-miss-runs", "2",
            ],
        )
        assert exit_code == 1


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
# AC: @comfy-memory-manager-compatibility ac-report-os-memory-observations
# Process-memory observation helpers — procfs reading and unavailable handling
# ===========================================================================


class TestProcessMemoryHelpers:
    """``collect_process_memory_observation`` reads procfs safely."""

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_no_pid_returns_unavailable_observation(self):
        obs = harness.collect_process_memory_observation(None, "before-run")
        assert obs.available is False
        assert obs.error
        assert obs.label == "before-run"
        assert obs.rss_kb is None
        assert obs.swap_kb is None
        assert obs.anonymous_kb is None

    # AC: @comfy-memory-manager-compatibility ac-report-os-memory-observations
    def test_smaps_rollup_supplies_anonymous_and_swap(self, tmp_path):
        # Build a fake procfs tree
        pid = 7777
        proc = tmp_path / "proc" / str(pid)
        proc.mkdir(parents=True)
        (proc / "smaps_rollup").write_text(
            "Rss:               12345 kB\n"
            "Swap:              4096 kB\n"
            "Anonymous:        20000 kB\n",
        )

        # Patch the path-style helpers via a /proc redirect
        orig_isfile = os.path.exists
        orig_open = open

        def fake_exists(p):
            if p == f"/proc/{pid}/smaps_rollup":
                return (proc / "smaps_rollup").exists()
            if p == f"/proc/{pid}/status":
                return (proc / "status").exists()
            return orig_isfile(p)

        def fake_open(p, *a, **kw):
            if p == f"/proc/{pid}/smaps_rollup":
                return orig_open(str(proc / "smaps_rollup"), *a, **kw)
            if p == f"/proc/{pid}/status":
                return orig_open(str(proc / "status"), *a, **kw)
            return orig_open(p, *a, **kw)

        with patch.object(harness.os.path, "exists", side_effect=fake_exists), \
             patch("builtins.open", side_effect=fake_open):
            obs = harness.collect_process_memory_observation(pid, "after-save")

        assert obs.available is True
        assert obs.source == "smaps_rollup"
        assert obs.rss_kb == 12345
        assert obs.swap_kb == 4096
        assert obs.anonymous_kb == 20000

    # AC: @comfy-memory-manager-compatibility ac-report-os-memory-observations
    def test_falls_back_to_status_when_rollup_missing(self, tmp_path):
        pid = 8888
        proc = tmp_path / "proc" / str(pid)
        proc.mkdir(parents=True)
        # Only /proc/<pid>/status exists, no smaps_rollup
        (proc / "status").write_text(
            "Name:   python\n"
            "VmRSS:  9876 kB\n"
            "VmSwap: 256 kB\n",
        )

        orig_exists = os.path.exists
        orig_open = open

        def fake_exists(p):
            if p == f"/proc/{pid}/smaps_rollup":
                return False
            if p == f"/proc/{pid}/status":
                return True
            return orig_exists(p)

        def fake_open(p, *a, **kw):
            if p == f"/proc/{pid}/status":
                return orig_open(str(proc / "status"), *a, **kw)
            return orig_open(p, *a, **kw)

        with patch.object(harness.os.path, "exists", side_effect=fake_exists), \
             patch("builtins.open", side_effect=fake_open):
            obs = harness.collect_process_memory_observation(pid, "after-return")

        assert obs.available is True
        assert obs.source == "status"
        assert obs.rss_kb == 9876
        assert obs.swap_kb == 256
        assert obs.anonymous_kb is None  # status does not expose Anonymous

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_unreadable_procfs_is_recorded_as_unavailable(self):
        # Use a pid that almost certainly does not exist on this machine.
        # (Process pids on Linux are 32-bit; 2147483646 is above any real pid.)
        obs = harness.collect_process_memory_observation(2147483646, "before-run")
        assert obs.available is False
        assert obs.rss_kb is None
        # Must not have raised — error string is populated instead.
        assert obs.error
        assert obs.label == "before-run"


# ===========================================================================
# Log-parsing helpers — WIDEN [mem] label extraction
# ===========================================================================


class TestParseWidenMemoryLog:
    """``parse_widen_memory_log`` extracts labeled RSS/VRAM observations."""

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_parses_after_checkpoint_save_label(self):
        log_text = (
            "2026-05-13 12:00:01 INFO ecaj.exit "
            "[mem] after-checkpoint-save: RSS=4096MB "
            "VRAM=2048MB(reserved=3072MB)\n"
        )
        parsed = harness.parse_widen_memory_log(
            log_text,
            ("after-checkpoint-save", "after-checkpoint-temp-model-release"),
        )
        assert parsed["after-checkpoint-save"].available is True
        assert parsed["after-checkpoint-save"].rss_mb == 4096
        assert parsed["after-checkpoint-save"].vram_alloc_mb == 2048
        assert parsed["after-checkpoint-save"].vram_reserved_mb == 3072

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_missing_labels_are_recorded_as_unavailable(self):
        log_text = "2026-05-13 INFO [mem] before-run: RSS=2000MB\n"
        parsed = harness.parse_widen_memory_log(
            log_text,
            ("after-checkpoint-save", "after-checkpoint-temp-model-release"),
        )
        assert parsed["after-checkpoint-save"].available is False
        assert parsed["after-checkpoint-save"].error
        assert parsed["after-checkpoint-temp-model-release"].available is False

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_handles_log_without_vram(self):
        log_text = "[mem] after-checkpoint-temp-model-release: RSS=3500MB\n"
        parsed = harness.parse_widen_memory_log(
            log_text,
            ("after-checkpoint-temp-model-release",),
        )
        obs = parsed["after-checkpoint-temp-model-release"]
        assert obs.available is True
        assert obs.rss_mb == 3500
        assert obs.vram_alloc_mb is None
        assert obs.vram_reserved_mb is None

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_keeps_last_occurrence_when_label_repeats(self):
        log_text = (
            "[mem] after-checkpoint-save: RSS=1000MB\n"
            "[mem] after-checkpoint-save: RSS=2500MB VRAM=500MB(reserved=600MB)\n"
        )
        parsed = harness.parse_widen_memory_log(
            log_text, ("after-checkpoint-save",),
        )
        obs = parsed["after-checkpoint-save"]
        assert obs.available is True
        assert obs.rss_mb == 2500
        assert obs.vram_alloc_mb == 500

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_empty_log_records_all_labels_unavailable(self):
        parsed = harness.parse_widen_memory_log(
            "",
            ("after-checkpoint-save", "after-checkpoint-temp-model-release"),
        )
        for label in (
            "after-checkpoint-save",
            "after-checkpoint-temp-model-release",
        ):
            assert parsed[label].available is False
            assert parsed[label].error


class TestReadLogSegment:
    """``read_log_segment`` reads new bytes since an offset; never raises."""

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_reads_appended_text_since_offset(self, tmp_path):
        log = tmp_path / "comfy.log"
        log.write_text("first\n")
        first_offset = harness.current_log_offset(str(log))
        with log.open("a") as f:
            f.write("[mem] after-checkpoint-save: RSS=900MB\n")
        text, end, err = harness.read_log_segment(str(log), first_offset)
        assert err == ""
        assert end > first_offset
        assert "after-checkpoint-save" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_unreadable_log_returns_error(self):
        text, end, err = harness.read_log_segment(
            "/nonexistent/path/should/not/exist.log", 0,
        )
        assert err
        assert text == ""
        assert end == 0


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
# Report formatting includes process-memory section
# ===========================================================================


class TestReportIncludesProcessMemorySection:
    """Report formatting renders the process-memory section when supplied."""

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_report_text_includes_supplied_pid(self):
        report = harness.CheckpointValidationReport(comfy_pid=4242)
        text = harness.format_report(report)
        assert "--comfy-pid" in text
        assert "4242" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_report_text_includes_supplied_log_path(self):
        report = harness.CheckpointValidationReport(
            comfy_log_path="/var/log/comfy.log",
        )
        text = harness.format_report(report)
        assert "/var/log/comfy.log" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_report_text_marks_unavailable_inputs(self):
        report = harness.CheckpointValidationReport()
        text = harness.format_report(report)
        assert "not supplied" in text

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_report_json_includes_new_fields(self):
        report = harness.CheckpointValidationReport(
            comfy_pid=42,
            comfy_log_path="/var/log/comfy.log",
            repeat_cache_miss_runs_requested=3,
            memory_accumulation_threshold_mb=512,
        )
        data = json.loads(report.to_json())
        assert data["comfy_pid"] == 42
        assert data["comfy_log_path"] == "/var/log/comfy.log"
        assert data["repeat_cache_miss_runs_requested"] == 3
        assert data["memory_accumulation_threshold_mb"] == 512

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_required_fields_include_new_memory_fields(self):
        required = harness._REQUIRED_REPORT_FIELDS
        assert "comfy_pid" in required
        assert "comfy_log_path" in required
        assert "repeat_cache_miss_runs_requested" in required
        assert "memory_accumulation_threshold_mb" in required
        assert "repeat_cache_miss_run_reports" in required
        assert "repeated_cache_miss_memory_result" in required

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_format_report_renders_per_run_observations(self):
        run = harness.RepeatedCacheMissRunReport(
            run_index=0,
            cache_miss_identity="repeat-000-abc",
            model_name="ecaj_checkpoint_validation_save_repeat-000-abc",
        )
        run.process_memory_before = harness.ProcessMemoryObservation(
            label="before-run", available=True, source="smaps_rollup",
            rss_kb=1000, swap_kb=10, anonymous_kb=900,
        )
        run.process_memory_after_save = harness.ProcessMemoryObservation(
            label="after-save", available=False,
            error="no --comfy-pid supplied",
        )
        run.process_memory_after_return = harness.ProcessMemoryObservation(
            label="after-return", available=True, source="status",
            rss_kb=1200, swap_kb=20, anonymous_kb=None,
        )
        report = harness.CheckpointValidationReport(
            repeat_cache_miss_run_reports=[run],
        )
        text = harness.format_report(report)
        assert "Repeated Cache-Miss Runs" in text
        assert "repeat-000-abc" in text
        assert "RSS=1000kB" in text
        assert "RSS=1200kB" in text
        assert "unavailable" in text


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
# AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
# Repeated cache-miss runs preserve per-run identities and accumulation result
# ===========================================================================


class TestRepeatCacheMissIdentities:
    """``make_repeat_cache_miss_identity`` produces distinct identities."""

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_distinct_identity_per_index(self):
        a_id, a_name = harness.make_repeat_cache_miss_identity(0, "tok1")
        b_id, b_name = harness.make_repeat_cache_miss_identity(1, "tok1")
        assert a_id != b_id
        assert a_name != b_name

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_distinct_identity_per_token_at_same_index(self):
        a_id, _ = harness.make_repeat_cache_miss_identity(0, "tokA")
        b_id, _ = harness.make_repeat_cache_miss_identity(0, "tokB")
        assert a_id != b_id

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_default_token_is_time_derived_and_unique(self):
        a_id, _ = harness.make_repeat_cache_miss_identity(0)
        b_id, _ = harness.make_repeat_cache_miss_identity(0)
        assert a_id != b_id

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_model_name_carries_identity(self):
        identity, model_name = harness.make_repeat_cache_miss_identity(2, "tok")
        assert identity in model_name
        assert model_name.startswith("ecaj_checkpoint_validation_save_")


class TestBuildRepeatCacheMissWorkflow:
    """Each repeat-cache-miss workflow embeds its own model_name."""

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_workflow_uses_supplied_model_name(self):
        wf = harness.build_repeat_cache_miss_workflow(
            "src.safetensors", "ecaj_repeat_007_xyz",
        )
        assert wf["3"]["inputs"]["model_name"] == "ecaj_repeat_007_xyz"
        assert wf["3"]["inputs"]["enable_cache"] is False
        assert wf["3"]["inputs"]["save_model"] is True

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_each_repeat_workflow_has_distinct_model_name(self):
        wf_a = harness.build_repeat_cache_miss_workflow("src.safetensors", "name_A")
        wf_b = harness.build_repeat_cache_miss_workflow("src.safetensors", "name_B")
        assert wf_a["3"]["inputs"]["model_name"] != wf_b["3"]["inputs"]["model_name"]


class TestRunRepeatedCacheMissRuns:
    """``run_repeated_cache_miss_runs`` produces per-run report entries."""

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_per_run_identities_are_distinct(self):
        submitted = []

        def mock_submit(base_url, workflow, name):
            submitted.append((name, workflow["3"]["inputs"]["model_name"]))
            return harness.WorkflowResult(
                name=name, accepted=True, prompt_id=f"p_{name}",
            )

        with patch.object(harness, "submit_workflow", side_effect=mock_submit):
            runs = harness.run_repeated_cache_miss_runs(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                repeat_count=3,
                comfy_pid=None,
                comfy_log_path=None,
            )

        assert len(runs) == 3
        identities = [r.cache_miss_identity for r in runs]
        assert len(set(identities)) == 3
        model_names = [r.model_name for r in runs]
        assert len(set(model_names)) == 3
        # Each submitted workflow's model_name must match its run report.
        submitted_names = [m for _name, m in submitted]
        assert submitted_names == model_names

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_each_run_records_save_result(self):
        with patch.object(
            harness, "submit_workflow",
            return_value=harness.WorkflowResult(name="r", accepted=True, prompt_id="x"),
        ):
            runs = harness.run_repeated_cache_miss_runs(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                repeat_count=2,
            )
        for run in runs:
            assert run.save_result.accepted is True

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_runs_record_unavailable_observation_when_pid_missing(self):
        with patch.object(
            harness, "submit_workflow",
            return_value=harness.WorkflowResult(name="r", accepted=True, prompt_id="x"),
        ):
            runs = harness.run_repeated_cache_miss_runs(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                repeat_count=1,
                comfy_pid=None,
                comfy_log_path=None,
            )
        run = runs[0]
        # All three lifecycle observations are unavailable since no pid.
        assert run.process_memory_before.available is False
        assert run.process_memory_after_save.available is False
        assert run.process_memory_after_return.available is False
        # All three carry a descriptive error rather than crashing.
        assert run.process_memory_before.error
        assert run.process_memory_after_save.error
        assert run.process_memory_after_return.error
        # Log labels are also marked unavailable.
        assert run.log_after_checkpoint_save.available is False
        assert run.log_after_checkpoint_temp_model_release.available is False

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_runs_record_log_observation_when_log_supplied(self, tmp_path):
        log_path = tmp_path / "comfy.log"
        log_path.write_text("")

        def mock_submit(base_url, workflow, name):
            # Simulate ComfyUI emitting WIDEN memory log lines during the run.
            with log_path.open("a") as f:
                f.write(
                    "2026-05-13 INFO [mem] after-checkpoint-save: "
                    "RSS=4096MB VRAM=2048MB(reserved=3072MB)\n",
                )
                f.write(
                    "2026-05-13 INFO [mem] "
                    "after-checkpoint-temp-model-release: RSS=2048MB\n",
                )
            return harness.WorkflowResult(name=name, accepted=True, prompt_id="x")

        with patch.object(harness, "submit_workflow", side_effect=mock_submit):
            runs = harness.run_repeated_cache_miss_runs(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                repeat_count=1,
                comfy_pid=None,
                comfy_log_path=str(log_path),
            )
        run = runs[0]
        assert run.log_after_checkpoint_save.available is True
        assert run.log_after_checkpoint_save.rss_mb == 4096
        assert run.log_after_checkpoint_temp_model_release.available is True
        assert run.log_after_checkpoint_temp_model_release.rss_mb == 2048

    # AC: @live-comfy-saved-output-validation ac-report-records-process-memory-points
    def test_runs_record_log_unavailable_when_log_path_missing_labels(
        self, tmp_path,
    ):
        log_path = tmp_path / "comfy.log"
        log_path.write_text("some unrelated log\n")
        with patch.object(
            harness, "submit_workflow",
            return_value=harness.WorkflowResult(name="r", accepted=True, prompt_id="x"),
        ):
            runs = harness.run_repeated_cache_miss_runs(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                repeat_count=1,
                comfy_pid=None,
                comfy_log_path=str(log_path),
            )
        run = runs[0]
        # Did not crash; labels are recorded as unavailable.
        assert run.log_after_checkpoint_save.available is False
        assert run.log_after_checkpoint_save.error


# ===========================================================================
# AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
# Accumulation analysis — failure when threshold exceeded
# ===========================================================================


class TestAnalyzeMemoryAccumulation:
    """``analyze_memory_accumulation`` flags growing RSS+Swap as failed."""

    @staticmethod
    def _run_with_post_return(index: int, rss_kb: int, swap_kb: int = 0):
        run = harness.RepeatedCacheMissRunReport(
            run_index=index,
            cache_miss_identity=f"repeat-{index:03d}",
            model_name=f"name_{index}",
        )
        run.process_memory_after_return = harness.ProcessMemoryObservation(
            label="after-return", available=True, source="smaps_rollup",
            rss_kb=rss_kb, swap_kb=swap_kb,
        )
        return run

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_failed_when_post_return_growth_exceeds_threshold(self):
        runs = [
            self._run_with_post_return(0, rss_kb=1_000_000),
            self._run_with_post_return(1, rss_kb=1_500_000),
            self._run_with_post_return(2, rss_kb=2_000_000),
        ]
        # 2_000_000 - 1_000_000 = 1_000_000 kB ~= 976 MB, above 256 MB.
        result = harness.analyze_memory_accumulation(runs, threshold_mb=256)
        assert result["result"] == "failed"
        assert result["threshold_mb"] == 256
        assert result["delta_kb"] == 1_000_000
        assert result["first_run_index"] == 0
        assert result["last_run_index"] == 2
        assert "accumulate" in result["detail"]

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_ok_when_growth_within_threshold(self):
        runs = [
            self._run_with_post_return(0, rss_kb=1_000_000),
            self._run_with_post_return(1, rss_kb=1_010_000),
        ]
        # 10_000 kB ~= 9.8 MB, well below 256 MB.
        result = harness.analyze_memory_accumulation(runs, threshold_mb=256)
        assert result["result"] == "ok"
        assert result["delta_kb"] == 10_000
        assert result["threshold_mb"] == 256

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_threshold_recorded_even_when_unavailable(self):
        runs = [
            harness.RepeatedCacheMissRunReport(run_index=0),
        ]
        result = harness.analyze_memory_accumulation(runs, threshold_mb=128)
        assert result["result"] == "unavailable"
        assert result["threshold_mb"] == 128
        assert result["delta_kb"] is None

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_swap_growth_counted_toward_threshold(self):
        runs = [
            self._run_with_post_return(0, rss_kb=1_000_000, swap_kb=0),
            self._run_with_post_return(
                1, rss_kb=1_000_000, swap_kb=600_000,
            ),  # +600 MB via swap
        ]
        result = harness.analyze_memory_accumulation(runs, threshold_mb=256)
        assert result["result"] == "failed"
        assert result["delta_kb"] == 600_000

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_unavailable_when_only_one_observation_available(self):
        runs = [
            self._run_with_post_return(0, rss_kb=1_000_000),
            harness.RepeatedCacheMissRunReport(run_index=1),
        ]
        result = harness.analyze_memory_accumulation(runs, threshold_mb=256)
        assert result["result"] == "unavailable"


# ===========================================================================
# AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
# AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
# run_validation integrates repeated cache-miss runs and accumulation result.
# ===========================================================================


class TestRunValidationRepeatCacheMiss:
    """``run_validation`` wires repeated cache-miss runs into the report."""

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_no_repeats_records_not_requested(self):
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }
        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", return_value=harness.WorkflowResult(
                 name="t", accepted=True, prompt_id="p1",
             )), \
             patch.object(harness, "check_cache_reuse", return_value=(True, "cached")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                report_output="/fake/r.json",
            )
        assert report.repeat_cache_miss_runs_requested == 0
        assert report.repeat_cache_miss_run_reports == []
        assert (
            report.repeated_cache_miss_memory_result.get("result")
            == "not-requested"
        )
        # Threshold default still recorded for auditability.
        assert (
            report.repeated_cache_miss_memory_result["threshold_mb"]
            == harness.DEFAULT_ACCUMULATION_THRESHOLD_MB
        )

    # AC: @live-comfy-saved-output-validation ac-report-records-repeated-cache-miss-memory
    def test_repeats_produce_distinct_artifact_identities(self):
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }
        submitted_names = []

        def mock_submit(base_url, workflow, name):
            submitted_names.append((name, workflow["3"]["inputs"].get("model_name", "")))
            return harness.WorkflowResult(name=name, accepted=True, prompt_id=f"p_{name}")

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", side_effect=mock_submit), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                report_output="/fake/r.json",
                repeat_cache_miss_runs=3,
            )

        assert report.repeat_cache_miss_runs_requested == 3
        assert len(report.repeat_cache_miss_run_reports) == 3
        identities = [r.cache_miss_identity for r in report.repeat_cache_miss_run_reports]
        assert len(set(identities)) == 3
        # Each repeat workflow's model_name is distinct.
        repeat_names = [
            m for n, m in submitted_names if n.startswith("repeat_cache_miss_")
        ]
        assert len(set(repeat_names)) == 3

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_accumulation_failure_marks_failure_category(self):
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }
        # Fake monotonic growth across two repeat runs.
        call_count = [0]

        def fake_collect(pid, label):
            # Per-run sequence: before, after-save, after-return × N runs.
            call_count[0] += 1
            # Only return-after observations affect the accumulation check.
            if label != "after-return":
                return harness.ProcessMemoryObservation(
                    label=label, available=True, source="smaps_rollup",
                    rss_kb=1_000_000, swap_kb=0, anonymous_kb=1_000_000,
                )
            # First "after-return" is small; second is way bigger.
            if call_count[0] <= 3:  # first run's three observations
                rss = 1_000_000
            else:
                rss = 5_000_000  # +4 GB
            return harness.ProcessMemoryObservation(
                label=label, available=True, source="smaps_rollup",
                rss_kb=rss, swap_kb=0, anonymous_kb=rss,
            )

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", return_value=harness.WorkflowResult(
                 name="r", accepted=True, prompt_id="p",
             )), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")), \
             patch.object(harness, "collect_process_memory_observation",
                          side_effect=fake_collect):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                report_output="/fake/r.json",
                comfy_pid=4242,
                repeat_cache_miss_runs=2,
                memory_accumulation_threshold_mb=256,
            )

        assert report.repeated_cache_miss_memory_result["result"] == "failed"
        # Threshold recorded.
        assert report.repeated_cache_miss_memory_result["threshold_mb"] == 256
        # failure_categories propagates the accumulation failure.
        assert "repeated_cache_miss_memory" in report.failure_categories
        assert (
            report.failure_categories["repeated_cache_miss_memory"]
            != "ok"
        )

    # AC: @comfy-memory-manager-compatibility ac-repeated-cache-miss-memory-bounded
    def test_accumulation_within_threshold_is_ok(self):
        mock_stats = {
            "system": {"comfyui_version": "0.3.4"},
            "devices": [{"vram_state": "normal"}],
        }
        # All observations roughly the same.
        flat_obs = harness.ProcessMemoryObservation(
            label="x", available=True, source="smaps_rollup",
            rss_kb=1_000_000, swap_kb=0, anonymous_kb=1_000_000,
        )

        def fake_collect(pid, label):
            obs = harness.ProcessMemoryObservation(
                label=label, available=True, source="smaps_rollup",
                rss_kb=1_000_000, swap_kb=0, anonymous_kb=1_000_000,
            )
            return obs

        with patch.object(harness, "query_system_stats", return_value=mock_stats), \
             patch.object(harness, "query_object_info", return_value={}), \
             patch.object(harness, "submit_workflow", return_value=harness.WorkflowResult(
                 name="r", accepted=True, prompt_id="p",
             )), \
             patch.object(harness, "check_cache_reuse", return_value=(False, "n/a")), \
             patch.object(harness, "collect_process_memory_observation",
                          side_effect=fake_collect):
            report = harness.run_validation(
                comfy_api_url="http://fake:8188",
                source_model="src.safetensors",
                report_output="/fake/r.json",
                comfy_pid=4242,
                repeat_cache_miss_runs=2,
                memory_accumulation_threshold_mb=256,
            )
        # _ = flat_obs  # ensure helper class is still constructible
        _ = flat_obs
        assert report.repeated_cache_miss_memory_result["result"] == "ok"
        assert (
            report.failure_categories["repeated_cache_miss_memory"] == "ok"
        )
