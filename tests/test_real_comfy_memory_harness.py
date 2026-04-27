"""Tests for the real ComfyUI memory-mode validation harness.

These tests validate guard behavior, report formatting, and test-discovery
exclusion. They NEVER import ComfyUI, mutate sys.path for a ComfyUI install,
or run real ComfyUI validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys  # noqa: I001

# ---------------------------------------------------------------------------
# Import harness module without triggering any ComfyUI work
# ---------------------------------------------------------------------------

# The harness is a standalone script outside the lib/nodes packages.
# Import it by path so tests can exercise the guard and report layers.

_HARNESS_PATH = "scripts/manual/real_comfy_memory_validation"


def _import_harness():
    """Import the harness module by path (no ComfyUI dependency)."""
    import importlib.util

    mod_name = "real_comfy_memory_validation"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        f"{_HARNESS_PATH}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec so dataclasses can resolve
    spec.loader.exec_module(mod)
    return mod


harness = _import_harness()


# ===========================================================================
# AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
# Guard refusal tests — harness refuses before importing ComfyUI
# ===========================================================================


class TestGuardRefusalNoEnvVar:
    """Guard refuses when COMFY_ECAJ_REAL_MEMORY_VALIDATION is missing."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_missing_env_var_refuses(self):
        result = harness.check_guards(
            env={},
            argv=[
                "--run-real-comfy-memory",
                "--comfy-root", "/fake/comfy",
                "--model-path", "/fake/model.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("COMFY_ECAJ_REAL_MEMORY_VALIDATION" in r for r in result.reasons)

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_wrong_env_var_value_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "0"},
            argv=[
                "--run-real-comfy-memory",
                "--comfy-root", "/fake/comfy",
                "--model-path", "/fake/model.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("COMFY_ECAJ_REAL_MEMORY_VALIDATION" in r for r in result.reasons)


class TestGuardRefusalNoFlag:
    """Guard refuses when --run-real-comfy-memory flag is missing."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_missing_flag_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=[
                "--comfy-root", "/fake/comfy",
                "--model-path", "/fake/model.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--run-real-comfy-memory" in r for r in result.reasons)


class TestGuardRefusalMissingPaths:
    """Guard refuses when required path arguments are missing."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_missing_comfy_root_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=[
                "--run-real-comfy-memory",
                "--model-path", "/fake/model.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--comfy-root" in r for r in result.reasons)

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_missing_model_path_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=[
                "--run-real-comfy-memory",
                "--comfy-root", "/fake/comfy",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--model-path" in r for r in result.reasons)

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_missing_report_output_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=[
                "--run-real-comfy-memory",
                "--comfy-root", "/fake/comfy",
                "--model-path", "/fake/model.safetensors",
            ],
        )
        assert not result.passed
        assert any("--report-output" in r for r in result.reasons)


class TestGuardRefusalMultipleMissing:
    """Guard reports ALL missing inputs, not just the first."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_no_inputs_reports_all_reasons(self):
        result = harness.check_guards(env={}, argv=[])
        assert not result.passed
        # Should have reasons for: env var, flag, comfy-root, model-path, report-output
        assert len(result.reasons) == 5

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_partial_inputs_still_refuses(self):
        """Having some but not all inputs still refuses."""
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=["--run-real-comfy-memory", "--comfy-root", "/fake/comfy"],
        )
        assert not result.passed
        assert len(result.reasons) == 2  # missing model-path and report-output


class TestGuardDoesNotImportComfy:
    """Guard failure must not import comfy or mutate sys.path."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_guard_failure_does_not_import_comfy(self):
        """check_guards() never imports comfy regardless of result."""
        # Remove comfy from sys.modules if present (from conftest mocks)
        original_modules = set(sys.modules.keys())

        # Run guards with no inputs
        result = harness.check_guards(env={}, argv=[])
        assert not result.passed

        # The harness module itself should not have imported any comfy modules
        # beyond what was already present
        new_modules = set(sys.modules.keys()) - original_modules
        comfy_modules = {m for m in new_modules if m.startswith("comfy")}
        assert not comfy_modules, f"Guard failure imported comfy modules: {comfy_modules}"

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_main_returns_1_on_guard_failure(self):
        """main() returns 1 and does not proceed past guards."""
        exit_code = harness.main(argv=[])
        assert exit_code == 1


class TestGuardAllPass:
    """All guards pass when every input is provided."""

    def test_all_guards_pass(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_REAL_MEMORY_VALIDATION": "1"},
            argv=[
                "--run-real-comfy-memory",
                "--comfy-root", "/fake/comfy",
                "--model-path", "/fake/model.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert result.passed
        assert result.reasons == ()


# ===========================================================================
# AC: @manual-comfy-memory-validation ac-report-identifies-memory-mode
# Report formatting tests — pure Python, fake data
# ===========================================================================


class TestReportFormattingWithFakeData:
    """Report formatting works with fake data and identifies memory mode."""

    # AC: @manual-comfy-memory-validation ac-report-identifies-memory-mode
    def test_report_includes_memory_mode(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            comfy_root="/fake/comfy",
            model_path="/fake/model.safetensors",
        )
        text = harness.format_report(report)
        assert "Memory Mode:" in text
        assert "normal" in text

    # AC: @manual-comfy-memory-validation ac-report-identifies-memory-mode
    def test_report_includes_lowvram_mode(self):
        report = harness.ValidationReport(memory_mode="lowvram")
        text = harness.format_report(report)
        assert "lowvram" in text

    # AC: @manual-comfy-memory-validation ac-report-identifies-memory-mode
    def test_report_includes_highvram_mode(self):
        report = harness.ValidationReport(memory_mode="highvram")
        text = harness.format_report(report)
        assert "highvram" in text

    def test_report_includes_patch_mode_behavior(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            patch_mode_behavior="install_merged_patches available",
        )
        text = harness.format_report(report)
        assert "Patch-mode behavior:" in text
        assert "install_merged_patches available" in text

    def test_report_includes_full_model_materialization(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            full_model_materialization="MaterializationSink available",
        )
        text = harness.format_report(report)
        assert "Full-model materialization:" in text

    def test_report_includes_memory_observations(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            memory_observations=[
                harness.MemoryObservation(
                    label="before-validation",
                    rss_mb=1024.0,
                    vram_allocated_mb=512.0,
                    vram_reserved_mb=768.0,
                ),
                harness.MemoryObservation(
                    label="after-validation",
                    rss_mb=1280.0,
                    vram_allocated_mb=640.0,
                    vram_reserved_mb=896.0,
                ),
            ],
        )
        text = harness.format_report(report)
        assert "before-validation" in text
        assert "after-validation" in text
        assert "RSS=1024MB" in text
        assert "VRAM=512MB" in text

    def test_report_includes_errors(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            errors=["Something went wrong"],
        )
        text = harness.format_report(report)
        assert "Something went wrong" in text

    def test_report_json_roundtrip(self):
        report = harness.ValidationReport(
            memory_mode="normal",
            comfy_root="/fake/comfy",
            model_path="/fake/model.safetensors",
            patch_mode_behavior="ok",
            full_model_materialization="ok",
            artifact_cache_hit=True,
            returned_model_behavior="ok",
            artifact_reuse="ok",
            weights_resident_in_cache=False,
            duration_seconds=1.5,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert data["memory_mode"] == "normal"
        assert data["comfy_root"] == "/fake/comfy"
        assert data["artifact_cache_hit"] is True
        assert data["weights_resident_in_cache"] is False
        assert data["duration_seconds"] == 1.5


class TestRefusalReport:
    """Refusal report includes clear reasons."""

    def test_refusal_report_includes_reasons(self):
        guard = harness.GuardResult(
            passed=False,
            reasons=("Missing env var.", "Missing flag."),
        )
        text = harness.format_refusal(guard)
        assert "REFUSED" in text
        assert "Missing env var." in text
        assert "Missing flag." in text
        assert "No ComfyUI modules were imported" in text
        assert "No ComfyUI installation was modified" in text
        assert "No ComfyUI process was started" in text


# ===========================================================================
# AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
# Default test discovery does not run real ComfyUI validation
# ===========================================================================


class TestDefaultDiscoveryExclusion:
    """Default project test commands do not perform real ComfyUI validation."""

    # AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
    def test_harness_not_in_test_paths(self):
        """The harness script lives outside the pytest testpaths (tests/)."""
        # scripts/manual/ is not under tests/, so pytest --collect-only
        # will never discover it as a test module.
        assert not _HARNESS_PATH.startswith("tests/")

    # AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
    def test_harness_has_no_test_functions(self):
        """The harness module contains no test_* functions or Test* classes."""
        members = dir(harness)
        test_items = [m for m in members if m.startswith("test_") or m.startswith("Test")]
        assert test_items == [], f"Harness contains test-like names: {test_items}"

    # AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
    def test_pytest_collect_does_not_find_harness(self):
        """Running pytest --collect-only does not collect the harness."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
        )
        assert "real_comfy_memory_validation" not in result.stdout

    # AC: @manual-comfy-memory-validation ac-explicit-opt-in-required
    def test_main_without_guards_does_not_run_validation(self):
        """Invoking main() without proper guards returns 1 (refusal)."""
        exit_code = harness.main(argv=[])
        assert exit_code == 1


# ===========================================================================
# AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
# run_validation rejects nonexistent model_path before doing ComfyUI work
# ===========================================================================


class TestRunValidationModelPathValidation:
    """run_validation refuses to proceed with a nonexistent model_path."""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_nonexistent_model_path_returns_error(self):
        """run_validation reports an error when model_path does not exist."""
        report = harness.run_validation(
            comfy_root="/fake/comfy",
            model_path="/nonexistent/model.safetensors",
            report_output="/fake/report.json",
        )
        assert len(report.errors) > 0
        assert any("does not exist" in e for e in report.errors)
        # Should not have proceeded to populate behavior fields
        assert report.memory_mode == "unknown"
        assert report.patch_mode_behavior == ""

    # AC: @manual-comfy-memory-validation ac-refuses-before-comfy-work-without-opt-in
    def test_nonexistent_model_path_does_not_import_comfy(self):
        """run_validation with bad model_path does not import comfy."""
        original_modules = set(sys.modules.keys())
        harness.run_validation(
            comfy_root="/fake/comfy",
            model_path="/nonexistent/model.safetensors",
            report_output="/fake/report.json",
        )
        new_modules = set(sys.modules.keys()) - original_modules
        comfy_modules = {m for m in new_modules if m.startswith("comfy")}
        assert not comfy_modules, (
            f"Nonexistent model_path imported comfy modules: {comfy_modules}"
        )


# ===========================================================================
# Import path ordering — project root before ComfyUI root
# ===========================================================================


class TestSetupImportPaths:
    """_setup_import_paths places project root before ComfyUI root."""

    def test_project_root_before_comfy_root(self):
        """After _setup_import_paths, project root precedes comfy root."""
        saved_path = sys.path[:]
        try:
            comfy_root = "/tmp/fake-comfy-root-for-test"
            harness._setup_import_paths(comfy_root)
            project_root = harness._resolve_project_root()
            proj_idx = sys.path.index(project_root)
            comfy_idx = sys.path.index(comfy_root)
            assert proj_idx < comfy_idx, (
                f"Project root at index {proj_idx} should precede "
                f"ComfyUI root at index {comfy_idx}"
            )
        finally:
            sys.path[:] = saved_path

    def test_setup_import_paths_is_idempotent(self):
        """Calling _setup_import_paths twice does not duplicate entries."""
        saved_path = sys.path[:]
        try:
            comfy_root = "/tmp/fake-comfy-root-for-test"
            harness._setup_import_paths(comfy_root)
            harness._setup_import_paths(comfy_root)
            project_root = harness._resolve_project_root()
            assert sys.path.count(project_root) == 1
            assert sys.path.count(comfy_root) == 1
        finally:
            sys.path[:] = saved_path

    def test_resolve_project_root_is_ancestor_of_scripts(self):
        """_resolve_project_root returns the directory containing scripts/."""
        project_root = harness._resolve_project_root()
        assert os.path.isdir(os.path.join(project_root, "scripts", "manual"))
