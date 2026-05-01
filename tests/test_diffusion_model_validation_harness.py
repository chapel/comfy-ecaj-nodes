"""Tests for the live ComfyUI diffusion-model saved-output validation harness.

These tests validate guard behavior, workflow shape correctness (UNETLoader
+ separately supplied CLIP/VAE), and report formatting.  They NEVER submit
real ComfyUI API prompts, import ComfyUI modules, mutate a ComfyUI
installation, start processes, or write large artifacts.

AC coverage for:
- @full-saved-model-output ac-diffusion-model-source-kind-round-trip
- @full-saved-model-output ac-diffusion-model-companion-separation
- @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in (mirrors
  the checkpoint harness opt-in contract for diffusion-model validation)
"""

from __future__ import annotations

import importlib.util
import sys

# ---------------------------------------------------------------------------
# Import harness module without triggering any ComfyUI work
# ---------------------------------------------------------------------------

_HARNESS_PATH = "scripts/manual/real_comfy_diffusion_model_validation"


def _import_harness():
    """Import the harness module by path (no ComfyUI dependency)."""
    mod_name = "real_comfy_diffusion_model_validation"
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
# Guard refusal tests
# ===========================================================================


class TestGuardRefusalNoEnvVar:
    """Guard refuses when COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION is missing."""

    def test_missing_env_var_refuses(self):
        result = harness.check_guards(
            env={},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any(
            "COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION" in r for r in result.reasons
        )

    def test_wrong_env_var_value_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "0"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed


class TestGuardRefusalMissingFlag:
    """Guard refuses when --run-diffusion-model-validation flag is missing."""

    def test_missing_flag_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any(
            "--run-diffusion-model-validation" in r for r in result.reasons
        )


class TestGuardRefusalMissingRequiredArgs:
    """Guard refuses when required CLI arguments are missing."""

    def test_missing_api_url_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--comfy-api-url" in r for r in result.reasons)

    def test_missing_source_diffusion_model_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--source-diffusion-model" in r for r in result.reasons)

    def test_missing_companion_clip_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--companion-clip" in r for r in result.reasons)

    def test_missing_companion_vae_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert not result.passed
        assert any("--companion-vae" in r for r in result.reasons)

    def test_missing_report_output_refuses(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
            ],
        )
        assert not result.passed
        assert any("--report-output" in r for r in result.reasons)


class TestGuardPasses:
    """All guards satisfied → check_guards returns passed=True."""

    def test_all_guards_pass(self):
        result = harness.check_guards(
            env={"COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION": "1"},
            argv=[
                "--run-diffusion-model-validation",
                "--comfy-api-url", "http://127.0.0.1:8188",
                "--source-diffusion-model", "model.safetensors",
                "--companion-clip", "clip.safetensors",
                "--companion-vae", "vae.safetensors",
                "--report-output", "/fake/report.json",
            ],
        )
        assert result.passed
        assert result.reasons == ()


# ===========================================================================
# Workflow shape tests
# AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
# AC: @full-saved-model-output ac-diffusion-model-companion-separation
# ===========================================================================


class TestDiffusionSaveWorkflowShape:
    """build_diffusion_save_workflow produces a UNETLoader → WIDENEntry →
    WIDENExit graph with no checkpoint loader anywhere."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_uses_unet_loader_not_checkpoint_loader(self):
        wf = harness.build_diffusion_save_workflow("flux1-schnell.safetensors")
        class_types = {node["class_type"] for node in wf.values()}
        assert "UNETLoader" in class_types
        # No CheckpointLoaderSimple — this is the diffusion-model contract.
        assert "CheckpointLoaderSimple" not in class_types

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_unet_loader_loads_source_diffusion_model(self):
        wf = harness.build_diffusion_save_workflow("flux1-schnell.safetensors")
        unet_nodes = [n for n in wf.values() if n["class_type"] == "UNETLoader"]
        assert len(unet_nodes) == 1
        assert unet_nodes[0]["inputs"]["unet_name"] == "flux1-schnell.safetensors"

    # AC: @full-saved-model-output ac-diffusion-model-companion-separation
    def test_widen_entry_does_not_supply_clip_or_vae(self):
        """A diffusion-model recipe must not bundle CLIP/VAE through WIDEN
        Entry.  Standalone diffusion-model saves are diffusion-only by
        contract — companion components stay outside the saved artifact."""
        wf = harness.build_diffusion_save_workflow("model.safetensors")
        entry_nodes = [n for n in wf.values() if n["class_type"] == "WIDENEntry"]
        assert len(entry_nodes) == 1
        inputs = entry_nodes[0]["inputs"]
        assert "model" in inputs
        assert "clip" not in inputs
        assert "vae" not in inputs

    def test_save_workflow_disables_cache_for_fresh_artifact(self):
        wf = harness.build_diffusion_save_workflow("model.safetensors")
        exit_nodes = [n for n in wf.values() if n["class_type"] == "WIDENExit"]
        assert len(exit_nodes) == 1
        assert exit_nodes[0]["inputs"]["save_model"] is True
        assert exit_nodes[0]["inputs"]["enable_cache"] is False


class TestDiffusionDownstreamWorkflowShape:
    """build_diffusion_downstream_workflow loads the saved diffusion-model
    artifact via UNETLoader and supplies CLIP / VAE separately."""

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    def test_downstream_loads_via_unet_loader(self):
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip.safetensors",
            companion_vae="vae.safetensors",
        )
        # The MODEL output for KSampler must come from a UNETLoader, not
        # a CheckpointLoaderSimple — this is the round-trip contract.
        ksampler = next(
            n for n in wf.values() if n["class_type"] == "KSampler"
        )
        model_ref = ksampler["inputs"]["model"]
        producer = wf[model_ref[0]]
        assert producer["class_type"] == "UNETLoader"
        assert producer["inputs"]["unet_name"] == "saved.safetensors"

    # AC: @full-saved-model-output ac-diffusion-model-companion-separation
    def test_companion_clip_supplied_separately_from_unet_artifact(self):
        """The CLIP feeding CLIPTextEncode must come from a separate CLIP
        loader (CLIPLoader / DualCLIPLoader), not from the UNETLoader.
        UNETLoader does not produce a CLIP output, but the contract is
        explicit: companion components are supplied by the surrounding
        workflow."""
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip.safetensors",
            companion_vae="vae.safetensors",
        )
        text_encoders = [
            n for n in wf.values() if n["class_type"] == "CLIPTextEncode"
        ]
        assert text_encoders, "downstream workflow must have CLIPTextEncode"
        for te in text_encoders:
            clip_ref = te["inputs"]["clip"]
            producer = wf[clip_ref[0]]
            assert producer["class_type"] in {"CLIPLoader", "DualCLIPLoader"}, (
                f"CLIP for text-encode came from {producer['class_type']!r}, "
                f"expected CLIPLoader or DualCLIPLoader (separate companion)"
            )

    # AC: @full-saved-model-output ac-diffusion-model-companion-separation
    def test_companion_vae_supplied_separately_from_unet_artifact(self):
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip.safetensors",
            companion_vae="vae.safetensors",
        )
        vae_decode = next(
            n for n in wf.values() if n["class_type"] == "VAEDecode"
        )
        vae_ref = vae_decode["inputs"]["vae"]
        producer = wf[vae_ref[0]]
        assert producer["class_type"] == "VAELoader"
        assert producer["inputs"]["vae_name"] == "vae.safetensors"

    # AC: @full-saved-model-output ac-diffusion-model-companion-separation
    def test_no_checkpoint_loader_in_downstream_workflow(self):
        """A diffusion-model round-trip must not silently fall back to
        CheckpointLoaderSimple — that would be a checkpoint-style load and
        would not validate the diffusion-model contract."""
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip.safetensors",
            companion_vae="vae.safetensors",
        )
        class_types = {n["class_type"] for n in wf.values()}
        assert "CheckpointLoaderSimple" not in class_types

    def test_dual_clip_loader_used_when_clip2_provided(self):
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip_l.safetensors",
            companion_clip2="t5xxl_fp16.safetensors",
            companion_clip_type="flux",
            companion_vae="ae.safetensors",
        )
        dual = [n for n in wf.values() if n["class_type"] == "DualCLIPLoader"]
        assert len(dual) == 1
        inputs = dual[0]["inputs"]
        assert inputs["clip_name1"] == "clip_l.safetensors"
        assert inputs["clip_name2"] == "t5xxl_fp16.safetensors"
        assert inputs["type"] == "flux"

    def test_single_clip_loader_used_when_clip2_absent(self):
        wf = harness.build_diffusion_downstream_workflow(
            "saved.safetensors",
            companion_clip="clip.safetensors",
            companion_vae="vae.safetensors",
        )
        single = [n for n in wf.values() if n["class_type"] == "CLIPLoader"]
        assert len(single) == 1
        assert "DualCLIPLoader" not in {
            n["class_type"] for n in wf.values()
        }


class TestDiffusionCacheReuseWorkflowShape:
    """build_diffusion_cache_reuse_workflow re-runs the save with caching."""

    def test_enable_cache_true_for_reuse_run(self):
        wf = harness.build_diffusion_cache_reuse_workflow("model.safetensors")
        exit_nodes = [n for n in wf.values() if n["class_type"] == "WIDENExit"]
        assert len(exit_nodes) == 1
        assert exit_nodes[0]["inputs"]["enable_cache"] is True

    def test_uses_unet_loader_consistent_with_save_run(self):
        wf = harness.build_diffusion_cache_reuse_workflow("model.safetensors")
        class_types = {n["class_type"] for n in wf.values()}
        assert "UNETLoader" in class_types
        assert "CheckpointLoaderSimple" not in class_types


# ===========================================================================
# Failure classification tests
# ===========================================================================


class TestFailureClassification:
    """classify_failure maps WorkflowResult errors into categories that
    surface diffusion-loader-specific failures."""

    def test_unet_loader_error_classified_as_loader(self):
        result = harness.WorkflowResult(
            name="x", accepted=False,
            error="node_error(UNETLoader): file not found",
        )
        assert harness.classify_failure(result) == "loader"

    def test_diffusion_keyword_classified_as_loader(self):
        result = harness.WorkflowResult(
            name="x", accepted=False,
            error=(
                "node_error(SomeNode): could not load diffusion model "
                "from artifact"
            ),
        )
        assert harness.classify_failure(result) == "loader"

    def test_oom_takes_priority_over_loader(self):
        result = harness.WorkflowResult(
            name="x", accepted=False,
            error="node_error(UNETLoader): OOM: cannot allocate vram",
        )
        assert harness.classify_failure(result) == "memory_mode"

    def test_accepted_is_none(self):
        result = harness.WorkflowResult(
            name="x", accepted=True, error="",
        )
        assert harness.classify_failure(result) == "none"


# ===========================================================================
# Report formatting tests
# ===========================================================================


class TestReportFields:
    """DiffusionModelValidationReport contains the required fields."""

    def test_default_report_has_required_fields(self):
        rep = harness.DiffusionModelValidationReport()
        ok, missing = harness.report_has_required_fields(rep.to_dict())
        assert ok, f"missing fields: {missing}"

    def test_to_json_round_trips(self):
        import json as _json
        rep = harness.DiffusionModelValidationReport(
            comfy_api_url="http://localhost:8188",
            source_diffusion_model="m.safetensors",
        )
        data = _json.loads(rep.to_json())
        assert data["comfy_api_url"] == "http://localhost:8188"
        assert data["source_diffusion_model"] == "m.safetensors"

    def test_format_report_renders_diffusion_specific_sections(self):
        rep = harness.DiffusionModelValidationReport(
            comfy_api_url="http://localhost:8188",
            source_diffusion_model="m.safetensors",
            companion_clip="c.safetensors",
            companion_vae="v.safetensors",
        )
        text = harness.format_report(rep)
        assert "Diffusion-Model Save" in text
        assert "Diffusion-Model Loader (UNETLoader)" in text
        assert "companion CLIP + VAE supplied separately" in text
        assert "c.safetensors" in text
        assert "v.safetensors" in text


class TestRefusalFormatting:
    """format_refusal lists missing inputs and confirms no work was done."""

    def test_refusal_lists_required_inputs(self):
        guard = harness.GuardResult(
            passed=False,
            reasons=("a", "b"),
        )
        text = harness.format_refusal(guard)
        assert "REFUSED" in text
        assert "COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION" in text
        assert "--run-diffusion-model-validation" in text
        assert "--source-diffusion-model" in text
        assert "--companion-clip" in text
        assert "--companion-vae" in text

    def test_refusal_confirms_no_work_done(self):
        guard = harness.GuardResult(
            passed=False, reasons=("missing X",),
        )
        text = harness.format_refusal(guard)
        assert "No ComfyUI API prompts were submitted" in text
        assert "No ComfyUI modules were imported" in text
        assert "No large artifacts were written" in text
