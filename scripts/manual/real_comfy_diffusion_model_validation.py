#!/usr/bin/env python3
"""Guarded live validation harness for diffusion-model saved-output workflows.

Companion to ``real_comfy_checkpoint_validation.py``.  Where that harness
validates checkpoint-style saves (CheckpointLoaderSimple round-trip), this
one validates **standalone diffusion-model** saves: a workflow that starts
from a Comfy diffusion-model file (loaded via UNETLoader / load_diffusion_model)
and confirms the WIDEN-saved artifact loads back through the same standalone
diffusion-model contract — with companion CLIP and VAE supplied separately
by the surrounding workflow, not bundled into the saved artifact.

This script is NOT part of the automated test suite.  It requires explicit
opt-in via environment variable, CLI flag, and all required paths before
any ComfyUI work is performed.

Usage::

    COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION=1 \\
        python scripts/manual/real_comfy_diffusion_model_validation.py \\
        --run-diffusion-model-validation \\
        --comfy-api-url http://127.0.0.1:8188 \\
        --source-diffusion-model flux1-schnell.safetensors \\
        --companion-clip clip_l.safetensors \\
        --companion-clip2 t5xxl_fp16.safetensors \\
        --companion-clip-type flux \\
        --companion-vae ae.safetensors \\
        --report-output /path/to/diffusion_report.json

All required inputs:
    1. COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION=1 environment variable
    2. --run-diffusion-model-validation CLI flag
    3. --comfy-api-url: URL of a running ComfyUI API server
    4. --source-diffusion-model: standalone diffusion-model file (e.g.
       flux1-schnell.safetensors) discoverable in ``diffusion_models``
    5. --companion-clip: CLIP filename (text encoder #1) for downstream
    6. --companion-vae: VAE filename for downstream
    7. --report-output: path where the JSON report will be written

Optional:
    --companion-clip2: second CLIP filename (e.g. T5 for Flux/SD3)
    --companion-clip-type: CLIP type identifier passed to DualCLIPLoader / CLIPLoader
        (default: "stable_diffusion"); set to "flux", "sdxl", "sd3", etc.
    --width, --height, --steps, --cfg, --seed, --sampler-name, --scheduler,
    --batch-size: standard KSampler defaults (small to keep the run bounded).

Saved artifact name:
    The harness saves under the fixed name
    ``ecaj_diffusion_model_validation_save.safetensors`` in Comfy's
    ``diffusion_models`` folder, which is the layout WIDEN Exit publishes
    to for diffusion-model recipes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Guard validation — must run BEFORE any ComfyUI import, API call, or mutation
# ---------------------------------------------------------------------------

_ENV_VAR = "COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION"
_CLI_FLAG = "--run-diffusion-model-validation"

_SAVED_ARTIFACT_NAME = "ecaj_diffusion_model_validation_save.safetensors"


@dataclass(frozen=True)
class GuardResult:
    """Result of guard validation. If passed is False, reasons lists why."""

    passed: bool
    reasons: tuple[str, ...] = ()


def check_guards(
    env: dict[str, str] | None = None,
    argv: list[str] | None = None,
) -> GuardResult:
    """Validate all explicit opt-in guards before any ComfyUI work.

    Checks (in order):
        1. COMFY_ECAJ_DIFFUSION_MODEL_VALIDATION=1 environment variable
        2. --run-diffusion-model-validation CLI flag
        3. --comfy-api-url argument present and non-empty
        4. --source-diffusion-model argument present and non-empty
        5. --companion-clip argument present and non-empty
        6. --companion-vae argument present and non-empty
        7. --report-output argument present and non-empty
    """
    if env is None:
        env = dict(os.environ)
    if argv is None:
        argv = sys.argv[1:]

    reasons: list[str] = []

    val = env.get(_ENV_VAR, "")
    if val != "1":
        reasons.append(f"Environment variable {_ENV_VAR} is not set to '1' (got {val!r}).")

    if _CLI_FLAG not in argv:
        reasons.append(f"CLI flag {_CLI_FLAG} is not present.")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfy-api-url", default=None)
    parser.add_argument("--source-diffusion-model", default=None)
    parser.add_argument("--companion-clip", default=None)
    parser.add_argument("--companion-vae", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument(_CLI_FLAG, action="store_true", dest="run_flag")
    known, _ = parser.parse_known_args(argv)

    if not known.comfy_api_url:
        reasons.append("--comfy-api-url is required but not provided.")
    if not known.source_diffusion_model:
        reasons.append("--source-diffusion-model is required but not provided.")
    if not known.companion_clip:
        reasons.append("--companion-clip is required but not provided.")
    if not known.companion_vae:
        reasons.append("--companion-vae is required but not provided.")
    if not known.report_output:
        reasons.append("--report-output is required but not provided.")

    if reasons:
        return GuardResult(passed=False, reasons=tuple(reasons))
    return GuardResult(passed=True)


def parse_validated_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments after guards have passed."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=("Live ComfyUI diffusion-model saved-output validation harness."),
    )
    parser.add_argument(
        _CLI_FLAG,
        action="store_true",
        dest="run_flag",
        help="Explicit opt-in flag to run diffusion-model validation.",
    )
    parser.add_argument(
        "--comfy-api-url",
        required=True,
        help="URL of a running ComfyUI API server (e.g., http://127.0.0.1:8188).",
    )
    parser.add_argument(
        "--source-diffusion-model",
        required=True,
        help=(
            "Standalone diffusion-model filename (e.g. flux1-schnell.safetensors) "
            "discoverable in ComfyUI's diffusion_models folder."
        ),
    )
    parser.add_argument(
        "--companion-clip",
        required=True,
        help=(
            "CLIP/text-encoder filename used by the downstream workflow.  "
            "WIDEN's saved diffusion-model artifact contains no CLIP keys, "
            "so a real CLIP must be supplied separately."
        ),
    )
    parser.add_argument(
        "--companion-clip2",
        default="",
        help=(
            "Optional second CLIP filename (e.g. T5xxl for Flux / SD3 dual "
            "CLIP).  When provided, the downstream workflow uses DualCLIPLoader; "
            "otherwise a single CLIPLoader is used."
        ),
    )
    parser.add_argument(
        "--companion-clip-type",
        default="stable_diffusion",
        help=(
            "CLIP type identifier passed to (Dual)CLIPLoader.  Examples: "
            "stable_diffusion, sdxl, flux, sd3, hunyuan_video.  Defaults to "
            "stable_diffusion."
        ),
    )
    parser.add_argument(
        "--companion-vae",
        required=True,
        help=(
            "VAE filename used by the downstream workflow.  WIDEN's saved "
            "diffusion-model artifact contains no VAE keys, so a real VAE "
            "must be supplied separately."
        ),
    )
    parser.add_argument(
        "--report-output",
        required=True,
        help="Path where the JSON validation report will be written.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-name", default="euler")
    parser.add_argument("--scheduler", default="normal")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Report builder — pure Python, testable with fake data
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Result of a single workflow submission to the ComfyUI API."""

    name: str = ""
    accepted: bool = False
    prompt_id: str = ""
    error: str = ""
    outputs: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class DiffusionModelValidationReport:
    """Report from a live ComfyUI diffusion-model saved-output validation run."""

    comfy_version: str = ""
    memory_mode: str = ""
    comfy_api_url: str = ""
    source_diffusion_model: str = ""
    companion_clip: str = ""
    companion_clip2: str = ""
    companion_clip_type: str = ""
    companion_vae: str = ""
    node_classes: list[str] = field(default_factory=list)
    diffusion_save_workflow_shape: dict = field(default_factory=dict)
    diffusion_save_result: WorkflowResult = field(default_factory=WorkflowResult)
    saved_artifact_path: str = ""
    saved_artifact_classification: str = ""
    diffusion_loader_result: WorkflowResult = field(default_factory=WorkflowResult)
    downstream_result: WorkflowResult = field(default_factory=WorkflowResult)
    cache_reuse_result: WorkflowResult = field(default_factory=WorkflowResult)
    cache_reuse_detail: str = ""
    failure_categories: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


_REQUIRED_REPORT_FIELDS = frozenset(
    {
        "comfy_version",
        "memory_mode",
        "diffusion_save_workflow_shape",
        "diffusion_save_result",
        "saved_artifact_path",
        "saved_artifact_classification",
        "diffusion_loader_result",
        "downstream_result",
        "cache_reuse_result",
        "cache_reuse_detail",
        "failure_categories",
    }
)


def report_has_required_fields(report_dict: dict) -> tuple[bool, list[str]]:
    """Check whether a report dict contains all required fields."""
    missing = [f for f in _REQUIRED_REPORT_FIELDS if f not in report_dict]
    return (len(missing) == 0, missing)


def format_report(report: DiffusionModelValidationReport) -> str:
    """Format a DiffusionModelValidationReport as human-readable text."""
    lines = [
        "=" * 60,
        "ComfyUI Diffusion-Model Saved-Output Validation Report",
        "=" * 60,
        "",
        f"ComfyUI Version:            {report.comfy_version}",
        f"Memory Mode:                {report.memory_mode}",
        f"API URL:                    {report.comfy_api_url}",
        f"Source Diffusion Model:     {report.source_diffusion_model}",
        f"Companion CLIP:             {report.companion_clip}",
        f"Companion CLIP2:            {report.companion_clip2 or '(none)'}",
        f"Companion CLIP type:        {report.companion_clip_type}",
        f"Companion VAE:              {report.companion_vae}",
        f"Duration:                   {report.duration_seconds:.2f}s",
        "",
    ]

    if report.node_classes:
        lines.append(f"Node Classes Found:         {len(report.node_classes)}")
        for nc in report.node_classes[:10]:
            lines.append(f"  - {nc}")
        if len(report.node_classes) > 10:
            lines.append(f"  ... and {len(report.node_classes) - 10} more")
        lines.append("")

    lines.extend(
        [
            "--- Diffusion-Model Save ---",
            f"  Accepted:     {report.diffusion_save_result.accepted}",
            f"  Prompt ID:    {report.diffusion_save_result.prompt_id}",
            f"  Artifact:     {report.saved_artifact_path}",
            f"  Classification: {report.saved_artifact_classification}",
            f"  Error:        {report.diffusion_save_result.error or 'none'}",
        ]
    )
    if report.diffusion_save_workflow_shape:
        shape_json = json.dumps(report.diffusion_save_workflow_shape)
        lines.append(f"  Workflow Shape: {shape_json}")
    lines.append("")

    lines.extend(
        [
            "--- Diffusion-Model Loader (UNETLoader) ---",
            f"  Accepted:     {report.diffusion_loader_result.accepted}",
            f"  Prompt ID:    {report.diffusion_loader_result.prompt_id}",
            f"  Error:        {report.diffusion_loader_result.error or 'none'}",
            "",
            "--- Downstream KSampler (companion CLIP + VAE supplied separately) ---",
            f"  Accepted:     {report.downstream_result.accepted}",
            f"  Prompt ID:    {report.downstream_result.prompt_id}",
            f"  Error:        {report.downstream_result.error or 'none'}",
            "",
            "--- Cache Reuse ---",
            f"  Accepted:     {report.cache_reuse_result.accepted}",
            f"  Prompt ID:    {report.cache_reuse_result.prompt_id}",
            f"  Detail:       {report.cache_reuse_detail or 'none'}",
            f"  Error:        {report.cache_reuse_result.error or 'none'}",
            "",
        ]
    )

    if report.failure_categories:
        lines.append("--- Failure Categories ---")
        for cat, desc in report.failure_categories.items():
            lines.append(f"  {cat}: {desc}")
        lines.append("")

    if report.errors:
        lines.append("--- Errors ---")
        for err in report.errors:
            lines.append(f"  ! {err}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_refusal(guard_result: GuardResult) -> str:
    """Format a guard refusal as human-readable text with clear reasons."""
    lines = [
        "=" * 60,
        "ComfyUI Diffusion-Model Validation — REFUSED (guards not satisfied)",
        "=" * 60,
        "",
        "The following required inputs are missing or invalid:",
        "",
    ]
    for reason in guard_result.reasons:
        lines.append(f"  - {reason}")
    lines.extend(
        [
            "",
            "All of the following are required to run diffusion-model validation:",
            f"  1. {_ENV_VAR}=1 environment variable",
            f"  2. {_CLI_FLAG} CLI flag",
            "  3. --comfy-api-url <url>",
            "  4. --source-diffusion-model <name>",
            "  5. --companion-clip <name>",
            "  6. --companion-vae <name>",
            "  7. --report-output <path>",
            "",
            "No ComfyUI API prompts were submitted.",
            "No ComfyUI modules were imported.",
            "No ComfyUI installation was modified.",
            "No processes were started.",
            "No large artifacts were written.",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComfyUI API client — only reachable after all guards pass
# ---------------------------------------------------------------------------


def _api_get(base_url: str, path: str, timeout: float = 30.0) -> dict:
    """GET a JSON endpoint from the ComfyUI API."""
    url = f"{base_url.rstrip('/')}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post_prompt(base_url: str, prompt: dict, timeout: float = 60.0) -> dict:
    """POST a workflow prompt to the ComfyUI /prompt endpoint."""
    url = f"{base_url.rstrip('/')}/prompt"
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_system_stats(base_url: str) -> dict:
    """Query /system_stats to get ComfyUI version and system info."""
    return _api_get(base_url, "/system_stats")


def query_object_info(base_url: str) -> dict:
    """Query /object_info to get available node class names."""
    return _api_get(base_url, "/object_info")


def extract_comfy_version(stats: dict) -> str:
    """Extract ComfyUI version from system_stats response."""
    system = stats.get("system", {})
    return system.get("comfyui_version", "unknown")


def extract_memory_mode(stats: dict) -> str:
    """Extract active memory-management mode from system_stats response."""
    devices = stats.get("devices", [])
    if devices:
        return devices[0].get("vram_state", "unknown")
    return "unknown"


def extract_node_classes(object_info: dict) -> list[str]:
    """Extract node class names from object_info response."""
    return sorted(object_info.keys())


def query_prompt_history(
    base_url: str,
    prompt_id: str,
    *,
    timeout: float = 30.0,
    max_polls: int = 30,
    poll_interval: float = 2.0,
) -> dict:
    """Poll /history/{prompt_id} until the prompt finishes or polls are exhausted."""
    for _ in range(max_polls):
        data = _api_get(base_url, f"/history/{prompt_id}", timeout=timeout)
        if prompt_id in data:
            return data[prompt_id]
        time.sleep(poll_interval)
    return {}


def check_cache_reuse(
    base_url: str,
    first_prompt_id: str,
    reuse_prompt_id: str,
    *,
    first_elapsed: float = 0.0,
    reuse_elapsed: float = 0.0,
    exit_node_id: str = "3",
) -> tuple[bool, str]:
    """Detect whether the reuse prompt reused cached node outputs.

    Same detection contract as the checkpoint harness:
    1. Output-based detection from /history outputs dict.
    2. ``execution_cached`` status messages.
    3. Timing comparison fallback when WIDEN's application-level cache hit
       leaves no node-output evidence.
    """
    try:
        first_hist = query_prompt_history(base_url, first_prompt_id)
        reuse_hist = query_prompt_history(base_url, reuse_prompt_id)

        if not first_hist or not reuse_hist:
            return False, "history unavailable for one or both prompts"

        first_outputs = first_hist.get("outputs", {})
        reuse_outputs = reuse_hist.get("outputs", {})

        first_had_exit_output = exit_node_id in first_outputs
        reuse_had_exit_output = exit_node_id in reuse_outputs

        if first_had_exit_output and not reuse_had_exit_output:
            return True, (
                f"WIDENExit node {exit_node_id} executed in first prompt "
                f"({first_prompt_id}) but was cached in reuse prompt "
                f"({reuse_prompt_id})"
            )

        reuse_status = reuse_hist.get("status", {})
        status_messages = reuse_status.get("messages", [])
        cached_nodes: list[str] = []
        for msg in status_messages:
            if isinstance(msg, list) and len(msg) >= 2:
                if msg[0] == "execution_cached":
                    cached_nodes.extend(msg[1].get("nodes", []))

        if exit_node_id in cached_nodes:
            return True, (
                f"WIDENExit node {exit_node_id} reported as execution_cached "
                f"in reuse prompt ({reuse_prompt_id})"
            )

        if reuse_had_exit_output:
            return False, (
                f"WIDENExit node {exit_node_id} re-executed in reuse prompt "
                f"({reuse_prompt_id}) — cache was NOT reused"
            )

        if first_elapsed > 0 and reuse_elapsed > 0:
            speedup = first_elapsed / reuse_elapsed
            if speedup >= 2.0:
                return True, (
                    f"application-level cache reuse confirmed via timing: "
                    f"first={first_elapsed:.3f}s, reuse={reuse_elapsed:.3f}s, "
                    f"speedup={speedup:.1f}x (threshold: 2.0x)"
                )
            return False, (
                f"timing does not indicate cache reuse: "
                f"first={first_elapsed:.3f}s, reuse={reuse_elapsed:.3f}s, "
                f"speedup={speedup:.1f}x (threshold: 2.0x)"
            )

        return False, (
            f"unable to determine cache status: first_exit_output="
            f"{first_had_exit_output}, reuse_exit_output={reuse_had_exit_output}, "
            f"no timing data available"
        )
    except Exception as exc:
        return False, f"cache reuse check failed: {exc}"


# ---------------------------------------------------------------------------
# Workflow builders — construct ComfyUI API prompt dicts
# ---------------------------------------------------------------------------


def build_diffusion_save_workflow(source_diffusion_model: str) -> dict:
    """Build a standalone diffusion-model WIDEN save_model workflow.

    Pipeline:
      UNETLoader(unet_name=<source>, weight_dtype="default")
        -> WIDENEntry(model=...)       -- no clip/vae => diffusion-only
        -> WIDENExit(save_model=True, model_name=<saved>)

    ``enable_cache=False`` so the save always performs the full
    materialization, producing a fresh artifact under the diffusion_models
    folder.

    Both ``unet_name`` and ``weight_dtype`` are required inputs of
    ComfyUI's UNETLoader; omitting ``weight_dtype`` causes prompt
    validation to reject the workflow with required_input_missing before
    the diffusion-model load path executes.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": source_diffusion_model,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "WIDENEntry",
            "inputs": {
                "model": ["1", 0],
            },
        },
        "3": {
            "class_type": "WIDENExit",
            "inputs": {
                "widen": ["2", 0],
                "save_model": True,
                "enable_cache": False,
                "model_name": _SAVED_ARTIFACT_NAME.removesuffix(".safetensors"),
            },
        },
    }


def build_diffusion_downstream_workflow(
    saved_artifact_name: str,
    *,
    companion_clip: str,
    companion_clip2: str = "",
    companion_clip_type: str = "stable_diffusion",
    companion_vae: str,
    width: int = 256,
    height: int = 256,
    steps: int = 1,
    cfg: float = 1.0,
    seed: int = 42,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
) -> dict:
    """Build a downstream workflow that loads the saved diffusion-model
    artifact via UNETLoader and supplies CLIP / VAE separately.

    AC: @full-saved-model-output ac-diffusion-model-companion-separation

    Pipeline:
        UNETLoader(<saved>)         -> MODEL
        DualCLIPLoader / CLIPLoader -> CLIP   (separate)
        VAELoader                   -> VAE    (separate)
        CLIPTextEncode (positive + negative) -> CONDITIONING
        EmptyLatentImage -> KSampler -> VAEDecode -> SaveImage

    ``saved_artifact_name`` is the full filename including extension
    (e.g. ``ecaj_diffusion_model_validation_save.safetensors``) since
    UNETLoader's ``unet_name`` accepts the suffixed filename.
    """
    if companion_clip2:
        clip_node = {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": companion_clip,
                "clip_name2": companion_clip2,
                "type": companion_clip_type,
            },
        }
    else:
        clip_node = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": companion_clip,
                "type": companion_clip_type,
            },
        }

    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": saved_artifact_name,
                "weight_dtype": "default",
            },
        },
        "2": clip_node,
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": companion_vae,
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "validation test",
                "clip": ["2", 0],
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["2", 0],
            },
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "ecaj_diffusion_model_validation",
            },
        },
    }


def build_diffusion_cache_reuse_workflow(source_diffusion_model: str) -> dict:
    """Build a workflow that re-runs diffusion save with enable_cache=True.

    Structurally identical to ``build_diffusion_save_workflow`` but with
    ``enable_cache=True`` so the WIDEN application-level cache should hit
    on the second run and skip recomputing the merge.
    """
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": source_diffusion_model,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "WIDENEntry",
            "inputs": {
                "model": ["1", 0],
            },
        },
        "3": {
            "class_type": "WIDENExit",
            "inputs": {
                "widen": ["2", 0],
                "save_model": True,
                "enable_cache": True,
                "model_name": _SAVED_ARTIFACT_NAME.removesuffix(".safetensors"),
            },
        },
    }


# ---------------------------------------------------------------------------
# Submission and classification — reuse the contract from the checkpoint harness
# ---------------------------------------------------------------------------


def _extract_node_errors_from_history(history: dict) -> str:
    """Extract node error details from a prompt history entry."""
    status = history.get("status", {})
    status_str = status.get("status_str", "")
    if status_str == "error":
        messages = status.get("messages", [])
        for msg in messages:
            if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                err_data = msg[1] if isinstance(msg[1], dict) else {}
                node_type = err_data.get("node_type", "unknown")
                exception_message = err_data.get(
                    "exception_message",
                    "unknown error",
                )
                return f"node_error({node_type}): {exception_message}"
        return f"execution_error: status_str={status_str}"
    return ""


def submit_workflow(
    base_url: str,
    workflow: dict,
    name: str,
) -> WorkflowResult:
    """Submit a workflow prompt, wait for completion, and classify the result."""
    result = WorkflowResult(name=name)
    t0 = time.monotonic()
    try:
        resp = _api_post_prompt(base_url, workflow)
        prompt_id = resp.get("prompt_id", "")
        result.prompt_id = prompt_id
        if not prompt_id:
            result.accepted = False
            node_errors = resp.get("node_errors", {})
            error_msg = resp.get("error", {})
            if node_errors:
                result.error = f"node_errors: {json.dumps(node_errors)}"
            elif error_msg:
                result.error = f"api_error: {json.dumps(error_msg)}"
            else:
                result.error = "rejected: no prompt_id returned"
            return result

        history = query_prompt_history(base_url, prompt_id)
        if not history:
            result.accepted = False
            result.error = "execution_timeout: prompt accepted but never completed in /history"
            return result

        node_error = _extract_node_errors_from_history(history)
        if node_error:
            result.accepted = False
            result.error = node_error
        else:
            result.accepted = True
            result.outputs = history.get("outputs", {})
    except urllib.error.HTTPError as exc:
        result.accepted = False
        try:
            body = exc.read().decode("utf-8", errors="replace")
            result.error = f"http_{exc.code}: {body[:500]}"
        except Exception:
            result.error = f"http_{exc.code}"
    except urllib.error.URLError as exc:
        result.accepted = False
        result.error = f"url_error: {exc.reason}"
    except Exception as exc:
        result.accepted = False
        result.error = f"unexpected: {exc}"
    finally:
        result.elapsed_seconds = time.monotonic() - t0
    return result


def _extract_failing_node_type(error: str) -> str:
    """Extract the failing node class type from a ``node_error(NodeType): ...`` string."""
    if error.startswith("node_error("):
        paren_end = error.find(")")
        if paren_end > len("node_error("):
            return error[len("node_error(") : paren_end]
    return ""


def classify_failure(result: WorkflowResult) -> str:
    """Classify a workflow result into a failure category."""
    if result.accepted:
        return "none"
    error = result.error.lower()
    if "memory" in error or "vram" in error or "oom" in error:
        return "memory_mode"
    if "no output" in error or "scheduler" in error or "not executable" in error:
        return "scheduling"
    if "save" in error and ("fail" in error or "error" in error):
        return "save"
    if "unet" in error or "diffusion" in error or "loader" in error or "load" in error:
        return "loader"
    if "ksampler" in error or "downstream" in error or "decode" in error:
        return "downstream"
    if "cache" in error or "reuse" in error:
        return "cache"
    if "url_error" in error or "connection" in error:
        return "connection"
    return "unknown"


# ---------------------------------------------------------------------------
# Validation runner — only reachable after all guards pass
# ---------------------------------------------------------------------------


def run_validation(
    comfy_api_url: str,
    source_diffusion_model: str,
    companion_clip: str,
    companion_vae: str,
    report_output: str,
    *,
    companion_clip2: str = "",
    companion_clip_type: str = "stable_diffusion",
    width: int = 256,
    height: int = 256,
    steps: int = 1,
    cfg: float = 1.0,
    seed: int = 42,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
) -> DiffusionModelValidationReport:
    """Run live ComfyUI diffusion-model saved-output validation via the API.

    Only called after all guards pass.  Submits workflow prompts to the
    ComfyUI API and classifies results.
    """
    report = DiffusionModelValidationReport(
        comfy_api_url=comfy_api_url,
        source_diffusion_model=source_diffusion_model,
        companion_clip=companion_clip,
        companion_clip2=companion_clip2,
        companion_clip_type=companion_clip_type,
        companion_vae=companion_vae,
    )
    start_time = time.time()

    try:
        try:
            stats = query_system_stats(comfy_api_url)
            report.comfy_version = extract_comfy_version(stats)
            report.memory_mode = extract_memory_mode(stats)
        except Exception as exc:
            report.errors.append(f"system_stats query failed: {exc}")
            report.failure_categories["environment"] = str(exc)
            report.duration_seconds = time.time() - start_time
            return report

        try:
            obj_info = query_object_info(comfy_api_url)
            report.node_classes = extract_node_classes(obj_info)
        except Exception as exc:
            report.errors.append(f"object_info query failed: {exc}")

        # --- Submit diffusion-model save_model workflow ---
        save_wf = build_diffusion_save_workflow(source_diffusion_model)
        report.diffusion_save_workflow_shape = save_wf
        report.diffusion_save_result = submit_workflow(
            comfy_api_url,
            save_wf,
            "diffusion_model_save",
        )
        report.failure_categories["save"] = classify_failure(
            report.diffusion_save_result,
        )

        # Determine saved artifact filename — fixed name under diffusion_models.
        saved_artifact_filename = ""
        if report.diffusion_save_result.accepted:
            saved_artifact_filename = _SAVED_ARTIFACT_NAME
            report.saved_artifact_path = saved_artifact_filename
            report.saved_artifact_classification = "diffusion_model"
        else:
            report.saved_artifact_path = ""
            report.saved_artifact_classification = "not_saved"

        # --- Load saved artifact + run downstream workflow ---
        if saved_artifact_filename:
            downstream_wf = build_diffusion_downstream_workflow(
                saved_artifact_filename,
                companion_clip=companion_clip,
                companion_clip2=companion_clip2,
                companion_clip_type=companion_clip_type,
                companion_vae=companion_vae,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
                sampler_name=sampler_name,
                scheduler=scheduler,
                batch_size=batch_size,
            )
            downstream_full_result = submit_workflow(
                comfy_api_url,
                downstream_wf,
                "downstream_ksampler",
            )
            failing_node = _extract_failing_node_type(
                downstream_full_result.error,
            )
            if downstream_full_result.accepted:
                report.diffusion_loader_result = WorkflowResult(
                    name="diffusion_loader",
                    accepted=True,
                    prompt_id=downstream_full_result.prompt_id,
                )
                report.downstream_result = downstream_full_result
            elif failing_node == "UNETLoader":
                # The diffusion-model loader itself failed — downstream never ran.
                # This is the bug the harness is designed to surface: a saved
                # artifact that can't load via Comfy's standalone diffusion
                # loader is a round-trip failure.
                report.diffusion_loader_result = WorkflowResult(
                    name="diffusion_loader",
                    accepted=False,
                    prompt_id=downstream_full_result.prompt_id,
                    error=downstream_full_result.error,
                )
                report.downstream_result = WorkflowResult(
                    name="downstream_ksampler",
                    accepted=False,
                    prompt_id=downstream_full_result.prompt_id,
                    error="skipped: UNETLoader (diffusion-model loader) failed",
                )
            else:
                # Loader succeeded; another node downstream failed.
                report.diffusion_loader_result = WorkflowResult(
                    name="diffusion_loader",
                    accepted=True,
                    prompt_id=downstream_full_result.prompt_id,
                )
                report.downstream_result = downstream_full_result
            report.failure_categories["loader"] = classify_failure(
                report.diffusion_loader_result,
            )
            report.failure_categories["downstream"] = classify_failure(
                report.downstream_result,
            )
        else:
            skip_reason = "skipped: diffusion-model save failed, no artifact to validate"
            report.diffusion_loader_result = WorkflowResult(
                name="diffusion_loader",
                accepted=False,
                error=skip_reason,
            )
            report.downstream_result = WorkflowResult(
                name="downstream_ksampler",
                accepted=False,
                error=skip_reason,
            )
            report.failure_categories["loader"] = "save"
            report.failure_categories["downstream"] = "save"

        # --- Submit cache reuse workflow ---
        cache_wf = build_diffusion_cache_reuse_workflow(source_diffusion_model)
        report.cache_reuse_result = submit_workflow(
            comfy_api_url,
            cache_wf,
            "diffusion_cache_reuse",
        )
        report.failure_categories["cache"] = classify_failure(
            report.cache_reuse_result,
        )

        if report.diffusion_save_result.accepted and report.cache_reuse_result.accepted:
            cache_reused, detail = check_cache_reuse(
                comfy_api_url,
                report.diffusion_save_result.prompt_id,
                report.cache_reuse_result.prompt_id,
                first_elapsed=report.diffusion_save_result.elapsed_seconds,
                reuse_elapsed=report.cache_reuse_result.elapsed_seconds,
            )
            report.cache_reuse_detail = detail
            if not cache_reused:
                report.failure_categories["cache"] = "cache_not_reused"
        elif report.cache_reuse_result.accepted:
            report.cache_reuse_detail = "first save was rejected; cannot verify reuse"
        else:
            report.cache_reuse_detail = "cache reuse prompt was rejected"

        memory_mode_failures = [
            wf_name
            for wf_name, category in report.failure_categories.items()
            if category == "memory_mode"
        ]
        if memory_mode_failures:
            report.failure_categories["memory_mode"] = (
                f"memory_mode failure in: {', '.join(memory_mode_failures)}"
            )
        else:
            report.failure_categories["memory_mode"] = "none"

    except Exception as exc:
        report.errors.append(f"validation error: {exc}")
    finally:
        report.duration_seconds = time.time() - start_time

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point with full guard checking.

    Returns 0 on success, 1 on guard refusal, 2 on validation error.
    """
    guard = check_guards(argv=argv)

    if not guard.passed:
        print(format_refusal(guard), file=sys.stderr)
        return 1

    args = parse_validated_args(argv)

    report = run_validation(
        comfy_api_url=args.comfy_api_url,
        source_diffusion_model=args.source_diffusion_model,
        companion_clip=args.companion_clip,
        companion_vae=args.companion_vae,
        report_output=args.report_output,
        companion_clip2=args.companion_clip2,
        companion_clip_type=args.companion_clip_type,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        batch_size=args.batch_size,
    )

    report_text = format_report(report)
    print(report_text)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.report_output)),
        exist_ok=True,
    )
    with open(args.report_output, "w") as f:
        f.write(report.to_json())

    print(f"\nJSON report written to: {args.report_output}")

    if report.errors:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
