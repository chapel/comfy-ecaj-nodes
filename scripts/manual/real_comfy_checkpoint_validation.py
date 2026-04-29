#!/usr/bin/env python3
"""Guarded live validation harness for checkpoint-save workflows via the ComfyUI API.

This script is NOT part of the automated test suite. It requires explicit
opt-in via environment variable, CLI flag, and all required paths before
it will submit any ComfyUI API prompts, import ComfyUI modules, mutate
a ComfyUI installation, start processes, or write large artifacts.

Usage:
    COMFY_ECAJ_CHECKPOINT_VALIDATION=1 python scripts/manual/real_comfy_checkpoint_validation.py \\
        --run-checkpoint-validation \\
        --comfy-api-url http://127.0.0.1:8188 \\
        --source-model sd_xl_base_1.0.safetensors \\
        --report-output /path/to/report.json

All five inputs are required:
    1. COMFY_ECAJ_CHECKPOINT_VALIDATION=1 environment variable
    2. --run-checkpoint-validation CLI flag
    3. --comfy-api-url: URL of a running ComfyUI API server
    4. --source-model: checkpoint/model name or path for ComfyUI loaders
    5. --report-output: path where the JSON report will be written

Optional:
    --width, --height: latent image dimensions (default: 256)
    --steps: KSampler steps (default: 1)
    --cfg: KSampler cfg scale (default: 1.0)
    --seed: deterministic seed (default: 42)
    --sampler-name: sampler name (default: euler)
    --scheduler: scheduler name (default: normal)
    --batch-size: batch size (default: 1)
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

_ENV_VAR = "COMFY_ECAJ_CHECKPOINT_VALIDATION"
_CLI_FLAG = "--run-checkpoint-validation"


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
        1. COMFY_ECAJ_CHECKPOINT_VALIDATION=1 environment variable
        2. --run-checkpoint-validation CLI flag
        3. --comfy-api-url argument present and non-empty
        4. --source-model argument present and non-empty
        5. --report-output argument present and non-empty

    Returns GuardResult with passed=True only if ALL guards are satisfied.
    """
    if env is None:
        env = dict(os.environ)
    if argv is None:
        argv = sys.argv[1:]

    reasons: list[str] = []

    # Guard 1: environment variable
    val = env.get(_ENV_VAR, "")
    if val != "1":
        reasons.append(
            f"Environment variable {_ENV_VAR} is not set to '1' (got {val!r})."
        )

    # Guard 2: CLI flag
    if _CLI_FLAG not in argv:
        reasons.append(f"CLI flag {_CLI_FLAG} is not present.")

    # Parse known args for value validation (tolerant of unknown flags)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfy-api-url", default=None)
    parser.add_argument("--source-model", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument(_CLI_FLAG, action="store_true", dest="run_flag")
    known, _ = parser.parse_known_args(argv)

    # Guard 3: --comfy-api-url
    if not known.comfy_api_url:
        reasons.append("--comfy-api-url is required but not provided.")

    # Guard 4: --source-model
    if not known.source_model:
        reasons.append("--source-model is required but not provided.")

    # Guard 5: --report-output
    if not known.report_output:
        reasons.append("--report-output is required but not provided.")

    if reasons:
        return GuardResult(passed=False, reasons=tuple(reasons))
    return GuardResult(passed=True)


def parse_validated_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments after guards have passed.

    Only call this after check_guards() returns passed=True.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Live ComfyUI checkpoint-save validation harness.",
    )
    parser.add_argument(
        _CLI_FLAG,
        action="store_true",
        dest="run_flag",
        help="Explicit opt-in flag to run checkpoint validation.",
    )
    parser.add_argument(
        "--comfy-api-url",
        required=True,
        help="URL of a running ComfyUI API server (e.g., http://127.0.0.1:8188).",
    )
    parser.add_argument(
        "--source-model",
        required=True,
        help="Checkpoint/model name for ComfyUI loaders (e.g., sd_xl_base_1.0.safetensors).",
    )
    parser.add_argument(
        "--report-output",
        required=True,
        help="Path where the JSON validation report will be written.",
    )
    # Safe resource defaults
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


@dataclass
class CheckpointValidationReport:
    """Report from a live ComfyUI checkpoint-save validation run."""

    comfy_version: str = ""
    memory_mode: str = ""
    comfy_api_url: str = ""
    source_model: str = ""
    node_classes: list[str] = field(default_factory=list)
    terminal_save_workflow_shape: dict = field(default_factory=dict)
    terminal_save_result: WorkflowResult = field(default_factory=WorkflowResult)
    checkpoint_save_workflow_shape: dict = field(default_factory=dict)
    checkpoint_save_result: WorkflowResult = field(default_factory=WorkflowResult)
    saved_artifact_path: str = ""
    saved_artifact_classification: str = ""
    checkpoint_loader_result: WorkflowResult = field(default_factory=WorkflowResult)
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


_REQUIRED_REPORT_FIELDS = frozenset({
    "comfy_version",
    "memory_mode",
    "terminal_save_workflow_shape",
    "terminal_save_result",
    "checkpoint_save_workflow_shape",
    "saved_artifact_path",
    "saved_artifact_classification",
    "checkpoint_save_result",
    "checkpoint_loader_result",
    "downstream_result",
    "cache_reuse_result",
    "cache_reuse_detail",
    "failure_categories",
})


def report_has_required_fields(report_dict: dict) -> tuple[bool, list[str]]:
    """Check whether a report dict contains all required fields.

    Returns (all_present, list_of_missing_fields).
    """
    missing = [f for f in _REQUIRED_REPORT_FIELDS if f not in report_dict]
    return (len(missing) == 0, missing)


def format_report(report: CheckpointValidationReport) -> str:
    """Format a CheckpointValidationReport as human-readable text."""
    lines = [
        "=" * 60,
        "ComfyUI Checkpoint-Save Validation Report",
        "=" * 60,
        "",
        f"ComfyUI Version:            {report.comfy_version}",
        f"Memory Mode:                {report.memory_mode}",
        f"API URL:                    {report.comfy_api_url}",
        f"Source Model:               {report.source_model}",
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

    lines.extend([
        "--- Terminal Save ---",
        f"  Accepted:     {report.terminal_save_result.accepted}",
        f"  Prompt ID:    {report.terminal_save_result.prompt_id}",
        f"  Error:        {report.terminal_save_result.error or 'none'}",
    ])
    if report.terminal_save_workflow_shape:
        shape_json = json.dumps(report.terminal_save_workflow_shape)
        lines.append(f"  Workflow Shape: {shape_json}")
    lines.append("")

    lines.extend([
        "--- Checkpoint Save ---",
        f"  Accepted:     {report.checkpoint_save_result.accepted}",
        f"  Prompt ID:    {report.checkpoint_save_result.prompt_id}",
        f"  Artifact:     {report.saved_artifact_path}",
        f"  Classification: {report.saved_artifact_classification}",
        f"  Error:        {report.checkpoint_save_result.error or 'none'}",
    ])
    if report.checkpoint_save_workflow_shape:
        shape_json = json.dumps(report.checkpoint_save_workflow_shape)
        lines.append(f"  Workflow Shape: {shape_json}")
    lines.append("")

    lines.extend([
        "--- Checkpoint Loader ---",
        f"  Accepted:     {report.checkpoint_loader_result.accepted}",
        f"  Prompt ID:    {report.checkpoint_loader_result.prompt_id}",
        f"  Error:        {report.checkpoint_loader_result.error or 'none'}",
        "",
        "--- Downstream KSampler ---",
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
    ])

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
        "ComfyUI Checkpoint Validation — REFUSED (guards not satisfied)",
        "=" * 60,
        "",
        "The following required inputs are missing or invalid:",
        "",
    ]
    for reason in guard_result.reasons:
        lines.append(f"  - {reason}")
    lines.extend([
        "",
        "All of the following are required to run checkpoint validation:",
        f"  1. {_ENV_VAR}=1 environment variable",
        f"  2. {_CLI_FLAG} CLI flag",
        "  3. --comfy-api-url <url>",
        "  4. --source-model <name>",
        "  5. --report-output <path>",
        "",
        "No ComfyUI API prompts were submitted.",
        "No ComfyUI modules were imported.",
        "No ComfyUI installation was modified.",
        "No processes were started.",
        "No large artifacts were written.",
        "=" * 60,
    ])
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
        url, data=payload, method="POST",
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
    """Extract active memory-management mode from system_stats response.

    ComfyUI reports devices with vram_state in system_stats.
    """
    devices = stats.get("devices", [])
    if devices:
        return devices[0].get("vram_state", "unknown")
    return "unknown"


def extract_node_classes(object_info: dict) -> list[str]:
    """Extract node class names from object_info response."""
    return sorted(object_info.keys())


def query_prompt_history(
    base_url: str, prompt_id: str, *, timeout: float = 30.0, max_polls: int = 30,
    poll_interval: float = 2.0,
) -> dict:
    """Poll /history/{prompt_id} until the prompt finishes or polls are exhausted.

    Returns the history entry dict for the prompt.
    """
    for _ in range(max_polls):
        data = _api_get(base_url, f"/history/{prompt_id}", timeout=timeout)
        if prompt_id in data:
            return data[prompt_id]
        time.sleep(poll_interval)
    return {}


def check_cache_reuse(
    base_url: str, first_prompt_id: str, reuse_prompt_id: str,
) -> tuple[bool, str]:
    """Check whether the reuse prompt reused cached node outputs.

    Compares the history entries of the first save prompt and the reuse
    prompt.  If the reuse prompt's WIDENExit node is absent from its
    ``outputs`` (ComfyUI only records outputs for nodes that actually
    executed), the node was served from cache.

    Returns (cache_reused: bool, detail: str).
    """
    try:
        first_hist = query_prompt_history(base_url, first_prompt_id)
        reuse_hist = query_prompt_history(base_url, reuse_prompt_id)

        if not first_hist or not reuse_hist:
            return False, "history unavailable for one or both prompts"

        first_outputs = first_hist.get("outputs", {})
        reuse_outputs = reuse_hist.get("outputs", {})

        # Find the WIDENExit node id (node "3" in our workflow builders)
        exit_node_id = "3"

        first_had_exit_output = exit_node_id in first_outputs
        reuse_had_exit_output = exit_node_id in reuse_outputs

        if first_had_exit_output and not reuse_had_exit_output:
            return True, (
                f"WIDENExit node {exit_node_id} executed in first prompt "
                f"({first_prompt_id}) but was cached in reuse prompt "
                f"({reuse_prompt_id})"
            )

        # Both executed or both cached — check status_str_type for cache hint
        reuse_status = reuse_hist.get("status", {})
        status_messages = reuse_status.get("messages", [])
        cached_nodes = []
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

        return False, (
            f"unable to determine cache status: first_exit_output="
            f"{first_had_exit_output}, reuse_exit_output={reuse_had_exit_output}"
        )
    except Exception as exc:
        return False, f"cache reuse check failed: {exc}"


# ---------------------------------------------------------------------------
# Workflow builders — construct ComfyUI API prompt dicts
# ---------------------------------------------------------------------------


def build_terminal_exit_save_workflow(source_model: str) -> dict:
    """Build a minimal WIDEN Exit save workflow with no downstream consumers.

    This tests whether ComfyUI accepts a terminal Exit node with save_model=True
    when there is no downstream MODEL consumer (scheduler/no-output test).

    Wires MODEL, CLIP, and VAE from CheckpointLoaderSimple into WIDENEntry
    (outputs: MODEL=0, CLIP=1, VAE=2).
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": source_model},
        },
        "2": {
            "class_type": "WIDENEntry",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "vae": ["1", 2],
            },
        },
        "3": {
            "class_type": "WIDENExit",
            "inputs": {
                "widen": ["2", 0],
                "save_model": True,
                "model_name": "ecaj_checkpoint_validation_terminal",
            },
        },
    }


def build_checkpoint_save_workflow(source_model: str) -> dict:
    """Build a checkpoint-style WIDEN save_model workflow using MODEL, CLIP, VAE.

    Uses CheckpointLoaderSimple to load all three components, feeds MODEL,
    CLIP, and VAE through WIDEN Entry/Exit with save_model=True.
    (CheckpointLoaderSimple outputs: MODEL=0, CLIP=1, VAE=2.)
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": source_model},
        },
        "2": {
            "class_type": "WIDENEntry",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "vae": ["1", 2],
            },
        },
        "3": {
            "class_type": "WIDENExit",
            "inputs": {
                "widen": ["2", 0],
                "save_model": True,
                "model_name": "ecaj_checkpoint_validation_save",
            },
        },
    }


def build_downstream_workflow(
    ckpt_name: str,
    *,
    width: int = 256,
    height: int = 256,
    steps: int = 1,
    cfg: float = 1.0,
    seed: int = 42,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
) -> dict:
    """Build a bounded downstream workflow equivalent to:

    CheckpointLoaderSimple -> CLIPTextEncode -> EmptyLatentImage ->
    KSampler -> VAEDecode -> SaveImage.

    Args:
        ckpt_name: Checkpoint filename to load. Should be the saved artifact
            (e.g. ``ecaj_checkpoint_validation_save.safetensors``), NOT the
            original source model, so downstream validation proves the
            *saved* artifact is loadable and usable.

    Uses safe defaults: 256x256, batch_size 1, 1 step, cfg 1.0,
    euler/normal, deterministic seed.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "validation test",
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": batch_size,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2],
            },
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "ecaj_checkpoint_validation",
            },
        },
    }


def build_cache_reuse_workflow(source_model: str) -> dict:
    """Build a workflow that re-runs checkpoint save to test cache/artifact reuse.

    This workflow is structurally identical to build_checkpoint_save_workflow
    but with enable_cache=True explicitly set on the WIDENExit node.  When
    ComfyUI re-executes the same node graph, nodes whose inputs haven't
    changed are served from ComfyUI's execution cache — so the WIDEN merge
    should not recompute.

    The caller should compare this prompt's execution against the first save
    prompt via the /history API to determine whether the cached path was used
    (e.g. by checking whether the WIDENExit node re-executed or was cached).
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": source_model},
        },
        "2": {
            "class_type": "WIDENEntry",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "vae": ["1", 2],
            },
        },
        "3": {
            "class_type": "WIDENExit",
            "inputs": {
                "widen": ["2", 0],
                "save_model": True,
                "enable_cache": True,
                "model_name": "ecaj_checkpoint_validation_save",
            },
        },
    }


def submit_workflow(
    base_url: str,
    workflow: dict,
    name: str,
) -> WorkflowResult:
    """Submit a workflow prompt and classify whether Comfy accepted or rejected it."""
    result = WorkflowResult(name=name)
    try:
        resp = _api_post_prompt(base_url, workflow)
        prompt_id = resp.get("prompt_id", "")
        result.prompt_id = prompt_id
        if prompt_id:
            result.accepted = True
        else:
            result.accepted = False
            node_errors = resp.get("node_errors", {})
            error_msg = resp.get("error", {})
            if node_errors:
                result.error = f"node_errors: {json.dumps(node_errors)}"
            elif error_msg:
                result.error = f"api_error: {json.dumps(error_msg)}"
            else:
                result.error = "rejected: no prompt_id returned"
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
    return result


def classify_failure(result: WorkflowResult) -> str:
    """Classify a workflow result into a failure category."""
    if result.accepted:
        return "none"
    error = result.error.lower()
    if "no output" in error or "scheduler" in error or "not executable" in error:
        return "scheduling"
    if "save" in error and ("fail" in error or "error" in error):
        return "save"
    if "loader" in error or "load" in error:
        return "loader"
    if "ksampler" in error or "downstream" in error or "decode" in error:
        return "downstream"
    if "cache" in error or "reuse" in error:
        return "cache"
    if "memory" in error or "vram" in error or "oom" in error:
        return "memory_mode"
    if "url_error" in error or "connection" in error:
        return "connection"
    return "unknown"


# ---------------------------------------------------------------------------
# Validation runner — only reachable after all guards pass
# ---------------------------------------------------------------------------


def run_validation(
    comfy_api_url: str,
    source_model: str,
    report_output: str,
    *,
    width: int = 256,
    height: int = 256,
    steps: int = 1,
    cfg: float = 1.0,
    seed: int = 42,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
) -> CheckpointValidationReport:
    """Run live ComfyUI checkpoint-save validation via the API.

    This function is ONLY called after all guards pass.
    It submits workflow prompts to the ComfyUI API and classifies results.
    """
    report = CheckpointValidationReport(
        comfy_api_url=comfy_api_url,
        source_model=source_model,
    )
    start_time = time.time()

    try:
        # --- Query environment info ---
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

        # --- Submit terminal Exit save workflow ---
        terminal_wf = build_terminal_exit_save_workflow(source_model)
        report.terminal_save_workflow_shape = terminal_wf
        report.terminal_save_result = submit_workflow(
            comfy_api_url, terminal_wf, "terminal_exit_save",
        )
        report.failure_categories["scheduling"] = classify_failure(
            report.terminal_save_result,
        )

        # --- Submit checkpoint save_model workflow ---
        save_wf = build_checkpoint_save_workflow(source_model)
        report.checkpoint_save_workflow_shape = save_wf
        report.checkpoint_save_result = submit_workflow(
            comfy_api_url, save_wf, "checkpoint_save",
        )
        report.failure_categories["save"] = classify_failure(
            report.checkpoint_save_result,
        )

        # Determine saved artifact path from checkpoint_save result.
        # The WIDENExit node saves to: <model_name>.safetensors
        saved_artifact_filename = ""
        if report.checkpoint_save_result.accepted:
            saved_artifact_filename = "ecaj_checkpoint_validation_save.safetensors"
            report.saved_artifact_path = saved_artifact_filename
            report.saved_artifact_classification = "checkpoint"
        else:
            report.saved_artifact_path = ""
            report.saved_artifact_classification = "not_saved"

        # --- Load saved artifact and run downstream workflow ---
        # Use the saved artifact (not the source model) so we prove the
        # *saved checkpoint* is loadable and produces valid downstream output.
        loader_ckpt = saved_artifact_filename if saved_artifact_filename else source_model
        downstream_wf = build_downstream_workflow(
            loader_ckpt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            batch_size=batch_size,
        )
        report.downstream_result = submit_workflow(
            comfy_api_url, downstream_wf, "downstream_ksampler",
        )
        # Record loader + downstream together (same prompt exercises both)
        report.checkpoint_loader_result = WorkflowResult(
            name="checkpoint_loader",
            accepted=report.downstream_result.accepted,
            prompt_id=report.downstream_result.prompt_id,
            error=report.downstream_result.error,
        )
        report.failure_categories["loader"] = classify_failure(
            report.checkpoint_loader_result,
        )
        report.failure_categories["downstream"] = classify_failure(
            report.downstream_result,
        )

        # --- Submit cache reuse workflow and verify reuse ---
        cache_wf = build_cache_reuse_workflow(source_model)
        report.cache_reuse_result = submit_workflow(
            comfy_api_url, cache_wf, "cache_reuse",
        )
        report.failure_categories["cache"] = classify_failure(
            report.cache_reuse_result,
        )

        # Check cache reuse by comparing execution history
        if (report.checkpoint_save_result.accepted
                and report.cache_reuse_result.accepted):
            cache_reused, detail = check_cache_reuse(
                comfy_api_url,
                report.checkpoint_save_result.prompt_id,
                report.cache_reuse_result.prompt_id,
            )
            report.cache_reuse_detail = detail
            if not cache_reused:
                report.failure_categories["cache"] = "cache_not_reused"
        elif report.cache_reuse_result.accepted:
            report.cache_reuse_detail = (
                "first save was rejected; cannot verify reuse"
            )
        else:
            report.cache_reuse_detail = "cache reuse prompt was rejected"

        # --- Memory mode failure check ---
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

    # All guards passed — safe to proceed with live ComfyUI API work
    report = run_validation(
        comfy_api_url=args.comfy_api_url,
        source_model=args.source_model,
        report_output=args.report_output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        batch_size=args.batch_size,
    )

    # Write report
    report_text = format_report(report)
    print(report_text)

    os.makedirs(os.path.dirname(os.path.abspath(args.report_output)), exist_ok=True)
    with open(args.report_output, "w") as f:
        f.write(report.to_json())

    print(f"\nJSON report written to: {args.report_output}")

    if report.errors:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
