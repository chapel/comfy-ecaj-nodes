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

Optional process-memory observation inputs (operator must supply explicitly):
    --comfy-pid <pid>: Linux pid of the running ComfyUI server. When
        supplied, the harness reads /proc/<pid>/status and
        /proc/<pid>/smaps_rollup at the before-run and after-return
        lifecycle points to record RSS, Swap, and Anonymous memory.
        OS-level "after save-time checkpoint writing" cannot be sampled
        from an external harness because the harness only regains
        control once ComfyUI's /history reports the prompt completed
        (i.e. after WIDENExit returned); the save-time evidence in the
        report therefore comes from the internal WIDEN ``[mem]`` log
        labels supplied via ``--comfy-log-path``, not from procfs.
    --comfy-log-path <path>: path to a ComfyUI log file that captures
        WIDEN memory log lines (``[mem] <label>: RSS=...``). When
        supplied, the harness parses the ``after-checkpoint-save`` and
        ``after-checkpoint-temp-model-release`` labels for each run.
    --repeat-cache-miss-runs <n>: number of additional cache-miss save
        runs to execute in the same ComfyUI process. Each repeat run
        uses a distinct model_name so cache-reuse cannot mask the save.
    --memory-accumulation-threshold-mb <int>: per-run-pair growth
        threshold in MB (RSS + Swap, post-return) above which the
        report marks repeated cache-miss memory as failed.
        Default: ``DEFAULT_ACCUMULATION_THRESHOLD_MB`` (256 MB).

The harness does not auto-discover a pid, log file, or systemd unit; if
the operator omits these inputs, the report records the observations as
unavailable rather than fabricating values or refusing to run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
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
        reasons.append(f"Environment variable {_ENV_VAR} is not set to '1' (got {val!r}).")

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
    # Optional process-memory observation inputs — operator must supply
    # each one explicitly; the harness never auto-discovers them.
    parser.add_argument(
        "--comfy-pid",
        type=int,
        default=None,
        help=(
            "Linux pid of the running ComfyUI process. When supplied, the "
            "report records RSS/Swap/Anonymous from procfs at each run's "
            "before-run and after-return lifecycle points. The OS-level "
            "save-time observation is recorded as unavailable because the "
            "external harness only regains control after WIDENExit returns; "
            "save-time evidence comes from --comfy-log-path."
        ),
    )
    parser.add_argument(
        "--comfy-log-path",
        default=None,
        help=(
            "Path to a ComfyUI log file capturing WIDEN [mem] log lines. "
            "When supplied, the report parses after-checkpoint-save and "
            "after-checkpoint-temp-model-release labels for each run."
        ),
    )
    parser.add_argument(
        "--repeat-cache-miss-runs",
        type=int,
        default=0,
        help=(
            "Number of additional cache-miss save runs to execute in the "
            "same ComfyUI process. Each repeat uses a distinct model_name "
            "so cache-reuse cannot mask the save."
        ),
    )
    parser.add_argument(
        "--memory-accumulation-threshold-mb",
        type=int,
        default=DEFAULT_ACCUMULATION_THRESHOLD_MB,
        help=(
            "Per-run-pair RSS+Swap growth threshold in MB (post-return). "
            "Repeated cache-miss memory is reported as failed when the "
            "growth exceeds this value. "
            f"Default: {DEFAULT_ACCUMULATION_THRESHOLD_MB} MB."
        ),
    )
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


# ---------------------------------------------------------------------------
# Process-memory observation types and helpers
# ---------------------------------------------------------------------------


DEFAULT_ACCUMULATION_THRESHOLD_MB = 256
"""Default per-run-pair growth budget for repeated cache-miss memory checks.

A delta greater than this many megabytes between the first repeat and the
final repeat's post-return process-memory observation marks the repeated
cache-miss memory result as failed. This constant is recorded in the JSON
report so the threshold is auditable; the CLI ``--memory-accumulation-
threshold-mb`` flag overrides it for a given run.
"""


@dataclass
class ProcessMemoryObservation:
    """Operating-system process-memory snapshot for one labeled point.

    All byte fields are in kilobytes to match the units used by
    ``/proc/<pid>/status`` and ``/proc/<pid>/smaps_rollup``. ``None``
    means the field was not readable; ``available`` is False whenever
    the harness could not obtain *any* numeric field.
    """

    label: str = ""
    available: bool = False
    source: str = ""
    pid: int | None = None
    rss_kb: int | None = None
    swap_kb: int | None = None
    anonymous_kb: int | None = None
    error: str = ""
    timestamp: float = 0.0


@dataclass
class LogMemoryObservation:
    """Parsed WIDEN ``[mem] <label>: ...`` log line, if available."""

    label: str = ""
    available: bool = False
    rss_mb: int | None = None
    vram_alloc_mb: int | None = None
    vram_reserved_mb: int | None = None
    raw_line: str = ""
    error: str = ""


@dataclass
class CheckpointSaveRunReport:
    """Per-run record for a checkpoint save with process-memory observations.

    Captures the save result and process/log memory observations around
    one checkpoint-save run. Used both for the primary checkpoint save
    in :func:`run_validation` and for each entry in the repeated
    cache-miss accumulation series in :func:`run_repeated_cache_miss_runs`.

    ``cache_miss_identity`` is a per-run discriminator that ensures
    the save cannot be served by the artifact cache (repeat runs use
    a fresh identity each iteration; the primary save uses
    ``"primary-checkpoint-save"``). Process-memory observations are
    recorded as explicit unavailable entries when the operator did not
    supply ``--comfy-pid``; log observations are recorded as unavailable
    when ``--comfy-log-path`` is not supplied or the labels are missing.

    ``process_memory_after_save`` is always recorded as unavailable from
    this external harness: ``submit_workflow`` blocks on ComfyUI's
    ``/history`` endpoint and only returns once the prompt has fully
    completed (i.e. after WIDENExit returned), so the harness has no
    opportunity to sample procfs at the actual save-time lifecycle
    point. The report therefore exposes the save-time evidence through
    ``log_after_checkpoint_save`` (populated from the WIDEN ``[mem]``
    log labels in ``--comfy-log-path``) rather than fabricating a
    save-time OS observation from a post-return procfs read.
    """

    run_index: int = 0
    cache_miss_identity: str = ""
    model_name: str = ""
    save_result: WorkflowResult = field(default_factory=WorkflowResult)
    process_memory_before: ProcessMemoryObservation = field(
        default_factory=ProcessMemoryObservation,
    )
    process_memory_after_save: ProcessMemoryObservation = field(
        default_factory=ProcessMemoryObservation,
    )
    process_memory_after_return: ProcessMemoryObservation = field(
        default_factory=ProcessMemoryObservation,
    )
    log_after_checkpoint_save: LogMemoryObservation = field(
        default_factory=LogMemoryObservation,
    )
    log_after_checkpoint_temp_model_release: LogMemoryObservation = field(
        default_factory=LogMemoryObservation,
    )


# Backwards-compatible alias for the previous name used during the
# initial implementation cycle. Kept so external callers (tests, ad-hoc
# operator scripts) that still import the old symbol continue to work.
RepeatedCacheMissRunReport = CheckpointSaveRunReport


def _read_proc_field_kb(path: str, key: str) -> int | None:
    """Read a single ``Key: N kB`` line from a procfs file.

    Returns ``None`` when the file is unreadable or the key is absent.
    Safe to call with arbitrary paths — never raises.
    """
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[1])
                        except ValueError:
                            return None
        return None
    except OSError:
        return None


def read_proc_status_memory(pid: int) -> dict[str, int | None]:
    """Read VmRSS and VmSwap from ``/proc/<pid>/status`` (Linux only).

    Returns a dict with ``rss_kb`` and ``swap_kb`` keys; values are
    ``None`` when the file or field is unavailable. Never raises.
    """
    path = f"/proc/{pid}/status"
    return {
        "rss_kb": _read_proc_field_kb(path, "VmRSS"),
        "swap_kb": _read_proc_field_kb(path, "VmSwap"),
    }


def read_proc_smaps_rollup_memory(pid: int) -> dict[str, int | None]:
    """Read Rss, Swap, and Anonymous from ``/proc/<pid>/smaps_rollup``.

    Returns a dict with ``rss_kb``, ``swap_kb``, and ``anonymous_kb``
    keys; values are ``None`` when the file or field is unavailable.
    Never raises.
    """
    path = f"/proc/{pid}/smaps_rollup"
    return {
        "rss_kb": _read_proc_field_kb(path, "Rss"),
        "swap_kb": _read_proc_field_kb(path, "Swap"),
        "anonymous_kb": _read_proc_field_kb(path, "Anonymous"),
    }


def collect_process_memory_observation(
    pid: int | None,
    label: str,
) -> ProcessMemoryObservation:
    """Snapshot process memory for ``pid`` at the given lifecycle label.

    When ``pid`` is None or the procfs files are unreadable, the
    returned observation has ``available=False`` and a descriptive
    ``error`` field — the harness must record the observation as
    unavailable rather than crashing.

    Prefers ``/proc/<pid>/smaps_rollup`` (includes Anonymous memory).
    Falls back to ``/proc/<pid>/status`` (VmRSS, VmSwap) when smaps
    rollup is unavailable.
    """
    obs = ProcessMemoryObservation(label=label, timestamp=time.time(), pid=pid)
    if pid is None:
        obs.error = "no --comfy-pid supplied"
        return obs

    rollup_path = f"/proc/{pid}/smaps_rollup"
    if os.path.exists(rollup_path):
        rollup = read_proc_smaps_rollup_memory(pid)
        if rollup["rss_kb"] is not None:
            obs.source = "smaps_rollup"
            obs.rss_kb = rollup["rss_kb"]
            obs.swap_kb = rollup["swap_kb"]
            obs.anonymous_kb = rollup["anonymous_kb"]
            obs.available = True
            return obs

    status_path = f"/proc/{pid}/status"
    if os.path.exists(status_path):
        status = read_proc_status_memory(pid)
        if status["rss_kb"] is not None:
            obs.source = "status"
            obs.rss_kb = status["rss_kb"]
            obs.swap_kb = status["swap_kb"]
            obs.available = True
            return obs

    obs.error = f"procfs unreadable for pid {pid}"
    return obs


_LOG_LINE_RE = re.compile(
    r"\[mem\]\s*(?P<label>[a-zA-Z0-9_\-]+):\s*RSS=(?P<rss>\d+)MB"
    r"(?:\s+VRAM=(?P<alloc>\d+)MB\(reserved=(?P<reserved>\d+)MB\))?",
)


def parse_widen_memory_log(
    log_text: str,
    labels: tuple[str, ...],
) -> dict[str, LogMemoryObservation]:
    """Parse the last occurrence of each WIDEN ``[mem]`` label.

    Returns a mapping from each requested label to its
    ``LogMemoryObservation``. Labels that are absent from the log text
    receive an observation with ``available=False`` and an explanatory
    ``error`` so the report can record the missing label rather than
    crashing.

    Matches the log line format emitted by ``nodes/exit._log_memory``:
    ``[mem] <label>: RSS=<n>MB`` optionally followed by
    ``VRAM=<n>MB(reserved=<n>MB)``.
    """
    found: dict[str, LogMemoryObservation] = {}
    for line in log_text.splitlines():
        match = _LOG_LINE_RE.search(line)
        if not match:
            continue
        label = match.group("label")
        if label not in labels:
            continue
        rss_mb = int(match.group("rss"))
        alloc = match.group("alloc")
        reserved = match.group("reserved")
        found[label] = LogMemoryObservation(
            label=label,
            available=True,
            rss_mb=rss_mb,
            vram_alloc_mb=int(alloc) if alloc is not None else None,
            vram_reserved_mb=int(reserved) if reserved is not None else None,
            raw_line=line.rstrip(),
        )

    result: dict[str, LogMemoryObservation] = {}
    for label in labels:
        if label in found:
            result[label] = found[label]
        else:
            result[label] = LogMemoryObservation(
                label=label,
                available=False,
                error=f"label {label!r} not found in supplied log text",
            )
    return result


def read_log_segment(log_path: str, start_offset: int) -> tuple[str, int, str]:
    """Read text appended to ``log_path`` since ``start_offset`` bytes.

    Returns ``(text, end_offset, error)``. When the file is unreadable
    the returned ``text`` is empty and ``error`` is populated; this is
    intentional so the report can mark log observations as unavailable
    instead of crashing.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(start_offset)
            data = f.read()
            end = f.tell()
        return data.decode("utf-8", errors="replace"), end, ""
    except OSError as exc:
        return "", start_offset, f"log read error: {exc}"


def current_log_offset(log_path: str | None) -> int:
    """Return the current end-of-file offset for ``log_path``.

    Returns 0 when ``log_path`` is None or unreadable; safe for callers
    that want a starting marker for later differential reads.
    """
    if not log_path:
        return 0
    try:
        return os.path.getsize(log_path)
    except OSError:
        return 0


_RUN_MEMORY_LABELS: tuple[str, ...] = (
    "after-checkpoint-save",
    "after-checkpoint-temp-model-release",
)


_AFTER_SAVE_UNAVAILABLE_REASON = (
    "OS-level save-time observation cannot be sampled from an external "
    "harness: submit_workflow blocks on /history and only returns after "
    "WIDENExit returned, so any procfs read here would be a post-return "
    "sample. See log_after_checkpoint_save for save-time evidence."
)


def unavailable_save_time_observation(
    pid: int | None = None,
) -> ProcessMemoryObservation:
    """Return an explicit unavailable save-time OS memory observation.

    The harness never samples procfs at the actual save-time lifecycle
    point because ``submit_workflow`` only returns after ComfyUI's
    ``/history`` endpoint reports completion (i.e. after WIDENExit
    returned). To prevent reports from falsely claiming save-time OS
    memory was observed, ``process_memory_after_save`` is always
    populated with this unavailable record, regardless of whether
    ``--comfy-pid`` is supplied.
    """
    return ProcessMemoryObservation(
        label="after-save",
        available=False,
        pid=pid,
        error=_AFTER_SAVE_UNAVAILABLE_REASON,
        timestamp=time.time(),
    )


def begin_save_run_memory_observation(
    comfy_pid: int | None,
    comfy_log_path: str | None,
) -> tuple[ProcessMemoryObservation, int]:
    """Snapshot before-run process memory and the starting log offset.

    Returns ``(before_run_observation, log_offset)``. Pair with
    :func:`finalize_save_run_memory_observation` to record the
    after-return and log observations after the workflow completes.
    Both helpers are safe to call when no memory inputs were supplied:
    the observations will be marked unavailable rather than crashing.
    """
    offset = current_log_offset(comfy_log_path)
    before = collect_process_memory_observation(comfy_pid, "before-run")
    return before, offset


def finalize_save_run_memory_observation(
    run: CheckpointSaveRunReport,
    comfy_pid: int | None,
    comfy_log_path: str | None,
    before_offset: int,
) -> None:
    """Fill the after-save/after-return and log observations on ``run``.

    Call after the save workflow's ``submit_workflow`` returns.
    ``process_memory_after_save`` is always recorded as unavailable
    because ``submit_workflow`` only returns once ComfyUI's
    ``/history`` reports completion (i.e. after WIDENExit returned);
    see :class:`CheckpointSaveRunReport`. When ``comfy_log_path`` is
    supplied, the WIDEN ``[mem]`` labels appearing in the log segment
    since ``before_offset`` are parsed for save-time evidence; missing
    labels are recorded as unavailable rather than treated as a crash.
    """
    run.process_memory_after_save = unavailable_save_time_observation(comfy_pid)
    run.process_memory_after_return = collect_process_memory_observation(
        comfy_pid,
        "after-return",
    )

    if comfy_log_path:
        segment, _end_offset, log_err = read_log_segment(
            comfy_log_path,
            before_offset,
        )
        if log_err:
            run.log_after_checkpoint_save = LogMemoryObservation(
                label="after-checkpoint-save",
                available=False,
                error=log_err,
            )
            run.log_after_checkpoint_temp_model_release = LogMemoryObservation(
                label="after-checkpoint-temp-model-release",
                available=False,
                error=log_err,
            )
        else:
            parsed = parse_widen_memory_log(segment, _RUN_MEMORY_LABELS)
            run.log_after_checkpoint_save = parsed["after-checkpoint-save"]
            run.log_after_checkpoint_temp_model_release = parsed[
                "after-checkpoint-temp-model-release"
            ]
    else:
        unavailable_msg = "no --comfy-log-path supplied"
        run.log_after_checkpoint_save = LogMemoryObservation(
            label="after-checkpoint-save",
            available=False,
            error=unavailable_msg,
        )
        run.log_after_checkpoint_temp_model_release = LogMemoryObservation(
            label="after-checkpoint-temp-model-release",
            available=False,
            error=unavailable_msg,
        )


def analyze_memory_accumulation(
    runs: list[CheckpointSaveRunReport],
    threshold_mb: int,
) -> dict[str, object]:
    """Decide whether repeated cache-miss runs show accumulating memory.

    Compares the first and last available post-return process-memory
    observations across ``runs`` and reports whether the growth exceeds
    ``threshold_mb`` (RSS + Swap, in megabytes). Returns a dict with:

    * ``result``: one of ``ok``, ``failed``, or ``unavailable``.
    * ``threshold_mb``: the threshold used, echoed for auditability.
    * ``first_run_index`` / ``last_run_index``: indices of the runs
      whose observations were compared (or ``None`` if unavailable).
    * ``first_rss_swap_kb`` / ``last_rss_swap_kb``: combined RSS+Swap
      for the two compared observations (or ``None``).
    * ``delta_kb``: ``last - first`` when both available.
    * ``detail``: human-readable description for the report.
    """
    threshold_kb = max(0, int(threshold_mb)) * 1024

    available_runs: list[tuple[int, int, int]] = []
    for run in runs:
        obs = run.process_memory_after_return
        if obs.available and obs.rss_kb is not None:
            swap = obs.swap_kb if obs.swap_kb is not None else 0
            available_runs.append((run.run_index, obs.rss_kb, swap))

    if len(available_runs) < 2:
        return {
            "result": "unavailable",
            "threshold_mb": threshold_mb,
            "first_run_index": None,
            "last_run_index": None,
            "first_rss_swap_kb": None,
            "last_rss_swap_kb": None,
            "delta_kb": None,
            "detail": (
                "need at least 2 post-return observations with --comfy-pid; "
                f"have {len(available_runs)}"
            ),
        }

    first_idx, first_rss, first_swap = available_runs[0]
    last_idx, last_rss, last_swap = available_runs[-1]
    first_total = first_rss + first_swap
    last_total = last_rss + last_swap
    delta_kb = last_total - first_total

    if delta_kb > threshold_kb:
        return {
            "result": "failed",
            "threshold_mb": threshold_mb,
            "first_run_index": first_idx,
            "last_run_index": last_idx,
            "first_rss_swap_kb": first_total,
            "last_rss_swap_kb": last_total,
            "delta_kb": delta_kb,
            "detail": (
                f"post-return RSS+Swap grew {delta_kb / 1024:.1f}MB across "
                f"runs {first_idx}->{last_idx}, exceeding threshold of "
                f"{threshold_mb}MB; temporary save-time payload appears to "
                f"accumulate"
            ),
        }

    return {
        "result": "ok",
        "threshold_mb": threshold_mb,
        "first_run_index": first_idx,
        "last_run_index": last_idx,
        "first_rss_swap_kb": first_total,
        "last_rss_swap_kb": last_total,
        "delta_kb": delta_kb,
        "detail": (
            f"post-return RSS+Swap grew {delta_kb / 1024:.1f}MB across "
            f"runs {first_idx}->{last_idx}, within threshold of "
            f"{threshold_mb}MB"
        ),
    }


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
    # Process-memory observation inputs (optional, operator-supplied).
    comfy_pid: int | None = None
    comfy_log_path: str = ""
    repeat_cache_miss_runs_requested: int = 0
    memory_accumulation_threshold_mb: int = DEFAULT_ACCUMULATION_THRESHOLD_MB
    primary_checkpoint_save_run: CheckpointSaveRunReport = field(
        default_factory=CheckpointSaveRunReport,
    )
    repeat_cache_miss_run_reports: list[CheckpointSaveRunReport] = field(
        default_factory=list,
    )
    repeated_cache_miss_memory_result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


_REQUIRED_REPORT_FIELDS = frozenset(
    {
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
        "comfy_pid",
        "comfy_log_path",
        "repeat_cache_miss_runs_requested",
        "memory_accumulation_threshold_mb",
        "primary_checkpoint_save_run",
        "repeat_cache_miss_run_reports",
        "repeated_cache_miss_memory_result",
    }
)


PRIMARY_CHECKPOINT_SAVE_IDENTITY = "primary-checkpoint-save"
"""``cache_miss_identity`` recorded for the primary ``run_validation`` save.

Distinguishes the primary checkpoint save's per-run memory observations
from the repeated cache-miss series. The primary save uses the fixed
``ecaj_checkpoint_validation_save`` model name (so the cache-reuse step
can re-submit the same model_name); the identity recorded in the per-run
report makes the role explicit when readers inspect the JSON.
"""


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

    lines.extend(
        [
            "--- Terminal Save ---",
            f"  Accepted:     {report.terminal_save_result.accepted}",
            f"  Prompt ID:    {report.terminal_save_result.prompt_id}",
            f"  Error:        {report.terminal_save_result.error or 'none'}",
        ]
    )
    if report.terminal_save_workflow_shape:
        shape_json = json.dumps(report.terminal_save_workflow_shape)
        lines.append(f"  Workflow Shape: {shape_json}")
    lines.append("")

    lines.extend(
        [
            "--- Checkpoint Save ---",
            f"  Accepted:     {report.checkpoint_save_result.accepted}",
            f"  Prompt ID:    {report.checkpoint_save_result.prompt_id}",
            f"  Artifact:     {report.saved_artifact_path}",
            f"  Classification: {report.saved_artifact_classification}",
            f"  Error:        {report.checkpoint_save_result.error or 'none'}",
        ]
    )
    if report.checkpoint_save_workflow_shape:
        shape_json = json.dumps(report.checkpoint_save_workflow_shape)
        lines.append(f"  Workflow Shape: {shape_json}")
    lines.append("")

    lines.extend(
        [
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
        ]
    )

    if report.failure_categories:
        lines.append("--- Failure Categories ---")
        for cat, desc in report.failure_categories.items():
            lines.append(f"  {cat}: {desc}")
        lines.append("")

    lines.append("--- Process-Memory Observation Inputs ---")
    pid_disp = report.comfy_pid if report.comfy_pid is not None else "not supplied"
    lines.append(f"  --comfy-pid:        {pid_disp}")
    lines.append(f"  --comfy-log-path:   {report.comfy_log_path or 'not supplied'}")
    lines.append(f"  --repeat-cache-miss-runs: {report.repeat_cache_miss_runs_requested}")
    lines.append(
        f"  --memory-accumulation-threshold-mb: {report.memory_accumulation_threshold_mb}"
    )
    lines.append("")

    primary_run = report.primary_checkpoint_save_run
    if primary_run.cache_miss_identity or primary_run.model_name:
        lines.append("--- Primary Checkpoint Save Memory ---")
        lines.append(
            f"  identity={primary_run.cache_miss_identity} "
            f"model_name={primary_run.model_name} "
            f"accepted={primary_run.save_result.accepted}"
        )
        for tag, obs in (
            ("before", primary_run.process_memory_before),
            ("after-save", primary_run.process_memory_after_save),
            ("after-return", primary_run.process_memory_after_return),
        ):
            if obs.available:
                extras = [f"RSS={obs.rss_kb}kB"]
                if obs.swap_kb is not None:
                    extras.append(f"Swap={obs.swap_kb}kB")
                if obs.anonymous_kb is not None:
                    extras.append(f"Anon={obs.anonymous_kb}kB")
                extras.append(f"source={obs.source}")
                lines.append(f"    {tag}: " + " ".join(extras))
            else:
                detail = obs.error or "unavailable"
                lines.append(f"    {tag}: unavailable ({detail})")
        for log_tag, log_obs in (
            ("log after-checkpoint-save", primary_run.log_after_checkpoint_save),
            (
                "log after-checkpoint-temp-model-release",
                primary_run.log_after_checkpoint_temp_model_release,
            ),
        ):
            if log_obs.available:
                parts = [f"RSS={log_obs.rss_mb}MB"]
                if log_obs.vram_alloc_mb is not None:
                    parts.append(f"VRAM={log_obs.vram_alloc_mb}MB")
                lines.append(f"    {log_tag}: " + " ".join(parts))
            else:
                detail = log_obs.error or "unavailable"
                lines.append(f"    {log_tag}: unavailable ({detail})")
        lines.append("")

    if report.repeat_cache_miss_run_reports:
        lines.append("--- Repeated Cache-Miss Runs ---")
        for run in report.repeat_cache_miss_run_reports:
            lines.append(
                f"  [run {run.run_index}] model_name={run.model_name} "
                f"identity={run.cache_miss_identity} "
                f"accepted={run.save_result.accepted}"
            )
            for tag, obs in (
                ("before", run.process_memory_before),
                ("after-save", run.process_memory_after_save),
                ("after-return", run.process_memory_after_return),
            ):
                if obs.available:
                    extras = [f"RSS={obs.rss_kb}kB"]
                    if obs.swap_kb is not None:
                        extras.append(f"Swap={obs.swap_kb}kB")
                    if obs.anonymous_kb is not None:
                        extras.append(f"Anon={obs.anonymous_kb}kB")
                    extras.append(f"source={obs.source}")
                    lines.append(f"    {tag}: " + " ".join(extras))
                else:
                    detail = obs.error or "unavailable"
                    lines.append(f"    {tag}: unavailable ({detail})")
            for log_tag, log_obs in (
                ("log after-checkpoint-save", run.log_after_checkpoint_save),
                (
                    "log after-checkpoint-temp-model-release",
                    run.log_after_checkpoint_temp_model_release,
                ),
            ):
                if log_obs.available:
                    parts = [f"RSS={log_obs.rss_mb}MB"]
                    if log_obs.vram_alloc_mb is not None:
                        parts.append(f"VRAM={log_obs.vram_alloc_mb}MB")
                    lines.append(f"    {log_tag}: " + " ".join(parts))
                else:
                    detail = log_obs.error or "unavailable"
                    lines.append(f"    {log_tag}: unavailable ({detail})")
        lines.append("")

    if report.repeated_cache_miss_memory_result:
        result = report.repeated_cache_miss_memory_result
        lines.append("--- Repeated Cache-Miss Memory Result ---")
        lines.append(f"  result:        {result.get('result', 'unknown')}")
        threshold_disp = result.get(
            "threshold_mb",
            report.memory_accumulation_threshold_mb,
        )
        lines.append(f"  threshold_mb:  {threshold_disp}")
        if result.get("delta_kb") is not None:
            lines.append(f"  delta_kb:      {result['delta_kb']}")
        lines.append(f"  detail:        {result.get('detail', '')}")
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
    lines.extend(
        [
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
    base_url: str,
    prompt_id: str,
    *,
    timeout: float = 30.0,
    max_polls: int = 30,
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
    base_url: str,
    first_prompt_id: str,
    reuse_prompt_id: str,
    *,
    first_elapsed: float = 0.0,
    reuse_elapsed: float = 0.0,
) -> tuple[bool, str]:
    """Check whether the reuse prompt reused cached node outputs.

    Detection methods (tried in order):

    1. **Output-based** — if WIDENExit produced UI outputs in the first
       prompt but not the reuse prompt, the reuse was served from cache.
    2. **execution_cached message** — if ComfyUI's status messages
       report the exit node as ``execution_cached``, cache was reused.
    3. **Timing-based** — WIDENExit returns ``MODEL`` (not a UI type),
       so ComfyUI may not record it in the ``outputs`` dict for either
       prompt.  When the exit node is absent from both outputs *and*
       ComfyUI didn't report ``execution_cached`` (because WIDEN uses
       an application-level artifact cache inside ``execute()`` rather
       than ComfyUI's ``IS_CHANGED`` cache), timing comparison serves
       as concrete evidence: a ≥ 2× speedup on the reuse prompt
       indicates the expensive merge/save was skipped and the artifact
       cache was hit.

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

        # Method 1: output-based detection
        if first_had_exit_output and not reuse_had_exit_output:
            return True, (
                f"WIDENExit node {exit_node_id} executed in first prompt "
                f"({first_prompt_id}) but was cached in reuse prompt "
                f"({reuse_prompt_id})"
            )

        # Method 2: execution_cached message
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

        # Method 3: timing-based detection for application-level cache.
        # WIDENExit returns MODEL (not a UI type), so ComfyUI does not
        # record it in the outputs dict.  When both prompts lack exit
        # node outputs, we compare wall-clock times: a ≥ 2× speedup
        # indicates the WIDEN artifact cache hit inside execute().
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
    CLIP, and VAE through WIDEN Entry/Exit with save_model=True and
    enable_cache=False.

    ``enable_cache`` is explicitly disabled so the save always performs the
    full merge-and-write, producing a fresh artifact.  The companion
    ``build_cache_reuse_workflow`` re-runs with ``enable_cache=True`` to
    verify that the application-level artifact cache returns the saved
    checkpoint without re-computing the merge.

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
                "enable_cache": False,
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


def build_repeat_cache_miss_workflow(
    source_model: str,
    model_name: str,
) -> dict:
    """Build a checkpoint-save workflow with a caller-supplied model_name.

    Used by the repeated cache-miss harness to give each iteration a
    distinct artifact identity so the requested repetitions cannot
    collapse into artifact-cache hits. ``enable_cache=False`` is
    preserved from the standard checkpoint-save workflow.
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
                "enable_cache": False,
                "model_name": model_name,
            },
        },
    }


def make_repeat_cache_miss_identity(
    run_index: int,
    run_token: str | None = None,
) -> tuple[str, str]:
    """Return ``(cache_miss_identity, model_name)`` for a repeat run.

    ``run_token`` defaults to a monotonic time-derived suffix so two
    repeated runs in one process always have distinct identities even
    when the operator does not supply one explicitly.
    """
    if not run_token:
        run_token = f"{time.time_ns():x}"
    cache_miss_identity = f"repeat-{run_index:03d}-{run_token}"
    model_name = f"ecaj_checkpoint_validation_save_{cache_miss_identity}"
    return cache_miss_identity, model_name


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


def _extract_node_errors_from_history(history: dict) -> str:
    """Extract node error details from a prompt history entry.

    ComfyUI records per-node execution status in ``status.messages``.
    If a node raised an exception during execution, the history entry
    also contains ``outputs`` that may be empty or partial plus an
    ``status.status_str`` of ``"error"``.
    """
    status = history.get("status", {})
    status_str = status.get("status_str", "")
    if status_str == "error":
        messages = status.get("messages", [])
        # Look for execution_error messages
        for msg in messages:
            if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                err_data = msg[1] if isinstance(msg[1], dict) else {}
                node_type = err_data.get("node_type", "unknown")
                exception_message = err_data.get("exception_message", "unknown error")
                return f"node_error({node_type}): {exception_message}"
        return f"execution_error: status_str={status_str}"
    return ""


def submit_workflow(
    base_url: str,
    workflow: dict,
    name: str,
) -> WorkflowResult:
    """Submit a workflow prompt, wait for completion, and classify the result.

    After /prompt returns a prompt_id, polls /history until execution
    completes.  Runtime node errors (save failures, loader errors,
    KSampler crashes) are detected from the history entry and reported
    as failures — not silently marked accepted.

    Captures wall-clock elapsed time (prompt submission through history
    completion) in ``elapsed_seconds`` for timing-based analysis.
    """
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

        # Poll /history for actual execution completion
        history = query_prompt_history(base_url, prompt_id)
        if not history:
            result.accepted = False
            result.error = "execution_timeout: prompt accepted but never completed in /history"
            return result

        # Check for runtime node errors in the completed history
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
    """Extract the failing node class type from a ``node_error(NodeType): ...`` string.

    Returns the node type (e.g. ``"KSampler"``, ``"CheckpointLoaderSimple"``)
    or ``""`` if the error string does not match the ``node_error(...)`` pattern.
    """
    if error.startswith("node_error("):
        paren_end = error.find(")")
        if paren_end > len("node_error("):
            return error[len("node_error(") : paren_end]
    return ""


def classify_failure(result: WorkflowResult) -> str:
    """Classify a workflow result into a failure category.

    Memory/VRAM/OOM errors are checked first because they represent resource
    exhaustion — an orthogonal failure dimension that takes priority over
    workflow-stage keywords.  Without this priority, an error like
    ``node_error(KSampler): OOM: cannot allocate`` would match "ksampler"
    and return "downstream", masking the real memory_mode failure.
    """
    if result.accepted:
        return "none"
    error = result.error.lower()
    # Memory/VRAM/OOM — check FIRST, before workflow-stage keywords.
    # A KSampler OOM or a save CUDA-out-of-memory is fundamentally a
    # memory issue, not a downstream or save issue.
    if "memory" in error or "vram" in error or "oom" in error:
        return "memory_mode"
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
    if "url_error" in error or "connection" in error:
        return "connection"
    return "unknown"


# ---------------------------------------------------------------------------
# Validation runner — only reachable after all guards pass
# ---------------------------------------------------------------------------


def run_repeated_cache_miss_runs(
    comfy_api_url: str,
    source_model: str,
    repeat_count: int,
    *,
    comfy_pid: int | None = None,
    comfy_log_path: str | None = None,
) -> list[CheckpointSaveRunReport]:
    """Run ``repeat_count`` cache-miss saves with distinct identities.

    Each iteration:

    1. Snapshots process memory ``before`` the run (when ``comfy_pid`` is
       supplied) and captures the current log offset.
    2. Submits a checkpoint-save workflow whose ``model_name`` is unique
       so the artifact cache cannot serve it. ``submit_workflow`` blocks
       on ``/history`` until ComfyUI reports the prompt completed, so
       control returns to this loop only after WIDENExit returned.
    3. Records ``process_memory_after_save`` as unavailable: the harness
       cannot sample procfs at the actual save-time lifecycle point
       from outside ComfyUI. Save-time evidence is taken from the WIDEN
       ``[mem]`` log labels instead (step 5).
    4. Snapshots process memory once ``after_return`` (used by
       ``analyze_memory_accumulation``).
    5. Parses the WIDEN ``[mem]`` log labels that appeared since the
       starting offset (when ``comfy_log_path`` is supplied). The
       ``after-checkpoint-save`` label captures save-time memory from
       inside the WIDENExit process; the ``after-checkpoint-temp-model
       -release`` label captures memory after the temp save payload was
       released, both before WIDENExit returned.

    Inputs that are unavailable are recorded as unavailable observations
    rather than crashing the harness.
    """
    results: list[CheckpointSaveRunReport] = []
    for run_index in range(repeat_count):
        identity, model_name = make_repeat_cache_miss_identity(run_index)
        run = CheckpointSaveRunReport(
            run_index=run_index,
            cache_miss_identity=identity,
            model_name=model_name,
        )
        run.process_memory_before, before_offset = begin_save_run_memory_observation(
            comfy_pid, comfy_log_path
        )
        workflow = build_repeat_cache_miss_workflow(source_model, model_name)
        run.save_result = submit_workflow(
            comfy_api_url,
            workflow,
            f"repeat_cache_miss_{run_index:03d}",
        )
        finalize_save_run_memory_observation(
            run,
            comfy_pid,
            comfy_log_path,
            before_offset,
        )
        results.append(run)
    return results


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
    comfy_pid: int | None = None,
    comfy_log_path: str | None = None,
    repeat_cache_miss_runs: int = 0,
    memory_accumulation_threshold_mb: int = DEFAULT_ACCUMULATION_THRESHOLD_MB,
) -> CheckpointValidationReport:
    """Run live ComfyUI checkpoint-save validation via the API.

    This function is ONLY called after all guards pass.
    It submits workflow prompts to the ComfyUI API and classifies results.

    Optional ``comfy_pid``, ``comfy_log_path``, ``repeat_cache_miss_runs``,
    and ``memory_accumulation_threshold_mb`` enable per-run process-memory
    observation and repeated cache-miss accumulation analysis. When the
    operator does not supply them, the corresponding report fields are
    recorded as unavailable rather than fabricated.
    """
    report = CheckpointValidationReport(
        comfy_api_url=comfy_api_url,
        source_model=source_model,
        comfy_pid=comfy_pid,
        comfy_log_path=comfy_log_path or "",
        repeat_cache_miss_runs_requested=max(0, int(repeat_cache_miss_runs)),
        memory_accumulation_threshold_mb=max(
            0,
            int(memory_accumulation_threshold_mb),
        ),
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
            comfy_api_url,
            terminal_wf,
            "terminal_exit_save",
        )
        report.failure_categories["scheduling"] = classify_failure(
            report.terminal_save_result,
        )

        # --- Submit checkpoint save_model workflow ---
        save_wf = build_checkpoint_save_workflow(source_model)
        report.checkpoint_save_workflow_shape = save_wf

        # Capture per-run process/log memory observations around the
        # primary checkpoint save. The record is always populated so the
        # JSON report has a stable shape; when the operator did not
        # supply ``--comfy-pid``/``--comfy-log-path`` the observations
        # are recorded as unavailable. This satisfies
        # @live-comfy-saved-output-validation
        # ac-report-records-process-memory-points for the normal
        # (non-repeat) save path: whenever the operator supplies
        # process-memory inputs, the primary save records before-run
        # and after-return observations and any save-time WIDEN log
        # labels — without requiring a non-zero --repeat-cache-miss-runs.
        primary_run = CheckpointSaveRunReport(
            run_index=0,
            cache_miss_identity=PRIMARY_CHECKPOINT_SAVE_IDENTITY,
            model_name="ecaj_checkpoint_validation_save",
        )
        primary_run.process_memory_before, primary_before_offset = (
            begin_save_run_memory_observation(comfy_pid, comfy_log_path)
        )
        report.checkpoint_save_result = submit_workflow(
            comfy_api_url,
            save_wf,
            "checkpoint_save",
        )
        primary_run.save_result = report.checkpoint_save_result
        finalize_save_run_memory_observation(
            primary_run,
            comfy_pid,
            comfy_log_path,
            primary_before_offset,
        )
        report.primary_checkpoint_save_run = primary_run
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
        # Only run loader/downstream if the checkpoint save succeeded.
        # If save was rejected, there is no saved artifact to validate;
        # falling back to the source model would give false confidence.
        if saved_artifact_filename:
            downstream_wf = build_downstream_workflow(
                saved_artifact_filename,
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
            # The downstream workflow contains both CheckpointLoaderSimple
            # (node "1") and KSampler (node "5").  A single submission
            # exercises both, but we need *distinct* outcomes: the loader
            # may succeed while KSampler fails (or vice-versa).
            #
            # ComfyUI executes nodes in dependency order.  When a node
            # raises an error, the error message encodes the failing
            # node type as ``node_error(NodeType): ...``.  If the failing
            # node is CheckpointLoaderSimple, the loader itself broke
            # and downstream never ran.  If the failing node is anything
            # else (KSampler, VAEDecode, …), the loader must have
            # succeeded — so the loader result is "accepted" and only
            # the downstream result carries the failure.
            failing_node = _extract_failing_node_type(
                downstream_full_result.error,
            )
            if downstream_full_result.accepted:
                # Everything succeeded — both loader and downstream are OK.
                report.checkpoint_loader_result = WorkflowResult(
                    name="checkpoint_loader",
                    accepted=True,
                    prompt_id=downstream_full_result.prompt_id,
                )
                report.downstream_result = downstream_full_result
            elif failing_node == "CheckpointLoaderSimple":
                # The loader node itself failed — downstream never ran.
                report.checkpoint_loader_result = WorkflowResult(
                    name="checkpoint_loader",
                    accepted=False,
                    prompt_id=downstream_full_result.prompt_id,
                    error=downstream_full_result.error,
                )
                report.downstream_result = WorkflowResult(
                    name="downstream_ksampler",
                    accepted=False,
                    prompt_id=downstream_full_result.prompt_id,
                    error="skipped: checkpoint loader failed",
                )
            else:
                # A non-loader node failed (e.g. KSampler, VAEDecode).
                # The loader succeeded; only downstream failed.
                report.checkpoint_loader_result = WorkflowResult(
                    name="checkpoint_loader",
                    accepted=True,
                    prompt_id=downstream_full_result.prompt_id,
                )
                report.downstream_result = downstream_full_result
            report.failure_categories["loader"] = classify_failure(
                report.checkpoint_loader_result,
            )
            report.failure_categories["downstream"] = classify_failure(
                report.downstream_result,
            )
        else:
            # Save failed — skip loader/downstream; record the skip reason.
            skip_reason = "skipped: checkpoint save failed, no artifact to validate"
            report.checkpoint_loader_result = WorkflowResult(
                name="checkpoint_loader",
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

        # --- Submit cache reuse workflow and verify reuse ---
        cache_wf = build_cache_reuse_workflow(source_model)
        report.cache_reuse_result = submit_workflow(
            comfy_api_url,
            cache_wf,
            "cache_reuse",
        )
        report.failure_categories["cache"] = classify_failure(
            report.cache_reuse_result,
        )

        # Check cache reuse by comparing execution history
        if report.checkpoint_save_result.accepted and report.cache_reuse_result.accepted:
            cache_reused, detail = check_cache_reuse(
                comfy_api_url,
                report.checkpoint_save_result.prompt_id,
                report.cache_reuse_result.prompt_id,
                first_elapsed=report.checkpoint_save_result.elapsed_seconds,
                reuse_elapsed=report.cache_reuse_result.elapsed_seconds,
            )
            report.cache_reuse_detail = detail
            if not cache_reused:
                report.failure_categories["cache"] = "cache_not_reused"
        elif report.cache_reuse_result.accepted:
            report.cache_reuse_detail = "first save was rejected; cannot verify reuse"
        else:
            report.cache_reuse_detail = "cache reuse prompt was rejected"

        # --- Repeated cache-miss runs (optional) ---
        if report.repeat_cache_miss_runs_requested > 0:
            report.repeat_cache_miss_run_reports = run_repeated_cache_miss_runs(
                comfy_api_url=comfy_api_url,
                source_model=source_model,
                repeat_count=report.repeat_cache_miss_runs_requested,
                comfy_pid=comfy_pid,
                comfy_log_path=comfy_log_path,
            )
            report.repeated_cache_miss_memory_result = analyze_memory_accumulation(
                report.repeat_cache_miss_run_reports,
                report.memory_accumulation_threshold_mb,
            )
            if report.repeated_cache_miss_memory_result.get("result") == "failed":
                report.failure_categories["repeated_cache_miss_memory"] = (
                    report.repeated_cache_miss_memory_result.get(
                        "detail",
                        "accumulation exceeded threshold",
                    )
                )
            else:
                report.failure_categories["repeated_cache_miss_memory"] = (
                    report.repeated_cache_miss_memory_result.get("result", "ok")
                )
        else:
            report.repeated_cache_miss_memory_result = {
                "result": "not-requested",
                "threshold_mb": report.memory_accumulation_threshold_mb,
                "detail": "no --repeat-cache-miss-runs requested",
                "first_run_index": None,
                "last_run_index": None,
                "first_rss_swap_kb": None,
                "last_rss_swap_kb": None,
                "delta_kb": None,
            }

        # --- Memory mode failure check ---
        # Scan all per-workflow failure categories for memory_mode
        # failures. If any workflow was classified as a memory_mode
        # failure, propagate that to the top-level category so the
        # report correctly surfaces OOM/VRAM issues.
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
        comfy_pid=args.comfy_pid,
        comfy_log_path=args.comfy_log_path,
        repeat_cache_miss_runs=args.repeat_cache_miss_runs,
        memory_accumulation_threshold_mb=args.memory_accumulation_threshold_mb,
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
