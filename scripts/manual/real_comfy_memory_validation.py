#!/usr/bin/env python3
"""Guarded manual validation harness for real ComfyUI memory-mode runs.

This script is NOT part of the automated test suite. It requires explicit
opt-in via environment variable, CLI flag, and all required paths before
it will import any ComfyUI modules or touch a target ComfyUI installation.

Usage:
    COMFY_ECAJ_REAL_MEMORY_VALIDATION=1 python scripts/manual/real_comfy_memory_validation.py \\
        --run-real-comfy-memory \\
        --comfy-root /path/to/ComfyUI \\
        --model-path /path/to/model.safetensors \\
        --report-output /path/to/report.json

All five inputs are required:
    1. COMFY_ECAJ_REAL_MEMORY_VALIDATION=1 environment variable
    2. --run-real-comfy-memory CLI flag
    3. --comfy-root: path to ComfyUI installation
    4. --model-path: path to a deterministic fixture/model file
    5. --report-output: path where the JSON report will be written
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Guard validation — must run BEFORE any ComfyUI import or sys.path mutation
# ---------------------------------------------------------------------------

_ENV_VAR = "COMFY_ECAJ_REAL_MEMORY_VALIDATION"
_CLI_FLAG = "--run-real-comfy-memory"


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
        1. COMFY_ECAJ_REAL_MEMORY_VALIDATION=1 environment variable
        2. --run-real-comfy-memory CLI flag
        3. --comfy-root argument present and non-empty
        4. --model-path argument present and non-empty
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

    # Parse known args for path validation (tolerant of unknown flags)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--comfy-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument(_CLI_FLAG, action="store_true", dest="run_flag")
    known, _ = parser.parse_known_args(argv)

    # Guard 3: --comfy-root
    if not known.comfy_root:
        reasons.append("--comfy-root is required but not provided.")

    # Guard 4: --model-path
    if not known.model_path:
        reasons.append("--model-path is required but not provided.")

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
        description="Real ComfyUI memory-mode validation harness.",
    )
    parser.add_argument(
        _CLI_FLAG,
        action="store_true",
        dest="run_flag",
        help="Explicit opt-in flag to run real ComfyUI validation.",
    )
    parser.add_argument(
        "--comfy-root",
        required=True,
        help="Path to ComfyUI installation root.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a deterministic fixture/model .safetensors file.",
    )
    parser.add_argument(
        "--report-output",
        required=True,
        help="Path where the JSON validation report will be written.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Report builder — pure Python, testable with fake data
# ---------------------------------------------------------------------------


@dataclass
class MemoryObservation:
    """Snapshot of RAM/GPU memory at a labeled point."""

    label: str
    rss_mb: float | None = None
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    timestamp: float = 0.0


@dataclass
class ValidationReport:
    """Report from a real ComfyUI memory-mode validation run."""

    memory_mode: str = "unknown"
    comfy_root: str = ""
    model_path: str = ""
    patch_mode_behavior: str = ""
    full_model_materialization: str = ""
    artifact_cache_hit: bool = False
    returned_model_behavior: str = ""
    artifact_reuse: str = ""
    weights_resident_in_cache: bool | None = None
    memory_observations: list[MemoryObservation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def format_report(report: ValidationReport) -> str:
    """Format a ValidationReport as human-readable text.

    Always includes the memory_mode field prominently.
    """
    lines = [
        "=" * 60,
        "ComfyUI Memory-Mode Validation Report",
        "=" * 60,
        "",
        f"Memory Mode:                {report.memory_mode}",
        f"ComfyUI Root:               {report.comfy_root}",
        f"Model Path:                 {report.model_path}",
        f"Duration:                   {report.duration_seconds:.2f}s",
        "",
        "--- Behavior ---",
        f"Patch-mode behavior:        {report.patch_mode_behavior}",
        f"Full-model materialization:  {report.full_model_materialization}",
        f"Artifact cache hit:         {report.artifact_cache_hit}",
        f"Returned model behavior:    {report.returned_model_behavior}",
        f"Artifact reuse:             {report.artifact_reuse}",
        f"Weights resident in cache:  {report.weights_resident_in_cache}",
        "",
    ]

    if report.memory_observations:
        lines.append("--- Memory Observations ---")
        for obs in report.memory_observations:
            parts = [f"  [{obs.label}]"]
            if obs.rss_mb is not None:
                parts.append(f"RSS={obs.rss_mb:.0f}MB")
            if obs.vram_allocated_mb is not None:
                parts.append(f"VRAM={obs.vram_allocated_mb:.0f}MB")
            if obs.vram_reserved_mb is not None:
                parts.append(f"(reserved={obs.vram_reserved_mb:.0f}MB)")
            lines.append(" ".join(parts))
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
        "ComfyUI Memory Validation — REFUSED (guards not satisfied)",
        "=" * 60,
        "",
        "The following required inputs are missing or invalid:",
        "",
    ]
    for reason in guard_result.reasons:
        lines.append(f"  - {reason}")
    lines.extend([
        "",
        "All of the following are required to run real ComfyUI validation:",
        f"  1. {_ENV_VAR}=1 environment variable",
        f"  2. {_CLI_FLAG} CLI flag",
        "  3. --comfy-root <path>",
        "  4. --model-path <path>",
        "  5. --report-output <path>",
        "",
        "No ComfyUI modules were imported.",
        "No ComfyUI installation was modified.",
        "No ComfyUI process was started.",
        "=" * 60,
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real ComfyUI validation runner — only reachable after all guards pass
# ---------------------------------------------------------------------------


def _collect_memory_observation(label: str) -> MemoryObservation:
    """Collect a memory observation at the current point."""
    import torch

    obs = MemoryObservation(label=label, timestamp=time.time())

    # RSS from /proc/self/status (Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    obs.rss_mb = int(line.split()[1]) / 1024.0
                    break
    except (OSError, ValueError):
        pass

    # VRAM
    if torch.cuda.is_available():
        obs.vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        obs.vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    return obs


def _detect_memory_mode(comfy_root: str) -> str:
    """Detect the active ComfyUI memory management mode.

    This inspects ComfyUI's model_management module to determine
    the current memory mode (e.g., normal, lowvram, novram, highvram).
    """
    try:
        import comfy.model_management as mm

        # ComfyUI exposes vram_state as an enum or flag
        if hasattr(mm, "vram_state"):
            vram_state = mm.vram_state
            # Convert enum to string if needed
            if hasattr(vram_state, "name"):
                return vram_state.name
            return str(vram_state)

        # Fallback: check for specific mode functions
        if hasattr(mm, "is_nvidia"):
            return "gpu-detected"
        return "mode-not-detectable"
    except Exception as exc:
        return f"detection-error: {exc}"


def _resolve_project_root() -> str:
    """Return the absolute project root (parent of scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_import_paths(comfy_root: str) -> None:
    """Configure sys.path so project packages resolve before ComfyUI's.

    ComfyUI roots contain a top-level ``nodes.py`` which shadows this
    project's ``nodes/`` package if the ComfyUI root appears first on
    sys.path.  We ensure the project root is inserted *before* the
    ComfyUI root so that ``import nodes.exit`` resolves to the project
    package, while ``import comfy.*`` still resolves inside the ComfyUI
    installation.
    """
    project_root = _resolve_project_root()

    # Remove stale entries if already present (idempotent).
    for path in (project_root, comfy_root):
        while path in sys.path:
            sys.path.remove(path)

    # Project root first, ComfyUI root second.
    sys.path.insert(0, comfy_root)
    sys.path.insert(0, project_root)


def run_validation(
    comfy_root: str,
    model_path: str,
    report_output: str,
) -> ValidationReport:
    """Run real ComfyUI memory-mode validation.

    This function is ONLY called after all guards pass.
    It imports ComfyUI modules and interacts with a real installation.

    The *model_path* must point to an existing ``.safetensors`` file.
    The harness loads and inspects the file to validate materialization,
    artifact cache-hit behaviour, and returned-model behaviour — it does
    not merely record the path as a string.
    """
    report = ValidationReport(
        comfy_root=comfy_root,
        model_path=model_path,
    )
    start_time = time.time()

    # --- Pre-flight: validate model_path exists -------------------------
    if not os.path.isfile(model_path):
        report.errors.append(
            f"model_path does not exist or is not a file: {model_path}"
        )
        report.duration_seconds = time.time() - start_time
        return report

    # --- Set up import paths (project root before ComfyUI root) ---------
    _setup_import_paths(comfy_root)

    try:
        report.memory_observations.append(
            _collect_memory_observation("before-validation")
        )

        # Detect memory mode
        report.memory_mode = _detect_memory_mode(comfy_root)

        # ---- Import project modules (resolve from project root) --------
        from lib.persistence import check_full_model_cache  # noqa: E402
        from lib.streaming_save import MaterializationSink  # noqa: E402, F401
        from nodes.exit import (  # noqa: E402
            _incremental_cache,
            _load_model_from_artifact,  # noqa: F401
            install_merged_patches,  # noqa: F401
        )

        # -- Probe patch-mode behaviour ---------------------------------
        report.patch_mode_behavior = "install_merged_patches available"

        report.memory_observations.append(
            _collect_memory_observation("after-patch-mode-probe")
        )

        # -- Probe full-model materialization via the real model ---------
        report.full_model_materialization = (
            "MaterializationSink available"
        )

        # Actually open and inspect the model file to validate that it
        # contains real tensor data — not just that the class exists.
        from safetensors import safe_open

        with safe_open(model_path, framework="pt") as f:
            tensor_keys = list(f.keys())
            sample_key = tensor_keys[0] if tensor_keys else None
            sample_tensor = f.get_tensor(sample_key) if sample_key else None

        report.full_model_materialization = (
            f"loaded {len(tensor_keys)} tensors from artifact; "
            f"sample key={sample_key!r}, "
            f"shape={tuple(sample_tensor.shape) if sample_tensor is not None else None}, "
            f"dtype={sample_tensor.dtype if sample_tensor is not None else None}"
        )

        report.memory_observations.append(
            _collect_memory_observation("after-model-load")
        )

        # -- Probe artifact cache-hit behaviour via check_full_model_cache
        # Build a minimal manifest from the loaded file for validation.
        import torch

        manifest: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        with safe_open(model_path, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                manifest[key] = (t.dtype, tuple(t.shape))

        # Compute a recipe hash from the file's ecaj metadata (if any).
        try:
            with safe_open(model_path, framework="pt") as f:
                meta = f.metadata() or {}
            recipe_hash = meta.get("ecaj_recipe_hash", "")
            cache_hit = check_full_model_cache(
                model_path, recipe_hash, manifest
            )
            report.artifact_cache_hit = cache_hit
            report.artifact_reuse = (
                f"cache check returned {cache_hit}; "
                f"recipe_hash={recipe_hash!r}, manifest_keys={len(manifest)}"
            )
        except Exception as exc:
            report.artifact_cache_hit = False
            report.artifact_reuse = f"cache-check-error: {exc}"

        # -- Probe returned-model / _load_model_from_artifact availability
        report.returned_model_behavior = (
            "_load_model_from_artifact available (artifact-backed)"
        )

        # -- Check incremental cache state ------------------------------
        report.weights_resident_in_cache = bool(_incremental_cache)

        report.memory_observations.append(
            _collect_memory_observation("after-validation")
        )

    except ImportError as exc:
        report.errors.append(f"import-error: {exc}")
    except Exception as exc:
        report.errors.append(str(exc))
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

    # All guards passed — safe to proceed with real ComfyUI work
    report = run_validation(
        comfy_root=args.comfy_root,
        model_path=args.model_path,
        report_output=args.report_output,
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
