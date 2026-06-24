#!/usr/bin/env python3
"""Header-only Krea 2 validation probe.

This helper is optional evidence tooling, not part of the automated suite. It
requires explicit operator-provided paths and inspects safetensors headers only:
key names, shapes, dtypes, and metadata. It does not materialize tensor payloads
or require a running ComfyUI process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from safetensors import safe_open

from lib.architecture import (
    ArchitectureDetectionError,
    detect_supported_architecture,
    format_architecture_evidence,
    match_architecture_evidence,
)
from lib.lora.krea2 import KREA2_COMPATIBILITY_FAMILIES, _parse_krea2_lora_key

ENV_VAR = "COMFY_ECAJ_KREA2_PROBE"
CLI_FLAG = "--run-krea2-probe"


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass
class HeaderProbe:
    path: str
    kind: str
    tensor_count: int = 0
    architecture: str = ""
    evidence: str = ""
    dtypes: dict[str, int] = field(default_factory=dict)
    supported_lora_groups: list[str] = field(default_factory=list)
    unsupported_lora_groups: list[str] = field(default_factory=list)
    incomplete_lora_groups: list[str] = field(default_factory=list)
    key_samples: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Krea2ProbeReport:
    model_probes: list[HeaderProbe] = field(default_factory=list)
    lora_probes: list[HeaderProbe] = field(default_factory=list)
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def check_guards(
    env: dict[str, str] | None = None,
    argv: list[str] | None = None,
) -> GuardResult:
    if env is None:
        env = dict(os.environ)
    if argv is None:
        argv = sys.argv[1:]

    reasons: list[str] = []
    if env.get(ENV_VAR, "") != "1":
        reasons.append(f"Environment variable {ENV_VAR} is not set to '1'.")
    if CLI_FLAG not in argv:
        reasons.append(f"CLI flag {CLI_FLAG} is not present.")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(CLI_FLAG, action="store_true", dest="run_flag")
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--lora-path", action="append", default=[])
    parser.add_argument("--report-output", default="")
    known, _ = parser.parse_known_args(argv)

    if not known.model_path and not known.lora_path:
        reasons.append("At least one --model-path or --lora-path is required.")
    for label, paths in (("--model-path", known.model_path), ("--lora-path", known.lora_path)):
        for path in paths:
            if not path:
                reasons.append(f"{label} cannot be empty.")
            elif not Path(path).is_file():
                reasons.append(f"{label} does not exist or is not a file: {path}")

    if reasons:
        return GuardResult(False, tuple(reasons))
    return GuardResult(True)


def parse_validated_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Header-only Krea 2 validation probe.")
    parser.add_argument(CLI_FLAG, action="store_true", dest="run_flag")
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Explicit path to a Krea 2 model safetensors file. Repeatable.",
    )
    parser.add_argument(
        "--lora-path",
        action="append",
        default=[],
        help="Explicit path to a Krea 2 LoRA safetensors file. Repeatable.",
    )
    parser.add_argument("--report-output", default="", help="Optional JSON report output path.")
    parser.add_argument(
        "--skip-reason",
        default="",
        help="Reason recorded when the operator cannot run a real Comfy/GPU smoke.",
    )
    return parser.parse_args(argv)


def _dtype_warnings(path: str, dtypes: Iterable[str]) -> list[str]:
    dtype_set = set(dtypes)
    warnings: list[str] = []
    float_dtypes = {dt for dt in dtype_set if dt.startswith(("F", "BF"))}
    int_dtypes = {dt for dt in dtype_set if dt.startswith(("I", "U"))}
    fp8_dtypes = {dt for dt in dtype_set if dt.startswith("F8")}
    if len(float_dtypes) > 1:
        warnings.append(
            f"{path}: mixed floating dtypes detected: {', '.join(sorted(float_dtypes))}"
        )
    if fp8_dtypes:
        warnings.append(
            f"{path}: FP8/float8 tensor headers detected; run only with matching runtime support."
        )
    if int_dtypes:
        warnings.append(
            f"{path}: integer or quantized-looking tensor headers detected: "
            f"{', '.join(sorted(int_dtypes))}"
        )
    return warnings


def _read_header(path: str) -> tuple[list[str], dict[str, int], dict[str, list[int]]]:
    keys: list[str] = []
    dtypes: dict[str, int] = defaultdict(int)
    shapes: dict[str, list[int]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_slice = f.get_slice(key)
            keys.append(key)
            dtypes[tensor_slice.get_dtype()] += 1
            shapes[key] = list(tensor_slice.get_shape())
    return keys, dict(sorted(dtypes.items())), shapes


def inspect_model_header(path: str) -> HeaderProbe:
    probe = HeaderProbe(path=path, kind="model")
    try:
        keys, dtypes, _shapes = _read_header(path)
    except Exception as exc:  # pragma: no cover - defensive for operator files.
        probe.errors.append(f"Could not read safetensors header: {exc}")
        return probe

    probe.tensor_count = len(keys)
    probe.dtypes = dtypes
    probe.key_samples = sorted(keys)[:12]
    krea_match = match_architecture_evidence(keys, "krea2")
    probe.evidence = format_architecture_evidence((krea_match,))
    probe.warnings.extend(_dtype_warnings(path, dtypes))
    try:
        probe.architecture = detect_supported_architecture(keys)
        if probe.architecture != "krea2":
            probe.warnings.append(f"{path}: detected supported architecture {probe.architecture}.")
    except ArchitectureDetectionError as exc:
        probe.errors.append(str(exc))
    return probe


def inspect_lora_header(path: str) -> HeaderProbe:
    probe = HeaderProbe(path=path, kind="lora")
    try:
        keys, dtypes, _shapes = _read_header(path)
    except Exception as exc:  # pragma: no cover - defensive for operator files.
        probe.errors.append(f"Could not read safetensors header: {exc}")
        return probe

    grouped: dict[str, set[str]] = defaultdict(set)
    unsupported: list[str] = []
    direct_groups: set[str] = set()
    for key in keys:
        if key.endswith(".alpha"):
            continue
        parsed = _parse_krea2_lora_key(key)
        if parsed is None:
            unsupported.append(key)
        elif parsed.direct_delta:
            direct_groups.add(parsed.group_key)
        else:
            grouped[parsed.group_key].add(parsed.direction)

    incomplete = [group for group, directions in grouped.items() if directions != {"down", "up"}]
    supported_groups = sorted(
        group for group, directions in grouped.items() if directions == {"down", "up"}
    )
    supported_groups.extend(sorted(direct_groups))

    probe.tensor_count = len(keys)
    probe.dtypes = dtypes
    probe.key_samples = sorted(keys)[:12]
    probe.supported_lora_groups = sorted(supported_groups)
    probe.unsupported_lora_groups = sorted(unsupported)[:24]
    probe.incomplete_lora_groups = sorted(incomplete)[:24]
    probe.evidence = "Supported Krea 2 package families: " + ", ".join(
        sorted(KREA2_COMPATIBILITY_FAMILIES)
    )
    probe.warnings.extend(_dtype_warnings(path, dtypes))
    if unsupported:
        probe.errors.append("unsupported Krea 2 LoRA tensor groups detected")
    if incomplete:
        probe.errors.append("incomplete Krea 2 LoRA up/down groups detected")
    if not supported_groups:
        probe.errors.append("no supported Krea 2 LoRA tensor groups detected")
    return probe


def build_report(
    model_paths: Iterable[str],
    lora_paths: Iterable[str],
    *,
    skip_reason: str = "",
) -> Krea2ProbeReport:
    return Krea2ProbeReport(
        model_probes=[inspect_model_header(path) for path in model_paths],
        lora_probes=[inspect_lora_header(path) for path in lora_paths],
        skip_reason=skip_reason,
    )


def format_refusal(guard_result: GuardResult) -> str:
    lines = [
        "Krea 2 header probe refused; required inputs are missing or invalid:",
        *[f"- {reason}" for reason in guard_result.reasons],
        "",
        f"Required opt-in: {ENV_VAR}=1 and {CLI_FLAG}.",
        "Provide explicit --model-path and/or --lora-path values.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    guard = check_guards(argv=argv)
    if not guard.passed:
        print(format_refusal(guard), file=sys.stderr)
        return 1

    args = parse_validated_args(argv)
    report = build_report(args.model_path, args.lora_path, skip_reason=args.skip_reason)
    output = report.to_json()
    if args.report_output:
        Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    has_errors = any(probe.errors for probe in report.model_probes + report.lora_probes)
    return 2 if has_errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
