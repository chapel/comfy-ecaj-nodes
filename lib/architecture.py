"""Architecture detection helpers shared by Entry nodes and checkpoint loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = [
    "ARCHITECTURE_RULES",
    "ArchitectureDetectionError",
    "ArchitectureMatch",
    "SUPPORTED_ARCHITECTURES",
    "detect_supported_architecture",
    "format_architecture_evidence",
    "match_architecture_evidence",
]


class ArchitectureDetectionError(ValueError):
    """Raised when supported architecture detection is unknown or ambiguous."""


@dataclass(frozen=True)
class ArchitectureMatch:
    """Evidence collected for one architecture rule."""

    arch: str
    matched: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return bool(self.matched) and not self.missing

    @property
    def has_partial_evidence(self) -> bool:
        return bool(self.matched) and bool(self.missing)


@dataclass(frozen=True)
class ArchitectureRule:
    """Declarative architecture signature rule."""

    arch: str
    checks: tuple[tuple[str, Callable[[frozenset[str]], bool]], ...]

    def evaluate(self, keys: frozenset[str]) -> ArchitectureMatch:
        matched: list[str] = []
        missing: list[str] = []
        for name, check in self.checks:
            if check(keys):
                matched.append(name)
            else:
                missing.append(name)
        return ArchitectureMatch(self.arch, tuple(matched), tuple(missing))


def _key_contains(keys: frozenset[str], needle: str) -> bool:
    return any(needle in key for key in keys)


def _has_krea2_main_block(keys: frozenset[str]) -> bool:
    return (
        _key_contains(keys, "diffusion_model.blocks.")
        and _key_contains(keys, ".mod.lin")
        and _key_contains(keys, ".attn.wq.")
    )


def _has_krea2_text_fusion(keys: frozenset[str]) -> bool:
    return (
        _key_contains(keys, "diffusion_model.txtfusion.layerwise_blocks.")
        and _key_contains(keys, "diffusion_model.txtfusion.refiner_blocks.")
        and _key_contains(keys, "diffusion_model.txtfusion.projector.")
    )


def _has_krea2_structural_projections(keys: frozenset[str]) -> bool:
    return (
        _key_contains(keys, "diffusion_model.first.")
        and _key_contains(keys, "diffusion_model.tproj.")
        and _key_contains(keys, "diffusion_model.last.")
    )


ARCHITECTURE_RULES: tuple[ArchitectureRule, ...] = (
    # Krea 2: single-stream MMDiT with blocks, txtfusion adapter, and Krea
    # structural projections. Requiring all three avoids matching on a generic
    # transformer/block key.
    ArchitectureRule(
        "krea2",
        (
            ("main_blocks", _has_krea2_main_block),
            ("text_fusion", _has_krea2_text_fusion),
            ("structural_projections", _has_krea2_structural_projections),
        ),
    ),
    # Z-Image: layers.N with noise_refiner.
    ArchitectureRule(
        "zimage",
        (
            ("layers", lambda keys: _key_contains(keys, "diffusion_model.layers.")),
            ("noise_refiner", lambda keys: _key_contains(keys, "noise_refiner")),
        ),
    ),
    # SDXL: input_blocks, middle_block, output_blocks structure.
    ArchitectureRule(
        "sdxl",
        (
            ("input_blocks", lambda keys: _key_contains(keys, "diffusion_model.input_blocks.")),
            ("middle_block", lambda keys: _key_contains(keys, "diffusion_model.middle_block.")),
            ("output_blocks", lambda keys: _key_contains(keys, "diffusion_model.output_blocks.")),
        ),
    ),
    # Flux Klein: double_blocks are required; single_blocks alone are not enough.
    ArchitectureRule(
        "flux",
        (("double_blocks", lambda keys: _key_contains(keys, "double_blocks")),),
    ),
    # Qwen: many transformer_blocks. This intentionally remains a count-based
    # rule because shallow transformer_blocks evidence overlaps too broadly.
    ArchitectureRule(
        "qwen",
        (
            (
                "transformer_blocks_60_plus",
                lambda keys: (
                    sum(1 for key in keys if key.startswith("diffusion_model.transformer_blocks."))
                    >= 60
                ),
            ),
        ),
    ),
)

SUPPORTED_ARCHITECTURES = frozenset(rule.arch for rule in ARCHITECTURE_RULES)


def _as_keyset(keys: Iterable[str]) -> frozenset[str]:
    return frozenset(keys)


def match_architecture_evidence(keys: Iterable[str], arch: str) -> ArchitectureMatch:
    """Return matched/missing evidence for one architecture."""
    keyset = _as_keyset(keys)
    for rule in ARCHITECTURE_RULES:
        if rule.arch == arch:
            return rule.evaluate(keyset)
    raise KeyError(f"Unknown architecture rule: {arch}")


def _collect_matches(keys: Iterable[str]) -> tuple[ArchitectureMatch, ...]:
    keyset = _as_keyset(keys)
    return tuple(rule.evaluate(keyset) for rule in ARCHITECTURE_RULES)


def format_architecture_evidence(matches: Iterable[ArchitectureMatch]) -> str:
    """Format architecture evidence for user-facing diagnostics."""
    parts: list[str] = []
    for match in matches:
        if not match.matched:
            continue
        status = "complete" if match.is_complete else "partial"
        missing = f"; missing: {', '.join(match.missing)}" if match.missing else ""
        parts.append(f"{match.arch}={status} (matched: {', '.join(match.matched)}{missing})")
    if not parts:
        return "no supported architecture evidence"
    return "; ".join(parts)


def detect_supported_architecture(keys: Iterable[str]) -> str:
    """Detect one supported architecture or raise with evidence details."""
    matches = _collect_matches(keys)
    complete = [match for match in matches if match.is_complete]

    if len(complete) == 1:
        return complete[0].arch

    evidence = format_architecture_evidence(matches)
    supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))

    if len(complete) > 1:
        candidates = ", ".join(match.arch for match in complete)
        raise ArchitectureDetectionError(
            f"Ambiguous model architecture: matched multiple supported "
            f"signatures ({candidates}). Evidence: {evidence}. "
            f"Supported architectures: {supported}."
        )

    raise ArchitectureDetectionError(
        f"Could not detect model architecture. Evidence: {evidence}. "
        f"Supported architectures: {supported}."
    )
