"""Krea 2 validation fixtures and optional probe coverage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from lib.architecture import detect_supported_architecture
from lib.block_classify import classify_key, classify_layer_type
from lib.executor import _get_block_t_factors
from lib.lora.krea2 import Krea2CompatibilityError, Krea2Loader
from lib.recipe import BlockConfig
from tests.conftest import MockModelPatcher
from tests.krea2_validation_fixtures import (
    KREA2_BLOCK_CONTROL_KEYS,
    KREA2_VALIDATION_KEYS,
    KREA2_VALIDATION_SHAPES,
    diffusers_krea2_lora_tensors,
    krea2_state_tensors,
    native_krea2_lora_tensors,
    unsupported_krea2_lora_tensors,
    write_safetensors,
)

_PROBE_PATH = "scripts/manual/krea2_header_probe.py"


def _import_probe():
    mod_name = "krea2_header_probe_test"
    spec = importlib.util.spec_from_file_location(mod_name, _PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _import_probe()


# AC: @krea2-architecture-support ac-krea2-detected-from-krea-signature
# AC: @krea2-architecture-support ac-ambiguous-or-unknown-architecture-rejected
def test_shared_krea2_validation_keys_exercise_positive_and_negative_detection() -> None:
    assert detect_supported_architecture(KREA2_VALIDATION_KEYS) == "krea2"
    assert MockModelPatcher(keys=KREA2_VALIDATION_KEYS).model_state_dict()

    incomplete = tuple(
        key for key in KREA2_VALIDATION_KEYS if not key.startswith("diffusion_model.txtfusion.")
    )
    with pytest.raises(ValueError) as exc_info:
        detect_supported_architecture(incomplete)

    message = str(exc_info.value)
    assert "krea2" in message
    assert "text_fusion" in message


# AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
# AC: @krea2-lora-package-compatibility ac-krea2-lora-strength-controls-are-stable
# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_shared_lora_fixtures_cover_supported_families_and_diagnostics(tmp_path: Path) -> None:
    diffusers_path = write_safetensors(
        tmp_path / "diffusers.safetensors",
        diffusers_krea2_lora_tensors(),
    )
    native_path = write_safetensors(
        tmp_path / "native.safetensors",
        native_krea2_lora_tensors(),
    )

    loader = Krea2Loader()
    loader.load(diffusers_path, strength=0.5, set_id="diffusers")
    loader.load(native_path, strength=0.5, set_id="native")
    loader.validate_compatible_keys(set(KREA2_VALIDATION_KEYS), KREA2_VALIDATION_SHAPES)

    assert "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight" in (
        loader.affected_keys_for_set("diffusers")
    )
    assert "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight" in (
        loader.affected_keys_for_set("native")
    )

    key = "diffusion_model.blocks.0.attn.wq.weight"
    specs_a = loader.get_delta_specs([key], {key: 0}, set_id="diffusers")
    loader_b = Krea2Loader()
    loader_b.load(diffusers_path, strength=0.25, set_id="diffusers")
    specs_b = loader_b.get_delta_specs([key], {key: 0}, set_id="diffusers")
    assert specs_a[0].scale == pytest.approx(specs_b[0].scale * 2)

    unsupported_path = write_safetensors(
        tmp_path / "unsupported.safetensors",
        unsupported_krea2_lora_tensors(),
    )
    rejected = Krea2Loader()
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        rejected.load(unsupported_path)
    assert "unsupported" in str(exc_info.value)
    assert "incomplete" in str(exc_info.value)
    assert rejected.affected_keys == frozenset()


# AC: @krea2-block-and-layer-controls ac-main-model-regions-are-controllable
# AC: @krea2-block-and-layer-controls ac-text-fusion-regions-are-controllable
# AC: @krea2-block-and-layer-controls ac-layer-category-controls-are-controllable
def test_shared_block_fixture_exercises_region_and_layer_controls() -> None:
    config = BlockConfig(
        arch="krea2",
        block_overrides=(("B00", 0.25), ("TF_REF1", 0.4)),
        layer_type_overrides=(("embedding_projection", 1.5), ("structural", 0.0)),
    )

    groups = _get_block_t_factors(KREA2_BLOCK_CONTROL_KEYS, config, "krea2", 1.0)

    assert groups == {
        0.25: [0],
        1.0: [1, 2, 6],
        0.4: [3],
        1.5: [4],
        0.0: [5],
    }
    assert classify_key(KREA2_BLOCK_CONTROL_KEYS[0], "krea2") == "B00"
    assert classify_key(KREA2_BLOCK_CONTROL_KEYS[3], "krea2") == "TF_REF1"
    assert classify_layer_type(KREA2_BLOCK_CONTROL_KEYS[4], "krea2") == "embedding_projection"
    assert classify_layer_type(KREA2_BLOCK_CONTROL_KEYS[5], "krea2") == "structural"


# AC: @krea2-architecture-support ac-krea2-detected-from-krea-signature
# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_optional_probe_reports_krea_model_header_and_dtype_warnings(tmp_path: Path) -> None:
    model_path = write_safetensors(
        tmp_path / "krea_model.safetensors",
        krea2_state_tensors(
            mixed_dtype_key="diffusion_model.txtfusion.projector.weight",
            mixed_dtype=torch.float16,
        ),
    )

    result = probe.inspect_model_header(model_path)

    assert result.architecture == "krea2"
    assert result.tensor_count == len(KREA2_VALIDATION_KEYS)
    assert result.dtypes == {"F16": 1, "F32": len(KREA2_VALIDATION_KEYS) - 1}
    assert "krea2=complete" in result.evidence
    assert any("mixed floating dtypes" in warning for warning in result.warnings)
    assert result.errors == []


# AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
# AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
def test_optional_probe_reports_lora_header_compatibility_without_loading_payloads(
    tmp_path: Path,
) -> None:
    supported_path = write_safetensors(
        tmp_path / "supported_lora.safetensors",
        diffusers_krea2_lora_tensors(),
    )
    lokr_path = write_safetensors(
        tmp_path / "lokr_lora.safetensors",
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
        },
    )
    incomplete_lokr_path = write_safetensors(
        tmp_path / "incomplete_lokr.safetensors",
        {"diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2)},
    )
    unsupported_path = write_safetensors(
        tmp_path / "unsupported_lora.safetensors",
        unsupported_krea2_lora_tensors(),
    )

    supported = probe.inspect_lora_header(supported_path)
    lokr = probe.inspect_lora_header(lokr_path)
    incomplete_lokr = probe.inspect_lora_header(incomplete_lokr_path)
    unsupported = probe.inspect_lora_header(unsupported_path)

    assert "diffusion_model.txtfusion.refiner_blocks.0.mlp.down.weight" in (
        supported.supported_lora_groups
    )
    assert supported.unsupported_lora_groups == []
    assert supported.incomplete_lora_groups == []
    assert supported.errors == []

    assert "diffusion_model.blocks.0.attn.gate.weight" in lokr.supported_lora_groups
    assert lokr.unsupported_lora_groups == []
    assert lokr.incomplete_lora_groups == []
    assert lokr.errors == []

    assert incomplete_lokr.incomplete_lora_groups == ["diffusion_model.blocks.0.attn.gate.weight"]
    assert "incomplete Krea 2 LoRA/LoKR factor groups detected" in incomplete_lokr.errors
    assert all("up/down" not in error for error in incomplete_lokr.errors)

    assert unsupported.unsupported_lora_groups
    assert unsupported.incomplete_lora_groups == ["diffusion_model.blocks.0.attn.wv.weight"]
    assert any("unsupported" in error for error in unsupported.errors)
    assert any("incomplete" in error for error in unsupported.errors)


# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
def test_optional_probe_rejects_unmapped_alpha_alongside_valid_lokr(tmp_path: Path) -> None:
    package_path = write_safetensors(
        tmp_path / "lokr_with_unmapped_alpha.safetensors",
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
            "unrelated.unsupported.alpha": torch.tensor(1.0),
        },
    )

    result = probe.inspect_lora_header(package_path)

    assert result.supported_lora_groups == ["diffusion_model.blocks.0.attn.gate.weight"]
    assert result.unsupported_lora_groups == ["unrelated.unsupported.alpha"]
    assert "unsupported Krea 2 LoRA tensor groups detected" in result.errors

    with pytest.raises(Krea2CompatibilityError, match="unrelated.unsupported.alpha"):
        Krea2Loader().load(package_path)


# AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
# AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
def test_optional_probe_accepts_complete_lora_and_lokr_for_same_mapped_key(
    tmp_path: Path,
) -> None:
    package_path = write_safetensors(
        tmp_path / "mixed_lora_lokr.safetensors",
        {
            "diffusion_model.blocks.0.attn.gate.lora_down.weight": torch.ones(2, 6),
            "diffusion_model.blocks.0.attn.gate.lora_up.weight": torch.ones(4, 2),
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
            "diffusion_model.blocks.0.attn.gate.alpha": torch.tensor(4.0),
        },
    )

    result = probe.inspect_lora_header(package_path)

    assert result.supported_lora_groups == ["diffusion_model.blocks.0.attn.gate.weight"]
    assert result.unsupported_lora_groups == []
    assert result.incomplete_lora_groups == []
    assert result.errors == []

    loader = Krea2Loader()
    loader.load(package_path)
    model_key = "diffusion_model.blocks.0.attn.gate.weight"
    specs = loader.get_delta_specs([model_key], {model_key: 0})
    assert [spec.kind for spec in specs] == ["standard", "lokr"]


# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
@pytest.mark.parametrize(
    "include_valid_lokr", [False, True], ids=["standard-only", "mixed-valid-lokr"]
)
def test_optional_probe_matches_loader_zero_rank_standard_rejection(
    tmp_path: Path,
    include_valid_lokr: bool,
) -> None:
    tensors = {
        "diffusion_model.blocks.0.attn.gate.lora_down.weight": torch.ones(0, 6),
        "diffusion_model.blocks.0.attn.gate.lora_up.weight": torch.ones(4, 0),
    }
    if include_valid_lokr:
        tensors.update(
            {
                "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
                "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
            }
        )
    package_path = write_safetensors(tmp_path / "zero_rank.safetensors", tensors)

    result = probe.inspect_lora_header(package_path)
    loader = Krea2Loader()
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.load(package_path)

    model_key = "diffusion_model.blocks.0.attn.gate.weight"
    assert "shape-incompatible groups" in str(exc_info.value)
    assert "rank must be positive" in str(exc_info.value)
    assert result.shape_incompatible_lora_groups == [f"{model_key} rank must be positive"]
    assert "shape-incompatible Krea 2 LoRA tensor groups detected" in result.errors
    assert result.supported_lora_groups == ([model_key] if include_valid_lokr else [])
    assert loader.affected_keys == frozenset()


# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
@pytest.mark.parametrize(
    ("case_name", "tensors"),
    [
        (
            "one_dimensional_lokr_w1",
            {
                "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(4),
                "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
            },
        ),
        (
            "three_dimensional_lokr_w2",
            {
                "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
                "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(1, 2, 3),
            },
        ),
        (
            "one_dimensional_standard_down",
            {
                "diffusion_model.blocks.0.attn.gate.lora_down.weight": torch.ones(6),
                "diffusion_model.blocks.0.attn.gate.lora_up.weight": torch.ones(4, 2),
            },
        ),
        (
            "standard_rank_mismatch",
            {
                "diffusion_model.blocks.0.attn.gate.lora_down.weight": torch.ones(3, 6),
                "diffusion_model.blocks.0.attn.gate.lora_up.weight": torch.ones(4, 2),
            },
        ),
        (
            "two_dimensional_direct_bias",
            {"diffusion_model.blocks.0.attn.gate.diff_b": torch.ones(2, 2)},
        ),
    ],
)
def test_optional_probe_matches_loader_shape_rejection_from_headers(
    tmp_path: Path,
    case_name: str,
    tensors: dict[str, torch.Tensor],
) -> None:
    package_path = write_safetensors(tmp_path / f"{case_name}.safetensors", tensors)

    result = probe.inspect_lora_header(package_path)
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        Krea2Loader().load(package_path)

    assert "shape-incompatible groups" in str(exc_info.value)
    assert result.shape_incompatible_lora_groups
    assert "shape-incompatible Krea 2 LoRA tensor groups detected" in result.errors
    assert not result.unsupported_lora_groups


# AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
def test_optional_probe_matches_loader_non_scalar_alpha_reason_category(tmp_path: Path) -> None:
    package_path = write_safetensors(
        tmp_path / "non_scalar_alpha.safetensors",
        {
            "diffusion_model.blocks.0.attn.gate.lora_down.weight": torch.ones(2, 6),
            "diffusion_model.blocks.0.attn.gate.lora_up.weight": torch.ones(4, 2),
            "diffusion_model.blocks.0.attn.gate.alpha": torch.ones(2),
        },
    )

    result = probe.inspect_lora_header(package_path)
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        Krea2Loader().load(package_path)

    assert "shape-incompatible groups" in str(exc_info.value)
    assert result.shape_incompatible_lora_groups == [
        "diffusion_model.blocks.0.attn.gate.alpha alpha must be scalar"
    ]
    assert result.unsupported_lora_groups == []
    assert "shape-incompatible Krea 2 LoRA tensor groups detected" in result.errors


# AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
def test_optional_probe_requires_explicit_operator_inputs(tmp_path: Path) -> None:
    model_path = write_safetensors(tmp_path / "krea_model.safetensors", krea2_state_tensors())

    refused = probe.check_guards(env={}, argv=[])
    assert not refused.passed
    assert any("COMFY_ECAJ_KREA2_PROBE" in reason for reason in refused.reasons)

    accepted = probe.check_guards(
        env={"COMFY_ECAJ_KREA2_PROBE": "1"},
        argv=["--run-krea2-probe", "--model-path", model_path],
    )
    assert accepted.passed
    assert accepted.reasons == ()

    report = probe.build_report([model_path], [], skip_reason="GPU assets unavailable")
    assert report.model_probes[0].architecture == "krea2"
    assert report.skip_reason == "GPU assets unavailable"
