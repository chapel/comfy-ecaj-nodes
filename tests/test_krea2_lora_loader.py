"""Tests for Krea 2 LoRA package compatibility."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.gpu_ops import apply_lora_batch_gpu
from lib.lora import get_loader
from lib.lora.krea2 import (
    Krea2CompatibilityError,
    Krea2Loader,
    _parse_krea2_lora_key,
)


def _write_lora(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "fixture.safetensors"
    save_file(tensors, str(path))
    return str(path)


def test_parse_diffusers_keys_preserves_text_fusion_compounds() -> None:
    # AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
    # AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
    parsed = _parse_krea2_lora_key(
        "transformer.text_fusion.layerwise_blocks.0.attn.to_out.0.lora_A.weight"
    )

    assert parsed.model_key == ("diffusion_model.txtfusion.layerwise_blocks.0.attn.wo.weight")
    assert parsed.direction == "down"
    assert parsed.direct_delta is False


def test_parse_native_keys_and_bias_deltas() -> None:
    # AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
    parsed = _parse_krea2_lora_key("diffusion_model.blocks.0.attn.wq.lora_down.weight")
    bias = _parse_krea2_lora_key("diffusion_model.txtmlp.1.diff_b")

    assert parsed.model_key == "diffusion_model.blocks.0.attn.wq.weight"
    assert parsed.direction == "down"
    assert bias.model_key == "diffusion_model.txtmlp.1.bias"
    assert bias.direct_delta is True


def test_parse_native_full_factor_lokr_keys_maps_to_base_weight() -> None:
    # AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
    w1 = _parse_krea2_lora_key("diffusion_model.blocks.7.attn.gate.lokr_w1")
    w2 = _parse_krea2_lora_key("diffusion_model.blocks.7.attn.gate.lokr_w2")

    assert w1 is not None
    assert w2 is not None
    assert w1.model_key == "diffusion_model.blocks.7.attn.gate.weight"
    assert w2.model_key == w1.model_key
    assert w1.group_key == w2.group_key == w1.model_key
    assert (w1.direction, w2.direction) == ("lokr_w1", "lokr_w2")
    assert not w1.direct_delta
    assert not w2.direct_delta


def test_diffusers_public_family_loads_without_manual_renaming(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.9.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.9.attn.to_q.lora_B.weight": torch.ones(4, 2),
            "transformer.text_fusion.refiner_blocks.1.ff.down.lora_A.weight": torch.ones(2, 5),
            "transformer.text_fusion.refiner_blocks.1.ff.down.lora_B.weight": torch.ones(3, 2),
            "transformer.img_in.lora_A.weight": torch.ones(2, 6),
            "transformer.img_in.lora_B.weight": torch.ones(7, 2),
            "transformer.txt_in.linear_2.lora_A.weight": torch.ones(2, 7),
            "transformer.txt_in.linear_2.lora_B.weight": torch.ones(7, 2),
            "transformer.final_layer.linear.lora_A.weight": torch.ones(2, 7),
            "transformer.final_layer.linear.lora_B.weight": torch.ones(8, 2),
        },
    )

    loader = Krea2Loader()
    loader.load(path, strength=0.5, set_id="diffusers")

    assert loader.affected_keys_for_set("diffusers") == {
        "diffusion_model.blocks.9.attn.wq.weight",
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.down.weight",
        "diffusion_model.first.weight",
        "diffusion_model.txtmlp.3.weight",
        "diffusion_model.last.linear.weight",
    }
    specs = loader.get_delta_specs(
        sorted(loader.affected_keys),
        {key: i for i, key in enumerate(sorted(loader.affected_keys))},
        set_id="diffusers",
    )
    assert len(specs) == 5
    assert {spec.kind for spec in specs} == {"standard"}


def test_diffusers_timestep_projection_groups_load_without_manual_renaming(
    tmp_path: Path,
) -> None:
    # AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    # AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
    path = _write_lora(
        tmp_path,
        {
            "transformer.time_embed.linear_1.lora_A.weight": torch.ones(2, 3),
            "transformer.time_embed.linear_1.lora_B.weight": torch.ones(4, 2),
            "transformer.time_embed.linear_2.lora_A.weight": torch.ones(2, 4),
            "transformer.time_embed.linear_2.lora_B.weight": torch.ones(5, 2),
            "transformer.time_mod_proj.lora_A.weight": torch.ones(2, 5),
            "transformer.time_mod_proj.lora_B.weight": torch.ones(6, 2),
        },
    )

    loader = Krea2Loader()
    loader.load(path, set_id="public")

    expected_keys = {
        "diffusion_model.tmlp.0.weight",
        "diffusion_model.tmlp.2.weight",
        "diffusion_model.tproj.1.weight",
    }
    assert loader.affected_keys_for_set("public") == expected_keys
    loader.validate_compatible_keys(
        expected_keys,
        {
            "diffusion_model.tmlp.0.weight": (4, 3),
            "diffusion_model.tmlp.2.weight": (5, 4),
            "diffusion_model.tproj.1.weight": (6, 5),
        },
    )

    specs = loader.get_delta_specs(
        sorted(expected_keys),
        {key: i for i, key in enumerate(sorted(expected_keys))},
        set_id="public",
    )
    assert len(specs) == 3
    assert {spec.kind for spec in specs} == {"standard"}


def test_native_public_family_loads_lora_and_bias_deltas(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load
    path = _write_lora(
        tmp_path,
        {
            "diffusion_model.blocks.0.attn.wk.lora_down.weight": torch.ones(2, 3),
            "diffusion_model.blocks.0.attn.wk.lora_up.weight": torch.ones(4, 2),
            "diffusion_model.txtfusion.layerwise_blocks.0.mlp.up.lora_down.weight": (
                torch.ones(2, 5)
            ),
            "diffusion_model.txtfusion.layerwise_blocks.0.mlp.up.lora_up.weight": (
                torch.ones(6, 2)
            ),
            "diffusion_model.tproj.1.diff_b": torch.arange(6, dtype=torch.float32),
        },
    )

    loader = Krea2Loader()
    loader.load(path, strength=0.25, set_id="native")

    keys = sorted(loader.affected_keys)
    key_indices = {key: i for i, key in enumerate(keys)}
    specs = loader.get_delta_specs(keys, key_indices, set_id="native")

    assert set(keys) == {
        "diffusion_model.blocks.0.attn.wk.weight",
        "diffusion_model.txtfusion.layerwise_blocks.0.mlp.up.weight",
        "diffusion_model.tproj.1.bias",
    }
    assert {spec.kind for spec in specs} == {"standard", "direct"}

    base = torch.zeros(1, 6)
    direct_key = "diffusion_model.tproj.1.bias"
    direct_specs = loader.get_delta_specs([direct_key], {direct_key: 0}, set_id="native")
    result = apply_lora_batch_gpu([direct_key], base, direct_specs, "cpu", torch.float32)
    assert torch.equal(result[0], torch.arange(6, dtype=torch.float32) * 0.25)


def test_native_lokr_family_loads_and_applies_kron_delta(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
    # AC: @krea2-lora-package-compatibility ac-krea2-lora-strength-controls-are-stable
    # AC: @batched-executor ac-7
    path = _write_lora(
        tmp_path,
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
            ),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.tensor(
                [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
            ),
            # Full w1/w2 LoKR packages carry alpha metadata, but LyCORIS/ai-toolkit
            # do not apply alpha scaling when both full factors are present.
            "diffusion_model.blocks.0.attn.gate.alpha": torch.tensor(9999220736.0),
        },
    )

    loader = Krea2Loader()
    loader.load(path, strength=0.25, set_id="lokr")

    key = "diffusion_model.blocks.0.attn.gate.weight"
    assert loader.affected_keys_for_set("lokr") == {key}
    loader.validate_compatible_keys({key}, {key: (4, 6)})

    specs = loader.get_delta_specs([key], {key: 0}, set_id="lokr")
    assert len(specs) == 1
    assert specs[0].kind == "lokr"
    assert specs[0].scale == pytest.approx(0.25)

    base = torch.zeros(1, 4, 6)
    result = apply_lora_batch_gpu([key], base, specs, "cpu", torch.float32)
    expected = 0.25 * torch.kron(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
    )
    assert torch.allclose(result[0], expected)


def test_native_lokr_expanded_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path,
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(2, 2),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
        },
    )
    key = "diffusion_model.blocks.0.attn.gate.weight"
    loader = Krea2Loader()
    loader.load(path, set_id="mismatch")

    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.validate_compatible_keys({key}, {key: (4, 5)})

    message = str(exc_info.value)
    assert "package delta shape (4, 6)" in message
    assert "current recipe shape (4, 5)" in message


@pytest.mark.parametrize("present_factor", ["lokr_w1", "lokr_w2"])
def test_native_lokr_missing_factor_is_rejected(
    tmp_path: Path,
    present_factor: str,
) -> None:
    # AC: @krea2-lora-package-compatibility ac-native-full-factor-lokr-packages-load
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path / present_factor,
        {f"diffusion_model.blocks.0.attn.gate.{present_factor}": torch.ones(2, 2)},
    )
    loader = Krea2Loader()

    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.load(path)

    message = str(exc_info.value)
    assert "incomplete LoKR w1/w2 groups" in message
    assert "diffusion_model.blocks.0.attn.gate.weight" in message
    assert "incomplete up/down groups" not in message
    assert loader.affected_keys == frozenset()


@pytest.mark.parametrize(
    "unsupported_suffix",
    ["lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t1", "lokr_t2"],
)
def test_native_lokr_decomposed_forms_fail_closed(
    tmp_path: Path,
    unsupported_suffix: str,
) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path / unsupported_suffix,
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.ones(1, 1),
            "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.ones(2, 3),
            f"diffusion_model.blocks.0.attn.gate.{unsupported_suffix}": torch.ones(1, 1),
        },
    )
    loader = Krea2Loader()

    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.load(path)

    assert "unsupported tensor groups" in str(exc_info.value)
    assert unsupported_suffix in str(exc_info.value)
    assert loader.affected_keys == frozenset()


def test_native_lokr_two_package_loaded_bytes_and_cleanup_are_exact(tmp_path: Path) -> None:
    # AC: @loader-memory-measurement ac-1
    # AC: @loader-memory-measurement ac-3
    # AC: @loader-memory-measurement ac-6
    first = {
        "diffusion_model.blocks.0.attn.gate.lokr_w1": torch.zeros(2, 2),
        "diffusion_model.blocks.0.attn.gate.lokr_w2": torch.zeros(2, 3),
    }
    second = {
        "diffusion_model.blocks.1.attn.gate.lokr_w1": torch.zeros(1, 2, dtype=torch.float64),
        "diffusion_model.blocks.1.attn.gate.lokr_w2": torch.zeros(3, 2, dtype=torch.float64),
    }
    path_a = _write_lora(tmp_path / "a", first)
    path_b = _write_lora(tmp_path / "b", second)
    loader = Krea2Loader()

    loader.load(path_a, set_id="a")
    loader.load(path_b, set_id="b")

    expected_bytes = sum(tensor.nbytes for tensor in (*first.values(), *second.values()))
    assert loader.loaded_bytes == expected_bytes
    assert loader.affected_keys == {
        "diffusion_model.blocks.0.attn.gate.weight",
        "diffusion_model.blocks.1.attn.gate.weight",
    }

    loader.cleanup()

    assert loader.loaded_bytes == 0
    assert loader.affected_keys == frozenset()
    assert loader.affected_keys_for_set("a") == set()
    assert loader.affected_keys_for_set("b") == set()


def test_strength_changes_only_scales_deterministically(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-krea2-lora-strength-controls-are-stable
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.0.ff.gate.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.ff.gate.lora_B.weight": torch.ones(4, 2),
        },
    )

    loader_a = Krea2Loader()
    loader_b = Krea2Loader()
    loader_c = Krea2Loader()
    loader_a.load(path, strength=1.0, set_id="s")
    loader_b.load(path, strength=0.5, set_id="s")
    loader_c.load(path, strength=0.5, set_id="s")

    key = "diffusion_model.blocks.0.mlp.gate.weight"
    specs_a = loader_a.get_delta_specs([key], {key: 0}, set_id="s")
    specs_b = loader_b.get_delta_specs([key], {key: 0}, set_id="s")
    specs_c = loader_c.get_delta_specs([key], {key: 0}, set_id="s")

    assert specs_a[0].scale == pytest.approx(specs_b[0].scale * 2)
    assert specs_b[0].scale == pytest.approx(specs_c[0].scale)
    assert torch.equal(specs_b[0].up, specs_c[0].up)
    assert torch.equal(specs_b[0].down, specs_c[0].down)


def test_explicit_alpha_uses_normalized_krea_key(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-krea2-lora-strength-controls-are-stable
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        },
    )

    loader = Krea2Loader()
    loader.load(path, strength=1.0, set_id="s")

    key = "diffusion_model.blocks.0.attn.wq.weight"
    specs = loader.get_delta_specs([key], {key: 0}, set_id="s")

    assert len(specs) == 1
    assert specs[0].scale == pytest.approx(0.5)


def test_unsupported_or_incomplete_groups_are_rejected(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path,
        {
            "transformer.unknown_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
        },
    )

    loader = Krea2Loader()
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.load(path)

    message = str(exc_info.value)
    assert "unsupported" in message
    assert "incomplete" in message
    assert "unknown_blocks" in message
    assert "diffusion_model.blocks.0.attn.wq.weight" in message
    assert loader.affected_keys == frozenset()


def test_shape_incompatible_groups_are_rejected(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": torch.ones(4, 5),
        },
    )

    loader = Krea2Loader()
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.load(path)

    assert "rank mismatch" in str(exc_info.value)
    assert "blocks.0.attn.wv" in str(exc_info.value)


def test_mapped_keys_must_exist_in_current_recipe_keyspace(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.3.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.3.attn.to_q.lora_B.weight": torch.ones(4, 2),
        },
    )

    loader = Krea2Loader()
    loader.load(path, set_id="s")

    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.validate_compatible_keys({"diffusion_model.blocks.0.attn.wq.weight"})

    message = str(exc_info.value)
    assert "not present in the current Krea 2 recipe" in message
    assert "diffusion_model.blocks.3.attn.wq.weight" in message


def test_mapped_keys_must_match_current_recipe_shapes(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        },
    )

    loader = Krea2Loader()
    loader.load(path, set_id="s")

    key = "diffusion_model.blocks.0.attn.wq.weight"
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.validate_compatible_keys({key}, {key: (5, 3)})

    message = str(exc_info.value)
    assert "shape-incompatible groups" in message
    assert key in message
    assert "package delta shape (4, 3)" in message
    assert "current recipe shape (5, 3)" in message


def test_each_loaded_set_shape_is_validated_before_execution(tmp_path: Path) -> None:
    # AC: @krea2-lora-package-compatibility ac-lora-compatibility-is-complete-or-rejected
    good_path = _write_lora(
        tmp_path / "good",
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
        },
    )
    bad_path = _write_lora(
        tmp_path / "bad",
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(5, 2),
        },
    )

    loader = Krea2Loader()
    loader.load(good_path, set_id="good")
    loader.load(bad_path, set_id="bad")

    key = "diffusion_model.blocks.0.attn.wq.weight"
    with pytest.raises(Krea2CompatibilityError) as exc_info:
        loader.validate_compatible_keys({key}, {key: (4, 3)})

    message = str(exc_info.value)
    assert "shape-incompatible groups" in message
    assert "set bad" in message
    assert "package delta shape (5, 3)" in message
    assert "current recipe shape (4, 3)" in message
    assert "set good" not in message


def test_krea2_loader_registry_and_cleanup(tmp_path: Path) -> None:
    # AC: @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths
    path = _write_lora(
        tmp_path,
        {
            "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight": torch.ones(2, 3),
            "transformer.transformer_blocks.0.attn.to_gate.lora_B.weight": torch.ones(4, 2),
        },
    )

    loader = get_loader("krea2")
    assert isinstance(loader, Krea2Loader)
    loader.load(path, set_id="a")
    assert loader.loaded_bytes > 0
    assert loader.affected_keys_for_set("a") == {"diffusion_model.blocks.0.attn.gate.weight"}

    loader.cleanup()

    assert loader.loaded_bytes == 0
    assert loader.affected_keys == frozenset()
    assert loader.affected_keys_for_set("a") == set()
