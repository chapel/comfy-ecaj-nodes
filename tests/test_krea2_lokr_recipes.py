"""Recipe-level behavior for native full-factor Krea 2 LoKR packages."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from lib.analysis import analyze_recipe
from lib.executor import evaluate_recipe
from lib.lora.krea2 import Krea2Loader
from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge


class _ObservableWIDEN:
    """Minimal deterministic merge boundary exposing evaluated branch tensors."""

    t_factor = 1.0

    def __init__(self) -> None:
        self.branch_calls: list[tuple[torch.Tensor, ...]] = []

    def filter_delta_batched(
        self,
        lora_applied: torch.Tensor,
        _backbone: torch.Tensor,
    ) -> torch.Tensor:
        return lora_applied

    def merge_weights_batched(
        self,
        weights_list: list[torch.Tensor],
        _backbone: torch.Tensor,
    ) -> torch.Tensor:
        self.branch_calls.append(tuple(weight.clone() for weight in weights_list))
        return torch.stack(weights_list).mean(dim=0)


def _write_lokr(
    directory: Path,
    name: str,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    save_file(
        {
            "diffusion_model.blocks.0.attn.gate.lokr_w1": w1,
            "diffusion_model.blocks.0.attn.gate.lokr_w2": w2,
        },
        str(path),
    )
    return str(path)


def _evaluate(
    recipe: RecipeMerge,
    loras: tuple[RecipeLoRA, ...],
    loader: object,
    widen: _ObservableWIDEN,
) -> torch.Tensor:
    return evaluate_recipe(
        keys=["diffusion_model.blocks.0.attn.gate.weight"],
        base_batch=torch.zeros(1, 2, 3),
        recipe_node=recipe,
        loader=loader,
        widen=widen,
        set_id_map={id(lora): str(id(lora)) for lora in loras},
        device="cpu",
        dtype=torch.float32,
        arch="krea2",
    )


def test_chained_lokr_packages_add_overlapping_contributions_deterministically(
    tmp_path: Path,
) -> None:
    # AC: @krea2-lora-package-compatibility ac-chained-krea2-adapters-compose-additively
    path_a = _write_lokr(
        tmp_path,
        "a.safetensors",
        torch.tensor([[2.0]]),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    path_b = _write_lokr(
        tmp_path,
        "b.safetensors",
        torch.tensor([[3.0]]),
        torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
    )
    lora = RecipeLoRA(
        loras=(
            {"path": Path(path_a).name, "strength": 0.5},
            {"path": Path(path_b).name, "strength": 0.25},
        )
    )
    recipe = RecipeMerge(
        base=RecipeBase(model_patcher=object(), arch="krea2"),
        target=lora,
        backbone=None,
        t_factor=1.0,
    )
    analysis = analyze_recipe(recipe, lora_path_resolver=lambda name: str(tmp_path / name))
    key = "diffusion_model.blocks.0.attn.gate.weight"
    set_id = str(id(lora))
    assert isinstance(analysis.loader, Krea2Loader)
    analysis.loader.validate_compatible_keys({key}, {key: (2, 3)})

    assert analysis.set_affected == {set_id: {key}}
    assert len(analysis.loader.get_delta_specs([key], {key: 0}, set_id=set_id)) == 2

    widen = _ObservableWIDEN()
    first = _evaluate(recipe, (lora,), analysis.loader, widen)
    second = _evaluate(recipe, (lora,), analysis.loader, widen)
    expected = 0.5 * torch.kron(
        torch.tensor([[2.0]]),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ) + 0.25 * torch.kron(
        torch.tensor([[3.0]]),
        torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
    )

    assert torch.allclose(first[0], expected)
    assert torch.equal(second, first)
    analysis.loader.cleanup()


def test_branched_lokr_packages_keep_overlapping_contributions_isolated(
    tmp_path: Path,
) -> None:
    # AC: @krea2-lora-package-compatibility ac-branched-krea2-adapters-remain-isolated
    w2_a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w2_b = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    path_a = _write_lokr(tmp_path, "a.safetensors", torch.tensor([[2.0]]), w2_a)
    path_b = _write_lokr(tmp_path, "b.safetensors", torch.tensor([[3.0]]), w2_b)
    lora_a = RecipeLoRA(loras=({"path": Path(path_a).name, "strength": 0.5},))
    lora_b = RecipeLoRA(loras=({"path": Path(path_b).name, "strength": 0.25},))
    recipe = RecipeMerge(
        base=RecipeBase(model_patcher=object(), arch="krea2"),
        target=RecipeCompose(branches=(lora_a, lora_b)),
        backbone=None,
        t_factor=1.0,
    )
    analysis = analyze_recipe(recipe, lora_path_resolver=lambda name: str(tmp_path / name))
    key = "diffusion_model.blocks.0.attn.gate.weight"
    assert isinstance(analysis.loader, Krea2Loader)
    analysis.loader.validate_compatible_keys({key}, {key: (2, 3)})

    assert analysis.set_affected == {
        str(id(lora_a)): {key},
        str(id(lora_b)): {key},
    }

    widen = _ObservableWIDEN()
    first = _evaluate(recipe, (lora_a, lora_b), analysis.loader, widen)
    second = _evaluate(recipe, (lora_a, lora_b), analysis.loader, widen)
    expected_a = 0.5 * torch.kron(torch.tensor([[2.0]]), w2_a)
    expected_b = 0.25 * torch.kron(torch.tensor([[3.0]]), w2_b)

    assert len(widen.branch_calls) == 2
    for branch_call in widen.branch_calls:
        assert torch.allclose(branch_call[0][0], expected_a)
        assert torch.allclose(branch_call[1][0], expected_b)
        assert not torch.equal(branch_call[0], branch_call[1])
    assert torch.allclose(first[0], torch.stack((expected_a, expected_b)).mean(dim=0))
    assert torch.equal(second, first)
    analysis.loader.cleanup()
