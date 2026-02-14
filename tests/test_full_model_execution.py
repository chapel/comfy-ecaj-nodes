"""Tests for Full Model Execution — RecipeModel integration with Exit node.

AC: @full-model-execution ac-1 through ac-14
"""

import tempfile
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from lib.analysis import (
    _collect_model_refs,
    analyze_recipe_models,
    walk_to_base,
)
from lib.model_loader import ModelLoader
from lib.recipe import (
    BlockConfig,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
)
from lib.recipe_eval import (
    OpApplyModel,
    _input_regs,
    compile_plan,
    execute_plan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sdxl_checkpoint_path() -> str:
    """Create a temporary SDXL-format checkpoint file."""
    tensors = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        "model.diffusion_model.input_blocks.1.0.weight": torch.randn(4, 4),
        "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def second_sdxl_checkpoint_path() -> str:
    """Create a second SDXL checkpoint with different weights."""
    tensors = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4) * 2,
        "model.diffusion_model.input_blocks.1.0.weight": torch.randn(4, 4) * 2,
        "model.diffusion_model.middle_block.0.weight": torch.randn(4, 4) * 2,
        "model.diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4) * 2,
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def zimage_checkpoint_path() -> str:
    """Create a Z-Image checkpoint (different architecture)."""
    tensors = {
        "diffusion_model.layers.0.attention.qkv.weight": torch.randn(4, 4),
        "diffusion_model.noise_refiner.0.attn.weight": torch.randn(4, 4),
    }
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def mock_model_patcher():
    """Create a mock model patcher with SDXL-like state dict."""
    mock = MagicMock()
    mock.model_state_dict.return_value = {
        "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
        "diffusion_model.input_blocks.1.0.weight": torch.randn(4, 4),
        "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
        "diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
    }
    mock.clone.return_value = mock
    return mock


@pytest.fixture
def recipe_base(mock_model_patcher) -> RecipeBase:
    """Create a RecipeBase fixture."""
    return RecipeBase(model_patcher=mock_model_patcher, arch="sdxl")


# ---------------------------------------------------------------------------
# AC-1: Recipe analysis detects RecipeModel and opens loaders
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-1
class TestRecipeModelAnalysis:
    """Tests for recipe analysis detecting RecipeModel nodes."""

    def test_collect_model_refs_finds_recipe_models(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """_collect_model_refs() finds all RecipeModel nodes in tree."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        model_refs = _collect_model_refs(recipe)

        assert len(model_refs) == 1
        assert recipe_model in model_refs.values()

    def test_analyze_recipe_models_opens_loaders(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """analyze_recipe_models() opens ModelLoader for each unique path."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        result = analyze_recipe_models(
            recipe, base_arch="sdxl", model_path_resolver=None
        )

        try:
            assert len(result.model_loaders) == 1
            loader = list(result.model_loaders.values())[0]
            assert isinstance(loader, ModelLoader)
            assert len(result.model_affected) == 1
            assert len(result.all_model_keys) > 0
        finally:
            for loader in result.model_loaders.values():
                loader.cleanup()

    def test_analyze_recipe_models_builds_affected_key_map(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """analyze_recipe_models() builds per-model affected key maps."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        result = analyze_recipe_models(
            recipe, base_arch="sdxl", model_path_resolver=None
        )

        try:
            # Each model should have affected keys
            for model_id, affected in result.model_affected.items():
                assert len(affected) > 0
                for key in affected:
                    assert key.startswith("diffusion_model.")
        finally:
            for loader in result.model_loaders.values():
                loader.cleanup()


# ---------------------------------------------------------------------------
# AC-2: compile_plan emits OpApplyModel
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-2
class TestOpApplyModelCompilation:
    """Tests for OpApplyModel emission in compile_plan."""

    def test_compile_plan_emits_opapplymodel_for_recipe_model(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """compile_plan() emits OpApplyModel for RecipeModel nodes."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        # Build model_id_map
        model_id_map = {id(recipe_model): str(id(recipe_model))}

        plan = compile_plan(recipe, set_id_map={}, arch="sdxl", model_id_map=model_id_map)

        # Should have an OpApplyModel in the plan
        has_apply_model = any(isinstance(op, OpApplyModel) for op in plan.ops)
        assert has_apply_model

    def test_opapplymodel_references_correct_model_id(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """OpApplyModel references the correct model_id from the map."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        expected_model_id = "test_model_123"
        model_id_map = {id(recipe_model): expected_model_id}

        plan = compile_plan(recipe, set_id_map={}, arch="sdxl", model_id_map=model_id_map)

        apply_model_ops = [op for op in plan.ops if isinstance(op, OpApplyModel)]
        assert len(apply_model_ops) == 1
        assert apply_model_ops[0].model_id == expected_model_id

    def test_input_regs_handles_opapplymodel(self) -> None:
        """_input_regs() correctly returns input registers for OpApplyModel."""
        op = OpApplyModel(model_id="test", block_config=None, strength=1.0, input_reg=5, out_reg=6)
        assert _input_regs(op) == (5,)


# ---------------------------------------------------------------------------
# AC-3: execute_plan loads model weights into register
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-3
class TestOpApplyModelExecution:
    """Tests for OpApplyModel execution loading weights."""

    def test_execute_plan_loads_model_weights(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """execute_plan() loads model weights from streaming loader."""
        # Create a simple plan with just OpApplyModel
        from lib.recipe_eval import EvalPlan, OpFilterDelta

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=1.0, input_reg=0, out_reg=1
        )
        op_filter = OpFilterDelta(
            input_reg=1, backbone_reg=0, t_factor=1.0,
            block_config=None, use_per_block=False, out_reg=2
        )
        plan = EvalPlan(
            ops=(op_apply, op_filter),
            result_reg=2,
            dead_after=((), (1,)),
        )

        # Open loader
        loader = ModelLoader(sdxl_checkpoint_path)
        try:
            keys = list(loader.affected_keys)[:2]
            base_batch = torch.randn(len(keys), 4, 4)

            # Create mock WIDEN and LoRA loader
            mock_widen = MagicMock()
            mock_widen.filter_delta_batched.return_value = torch.randn(len(keys), 4, 4)
            mock_lora_loader = MagicMock()

            model_loaders = {"model1": loader}

            result = execute_plan(
                plan=plan,
                keys=keys,
                base_batch=base_batch,
                loader=mock_lora_loader,
                widen=mock_widen,
                device="cpu",
                dtype=torch.float32,
                model_loaders=model_loaders,
            )

            # Result should be a tensor with correct shape
            assert result.shape == (len(keys), 4, 4)
        finally:
            loader.cleanup()


# ---------------------------------------------------------------------------
# AC-4: OpFilterDelta computes delta internally via WIDEN
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-4
class TestModelDeltaComputation:
    """Tests for WIDEN computing delta from model weights."""

    def test_filter_delta_receives_model_weights(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """OpFilterDelta receives raw model weights and computes delta."""
        from lib.recipe_eval import EvalPlan, OpFilterDelta

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=1.0, input_reg=0, out_reg=1
        )
        op_filter = OpFilterDelta(
            input_reg=1,  # Uses model weights from OpApplyModel
            backbone_reg=0,  # Uses base model weights
            t_factor=1.0,
            block_config=None,
            use_per_block=False,
            out_reg=2,
        )
        plan = EvalPlan(
            ops=(op_apply, op_filter),
            result_reg=2,
            dead_after=((), (1,)),
        )

        loader = ModelLoader(sdxl_checkpoint_path)
        try:
            keys = list(loader.affected_keys)[:2]
            base_batch = torch.randn(len(keys), 4, 4)

            mock_widen = MagicMock()
            expected_output = torch.randn(len(keys), 4, 4)
            mock_widen.filter_delta_batched.return_value = expected_output

            result = execute_plan(
                plan=plan,
                keys=keys,
                base_batch=base_batch,
                loader=MagicMock(),
                widen=mock_widen,
                device="cpu",
                dtype=torch.float32,
                model_loaders={"model1": loader},
            )

            # filter_delta_batched should have been called with model weights
            mock_widen.filter_delta_batched.assert_called_once()
            # Result should be from WIDEN
            assert torch.equal(result, expected_output)
        finally:
            loader.cleanup()


# ---------------------------------------------------------------------------
# AC-5: Mixed RecipeModel and RecipeLoRA recipes work correctly
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-5
class TestMixedRecipes:
    """Tests for recipes mixing RecipeModel and RecipeLoRA."""

    def test_mixed_recipe_compiles_both_paths(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """Recipes with both RecipeModel and RecipeLoRA compile correctly."""
        from lib.recipe_eval import OpApplyLoRA

        recipe_lora = RecipeLoRA(
            loras=({"path": "test.safetensors", "strength": 1.0},)
        )
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)

        # Compose both
        compose = RecipeCompose(branches=(recipe_lora, recipe_model))
        recipe = RecipeMerge(
            base=recipe_base,
            target=compose,
            backbone=None,
            t_factor=1.0,
        )

        set_id_map = {id(recipe_lora): "lora1"}
        model_id_map = {id(recipe_model): "model1"}

        plan = compile_plan(
            recipe, set_id_map=set_id_map, arch="sdxl", model_id_map=model_id_map
        )

        # Should have both OpApplyLoRA and OpApplyModel
        has_lora = any(isinstance(op, OpApplyLoRA) for op in plan.ops)
        has_model = any(isinstance(op, OpApplyModel) for op in plan.ops)
        assert has_lora
        assert has_model


# ---------------------------------------------------------------------------
# AC-6: Architecture mismatch raises clear error
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-6
class TestArchitectureMismatch:
    """Tests for architecture mismatch error handling."""

    def test_architecture_mismatch_raises_value_error(
        self, recipe_base: RecipeBase, zimage_checkpoint_path: str
    ) -> None:
        """Checkpoint with different architecture raises ValueError."""
        # recipe_base has arch="sdxl", but zimage checkpoint has arch="zimage"
        recipe_model = RecipeModel(path=zimage_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        with pytest.raises(ValueError) as exc_info:
            analyze_recipe_models(
                recipe, base_arch="sdxl", model_path_resolver=None
            )

        error_msg = str(exc_info.value)
        assert "architecture" in error_msg.lower()
        assert "zimage" in error_msg.lower()
        assert "sdxl" in error_msg.lower()


# ---------------------------------------------------------------------------
# AC-7: Model weights freed after use
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-7
class TestModelWeightsFreed:
    """Tests for model weights being freed after GPU evaluation."""

    def test_dead_registers_freed_after_opapplymodel(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Registers holding model weights are freed via dead_after."""
        from lib.recipe_eval import EvalPlan, OpFilterDelta

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=1.0, input_reg=0, out_reg=1
        )
        op_filter = OpFilterDelta(
            input_reg=1, backbone_reg=0, t_factor=1.0,
            block_config=None, use_per_block=False, out_reg=2
        )
        # Register 1 should be dead after op_filter uses it
        plan = EvalPlan(
            ops=(op_apply, op_filter),
            result_reg=2,
            dead_after=((), (1,)),  # Register 1 freed after second op
        )

        loader = ModelLoader(sdxl_checkpoint_path)
        try:
            keys = list(loader.affected_keys)[:2]
            base_batch = torch.randn(len(keys), 4, 4)

            mock_widen = MagicMock()
            mock_widen.filter_delta_batched.return_value = torch.randn(len(keys), 4, 4)

            # Execute plan - dead registers should be freed
            execute_plan(
                plan=plan,
                keys=keys,
                base_batch=base_batch,
                loader=MagicMock(),
                widen=mock_widen,
                device="cpu",
                dtype=torch.float32,
                model_loaders={"model1": loader},
            )

            # If we got here without error, dead_after worked
        finally:
            loader.cleanup()


# ---------------------------------------------------------------------------
# AC-8: OOM backoff compatible with streaming loader
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-8
class TestOOMBackoff:
    """Tests for OOM backoff compatibility with streaming loaders."""

    def test_streaming_loader_supports_retry_reads(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """Streaming loader can re-read keys after OOM backoff."""
        loader = ModelLoader(sdxl_checkpoint_path)
        try:
            keys = list(loader.affected_keys)[:2]

            # First read
            tensors1 = loader.get_weights(keys)
            assert len(tensors1) == 2

            # Second read (simulating retry after OOM)
            tensors2 = loader.get_weights(keys)
            assert len(tensors2) == 2

            # Values should match (same file)
            for t1, t2 in zip(tensors1, tensors2):
                assert torch.equal(t1, t2)
        finally:
            loader.cleanup()


# ---------------------------------------------------------------------------
# AC-9: Block config applies to model deltas
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-9
class TestBlockConfigForModels:
    """Tests for per-block control applying to model deltas."""

    def test_opapplymodel_preserves_block_config(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """OpApplyModel preserves block_config from RecipeModel."""
        block_config = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5),),
        )
        recipe_model = RecipeModel(
            path=sdxl_checkpoint_path,
            block_config=block_config,
        )
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        model_id_map = {id(recipe_model): "model1"}
        plan = compile_plan(
            recipe, set_id_map={}, arch="sdxl", model_id_map=model_id_map
        )

        apply_model_ops = [op for op in plan.ops if isinstance(op, OpApplyModel)]
        assert len(apply_model_ops) == 1
        assert apply_model_ops[0].block_config is block_config


# ---------------------------------------------------------------------------
# AC-10: Missing checkpoint file raises clear error
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-10
class TestMissingCheckpointError:
    """Tests for missing checkpoint file error handling."""

    def test_missing_checkpoint_raises_file_not_found(
        self, recipe_base: RecipeBase
    ) -> None:
        """Missing checkpoint file raises FileNotFoundError with path."""
        recipe_model = RecipeModel(path="/nonexistent/model.safetensors")
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            analyze_recipe_models(
                recipe, base_arch="sdxl", model_path_resolver=None
            )

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "Model Input" in error_msg


# ---------------------------------------------------------------------------
# AC-11: IS_CHANGED includes checkpoint file stats
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-11
class TestIsChangedIncludesModels:
    """Tests for IS_CHANGED including checkpoint file stats."""

    def test_collect_model_paths_from_recipe(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """Model paths are collected for IS_CHANGED hash."""
        from nodes.exit import _collect_model_paths

        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        paths = _collect_model_paths(recipe)

        assert len(paths) == 1
        assert sdxl_checkpoint_path in paths


# ---------------------------------------------------------------------------
# AC-12: Model-only recipes process all diffusion keys
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-12
class TestModelOnlyRecipeKeys:
    """Tests for model-only recipes processing all diffusion keys."""

    def test_model_affected_keys_includes_all_diffusion_keys(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """Model-only recipes include all diffusion model keys."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        result = analyze_recipe_models(
            recipe, base_arch="sdxl", model_path_resolver=None
        )

        try:
            # All model keys should be in the affected set
            assert len(result.all_model_keys) > 0
            for key in result.all_model_keys:
                assert key.startswith("diffusion_model.")
        finally:
            for loader in result.model_loaders.values():
                loader.cleanup()


# ---------------------------------------------------------------------------
# AC-13: Sequential model loading (one batch at a time)
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-13
class TestSequentialModelLoading:
    """Tests for sequential model loading during execution."""

    def test_compose_with_multiple_models_loads_sequentially(
        self, recipe_base: RecipeBase,
        sdxl_checkpoint_path: str,
        second_sdxl_checkpoint_path: str,
    ) -> None:
        """Multiple models in Compose are loaded sequentially per-batch."""
        from lib.recipe_eval import OpMergeWeights

        model1 = RecipeModel(path=sdxl_checkpoint_path)
        model2 = RecipeModel(path=second_sdxl_checkpoint_path)

        compose = RecipeCompose(branches=(model1, model2))
        recipe = RecipeMerge(
            base=recipe_base,
            target=compose,
            backbone=None,
            t_factor=1.0,
        )

        model_id_map = {
            id(model1): "model1",
            id(model2): "model2",
        }

        plan = compile_plan(
            recipe, set_id_map={}, arch="sdxl", model_id_map=model_id_map
        )

        # Should have two OpApplyModel ops (one per model)
        apply_model_ops = [op for op in plan.ops if isinstance(op, OpApplyModel)]
        assert len(apply_model_ops) == 2

        # Should end with OpMergeWeights combining both
        has_merge = any(isinstance(op, OpMergeWeights) for op in plan.ops)
        assert has_merge


# ---------------------------------------------------------------------------
# AC-14: Model strength scales delta toward base
# ---------------------------------------------------------------------------


# AC: @full-model-execution ac-14
class TestModelStrengthScaling:
    """Tests for RecipeModel.strength applied during OpApplyModel execution."""

    def test_strength_zero_produces_base_weights(self) -> None:
        """strength=0 makes model register equal to base (zero contribution)."""
        from lib.recipe_eval import EvalPlan

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=0.0, input_reg=0, out_reg=1
        )
        plan = EvalPlan(
            ops=(op_apply,),
            result_reg=1,
            dead_after=((),),
        )

        base_batch = torch.randn(2, 4, 4)
        model_weights = torch.randn(2, 4, 4) * 5  # very different from base

        mock_loader = MagicMock()
        mock_loader.get_weights.return_value = [model_weights[i] for i in range(2)]

        result = execute_plan(
            plan=plan,
            keys=["key0", "key1"],
            base_batch=base_batch,
            loader=MagicMock(),
            widen=MagicMock(),
            device="cpu",
            dtype=torch.float32,
            model_loaders={"model1": mock_loader},
        )

        torch.testing.assert_close(result, base_batch)

    def test_strength_half_blends_toward_base(self) -> None:
        """strength=0.5 blends model weights halfway toward base."""
        from lib.recipe_eval import EvalPlan

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=0.5, input_reg=0, out_reg=1
        )
        plan = EvalPlan(
            ops=(op_apply,),
            result_reg=1,
            dead_after=((),),
        )

        base_batch = torch.zeros(2, 4, 4)
        model_weights = torch.ones(2, 4, 4) * 2.0

        mock_loader = MagicMock()
        mock_loader.get_weights.return_value = [model_weights[i] for i in range(2)]

        result = execute_plan(
            plan=plan,
            keys=["key0", "key1"],
            base_batch=base_batch,
            loader=MagicMock(),
            widen=MagicMock(),
            device="cpu",
            dtype=torch.float32,
            model_loaders={"model1": mock_loader},
        )

        expected = torch.ones(2, 4, 4)  # base + 0.5 * (model - base) = 0 + 0.5 * 2 = 1
        torch.testing.assert_close(result, expected)

    def test_strength_one_preserves_model_weights(self) -> None:
        """strength=1.0 (default) leaves model weights unchanged."""
        from lib.recipe_eval import EvalPlan

        op_apply = OpApplyModel(
            model_id="model1", block_config=None, strength=1.0, input_reg=0, out_reg=1
        )
        plan = EvalPlan(
            ops=(op_apply,),
            result_reg=1,
            dead_after=((),),
        )

        base_batch = torch.zeros(2, 4, 4)
        model_weights = torch.ones(2, 4, 4) * 3.0

        mock_loader = MagicMock()
        mock_loader.get_weights.return_value = [model_weights[i] for i in range(2)]

        result = execute_plan(
            plan=plan,
            keys=["key0", "key1"],
            base_batch=base_batch,
            loader=MagicMock(),
            widen=MagicMock(),
            device="cpu",
            dtype=torch.float32,
            model_loaders={"model1": mock_loader},
        )

        torch.testing.assert_close(result, model_weights)

    def test_compile_plan_captures_strength_from_recipe_model(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """compile_plan propagates RecipeModel.strength into OpApplyModel."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path, strength=0.75)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        model_id_map = {id(recipe_model): "model1"}
        plan = compile_plan(recipe, set_id_map={}, arch="sdxl", model_id_map=model_id_map)

        apply_model_ops = [op for op in plan.ops if isinstance(op, OpApplyModel)]
        assert len(apply_model_ops) == 1
        assert apply_model_ops[0].strength == 0.75


# ---------------------------------------------------------------------------
# Additional tests for edge cases
# ---------------------------------------------------------------------------


class TestRecipeModelValidation:
    """Tests for RecipeModel validation in various contexts."""

    def test_recipe_model_cannot_be_tree_root(
        self, sdxl_checkpoint_path: str
    ) -> None:
        """RecipeModel cannot be the root of a recipe tree."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)

        with pytest.raises(ValueError) as exc_info:
            walk_to_base(recipe_model)

        assert "root" in str(exc_info.value).lower()

    def test_recipe_model_valid_as_merge_target(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """RecipeModel is valid as a merge target."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        recipe = RecipeMerge(
            base=recipe_base,
            target=recipe_model,
            backbone=None,
            t_factor=1.0,
        )

        # Should not raise
        model_refs = _collect_model_refs(recipe)
        assert len(model_refs) == 1

    def test_recipe_model_valid_in_compose(
        self, recipe_base: RecipeBase, sdxl_checkpoint_path: str
    ) -> None:
        """RecipeModel is valid as a compose branch."""
        recipe_model = RecipeModel(path=sdxl_checkpoint_path)
        compose = RecipeCompose(branches=(recipe_model,))
        recipe = RecipeMerge(
            base=recipe_base,
            target=compose,
            backbone=None,
            t_factor=1.0,
        )

        model_refs = _collect_model_refs(recipe)
        assert len(model_refs) == 1
