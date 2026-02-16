"""Tests for LoRA set scoping fix — @fix-lora-set-scoping.

Verifies that LoRA deltas are correctly scoped by set_id so that
compose branches with overlapping keys produce distinct results,
and missing set_id raises an explicit error.

AC-1: Two LoRA sets affecting the same key produce distinct branch results
AC-2: Missing set_id in set_id_map raises RuntimeError, not silent no-op
AC-3: get_delta_specs(set_id=X) returns only deltas from set X
AC-4: analyze_recipe wires set_id through loader.load() and set_affected
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.executor import apply_lora_batch_gpu, evaluate_recipe
from lib.lora.sdxl import SDXLLoader
from lib.lora.zimage import ZImageLoader
from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def overlapping_sdxl_lora_a() -> str:
    """SDXL LoRA file A affecting input_blocks.0.0.weight."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        # Deterministic tensors for reproducible assertions
        torch.manual_seed(42)
        tensors = {
            "lora_unet_input_blocks_0_0.lora_up.weight": torch.randn(64, 8),
            "lora_unet_input_blocks_0_0.lora_down.weight": torch.randn(8, 32),
        }
        save_file(tensors, f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def overlapping_sdxl_lora_b() -> str:
    """SDXL LoRA file B affecting the SAME key (input_blocks.0.0.weight)
    but with different tensors and strength."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        torch.manual_seed(99)
        tensors = {
            "lora_unet_input_blocks_0_0.lora_up.weight": torch.randn(64, 4),
            "lora_unet_input_blocks_0_0.lora_down.weight": torch.randn(4, 32),
        }
        save_file(tensors, f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-1: Two LoRA sets affecting the same key produce distinct branch results
# ---------------------------------------------------------------------------


class TestSetScopingCorrectness:
    """AC: @fix-lora-set-scoping ac-1

    Given: two LoRA sets (A and B) both affecting the same model key
    When: get_delta_specs is called with set_id for each set
    Then: set A returns only A's deltas, set B returns only B's deltas
    """

    def test_sdxl_overlapping_keys_produce_distinct_deltas(
        self, overlapping_sdxl_lora_a: str, overlapping_sdxl_lora_b: str
    ):
        """Two SDXL LoRAs on same key produce different deltas per set.

        # AC: @fix-lora-set-scoping ac-1
        """
        loader = SDXLLoader()

        # Load into separate sets
        loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")
        loader.load(overlapping_sdxl_lora_b, strength=0.5, set_id="set_b")

        key = "diffusion_model.input_blocks.0.0.weight"
        assert key in loader.affected_keys
        assert key in loader.affected_keys_for_set("set_a")
        assert key in loader.affected_keys_for_set("set_b")

        keys = [key]
        key_indices = {key: 0}

        # Get deltas scoped to each set
        specs_a = loader.get_delta_specs(keys, key_indices, set_id="set_a")
        specs_b = loader.get_delta_specs(keys, key_indices, set_id="set_b")

        assert len(specs_a) == 1, f"Set A should have 1 spec, got {len(specs_a)}"
        assert len(specs_b) == 1, f"Set B should have 1 spec, got {len(specs_b)}"

        # Deltas should be different (different LoRA files)
        delta_a = specs_a[0].scale * (specs_a[0].up @ specs_a[0].down)
        delta_b = specs_b[0].scale * (specs_b[0].up @ specs_b[0].down)
        assert not torch.allclose(delta_a, delta_b), (
            "Set A and Set B should produce different deltas"
        )

        # Scales should differ (A=1.0, B=0.5)
        assert specs_a[0].scale != specs_b[0].scale

        loader.cleanup()

    def test_unscoped_get_delta_specs_returns_all(
        self, overlapping_sdxl_lora_a: str, overlapping_sdxl_lora_b: str
    ):
        """Calling get_delta_specs without set_id returns deltas from all sets.

        # AC: @fix-lora-set-scoping ac-1
        """
        loader = SDXLLoader()
        loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")
        loader.load(overlapping_sdxl_lora_b, strength=0.5, set_id="set_b")

        key = "diffusion_model.input_blocks.0.0.weight"
        keys = [key]
        key_indices = {key: 0}

        # Without set_id: should return specs from BOTH sets
        specs_all = loader.get_delta_specs(keys, key_indices, set_id=None)
        assert len(specs_all) == 2, f"Unscoped should return 2 specs, got {len(specs_all)}"

        loader.cleanup()

    def test_compose_branches_get_distinct_results(
        self, overlapping_sdxl_lora_a: str, overlapping_sdxl_lora_b: str
    ):
        """Compose branches with overlapping keys produce distinct merged weights.

        # AC: @fix-lora-set-scoping ac-1
        This is the end-to-end test for the core bug: two branches in a
        RecipeCompose that both affect the same key should apply different
        deltas (scoped to their own LoRA set), not the combined deltas
        of all loaded LoRAs.
        """
        loader = SDXLLoader()
        loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")
        loader.load(overlapping_sdxl_lora_b, strength=0.5, set_id="set_b")

        key = "diffusion_model.input_blocks.0.0.weight"
        keys = [key]
        key_indices = {key: 0}
        base_weight = torch.randn(1, 64, 32)

        # Simulate what evaluate_recipe does for each branch:
        # Branch A: apply only set_a's deltas
        specs_a = loader.get_delta_specs(keys, key_indices, set_id="set_a")
        result_a = apply_lora_batch_gpu(keys, base_weight.clone(), specs_a, "cpu", torch.float32)

        # Branch B: apply only set_b's deltas
        specs_b = loader.get_delta_specs(keys, key_indices, set_id="set_b")
        result_b = apply_lora_batch_gpu(keys, base_weight.clone(), specs_b, "cpu", torch.float32)

        # Results should differ because different LoRAs were applied
        assert not torch.allclose(result_a, result_b), (
            "Branch A and B should produce different results"
        )

        # Each should differ from base (deltas were applied)
        assert not torch.allclose(result_a, base_weight), "Branch A should differ from base"
        assert not torch.allclose(result_b, base_weight), "Branch B should differ from base"

        loader.cleanup()

    def test_zimage_overlapping_keys_produce_distinct_deltas(self):
        """Z-Image loader also correctly scopes by set_id.

        # AC: @fix-lora-set-scoping ac-1
        """
        # Create two Z-Image LoRA files affecting the same key
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fa:
            torch.manual_seed(42)
            save_file(
                {
                    "transformer.layers.0.ff.linear_1.lora_A.weight": torch.randn(8, 64),
                    "transformer.layers.0.ff.linear_1.lora_B.weight": torch.randn(128, 8),
                },
                fa.name,
            )
            path_a = fa.name

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fb:
            torch.manual_seed(99)
            save_file(
                {
                    "transformer.layers.0.ff.linear_1.lora_A.weight": torch.randn(4, 64),
                    "transformer.layers.0.ff.linear_1.lora_B.weight": torch.randn(128, 4),
                },
                fb.name,
            )
            path_b = fb.name

        try:
            loader = ZImageLoader()
            loader.load(path_a, strength=1.0, set_id="set_a")
            loader.load(path_b, strength=0.5, set_id="set_b")

            key = "diffusion_model.layers.0.ff.linear_1.weight"
            assert key in loader.affected_keys
            assert key in loader.affected_keys_for_set("set_a")
            assert key in loader.affected_keys_for_set("set_b")

            keys = [key]
            key_indices = {key: 0}

            specs_a = loader.get_delta_specs(keys, key_indices, set_id="set_a")
            specs_b = loader.get_delta_specs(keys, key_indices, set_id="set_b")

            assert len(specs_a) == 1
            assert len(specs_b) == 1

            delta_a = specs_a[0].scale * (specs_a[0].up @ specs_a[0].down)
            delta_b = specs_b[0].scale * (specs_b[0].up @ specs_b[0].down)
            assert not torch.allclose(delta_a, delta_b)

            loader.cleanup()
        finally:
            Path(path_a).unlink(missing_ok=True)
            Path(path_b).unlink(missing_ok=True)

    def test_zimage_qkv_scoped_by_set(self):
        """Z-Image QKV deltas are also correctly scoped by set_id.

        # AC: @fix-lora-set-scoping ac-1
        """
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fa:
            torch.manual_seed(42)
            save_file(
                {
                    "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 64),
                    "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 8),
                },
                fa.name,
            )
            path_a = fa.name

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as fb:
            torch.manual_seed(99)
            save_file(
                {
                    "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(4, 64),
                    "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(32, 4),
                },
                fb.name,
            )
            path_b = fb.name

        try:
            loader = ZImageLoader()
            loader.load(path_a, strength=1.0, set_id="set_a")
            loader.load(path_b, strength=0.5, set_id="set_b")

            key = "diffusion_model.layers.0.attention.qkv.weight"
            keys = [key]
            key_indices = {key: 0}

            specs_a = loader.get_delta_specs(keys, key_indices, set_id="set_a")
            specs_b = loader.get_delta_specs(keys, key_indices, set_id="set_b")

            assert len(specs_a) == 1
            assert len(specs_b) == 1
            assert specs_a[0].kind == "qkv_q"
            assert specs_b[0].kind == "qkv_q"

            delta_a = specs_a[0].scale * (specs_a[0].up @ specs_a[0].down)
            delta_b = specs_b[0].scale * (specs_b[0].up @ specs_b[0].down)
            assert not torch.allclose(delta_a, delta_b)

            loader.cleanup()
        finally:
            Path(path_a).unlink(missing_ok=True)
            Path(path_b).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-2: Missing set_id raises RuntimeError
# ---------------------------------------------------------------------------


class TestMissingSetIdError:
    """AC: @fix-lora-set-scoping ac-2

    Given: a RecipeLoRA node not registered in set_id_map
    When: evaluate_recipe encounters it
    Then: RuntimeError is raised with a descriptive message (not silent no-op)
    """

    def test_missing_set_id_raises_runtime_error(self):
        """Unregistered RecipeLoRA raises RuntimeError.

        # AC: @fix-lora-set-scoping ac-2
        """
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            t_factor = 1.0

            def filter_delta_batched(self, lora_applied, backbone):
                return lora_applied

        loader = MockLoader()
        widen = MockWIDEN()
        # Empty set_id_map — lora is not registered
        set_id_map = {}

        with pytest.raises(RuntimeError, match="RecipeLoRA has no set_id mapping"):
            evaluate_recipe(
                keys=["k0"],
                base_batch=base_batch,
                recipe_node=merge,
                loader=loader,
                widen=widen,
                set_id_map=set_id_map,
                device="cpu",
                dtype=torch.float32,
            )

    def test_error_includes_lora_info(self):
        """RuntimeError message includes LoRA path info for debugging.

        # AC: @fix-lora-set-scoping ac-2
        """
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "my_lora.safetensors", "strength": 0.75},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            t_factor = 1.0

            def filter_delta_batched(self, lora_applied, backbone):
                return lora_applied

        with pytest.raises(RuntimeError) as exc_info:
            evaluate_recipe(
                keys=["k0"],
                base_batch=base_batch,
                recipe_node=merge,
                loader=MockLoader(),
                widen=MockWIDEN(),
                set_id_map={},
                device="cpu",
                dtype=torch.float32,
            )

        error_msg = str(exc_info.value)
        assert "my_lora.safetensors" in error_msg
        assert "0.75" in error_msg

    def test_one_registered_one_missing_raises_for_missing(self):
        """In a compose, one registered and one missing still raises.

        # AC: @fix-lora-set-scoping ac-2
        """
        batch_size = 1
        base_batch = torch.randn(batch_size, 4, 4)

        base = RecipeBase(model_patcher=None, arch="sdxl")
        lora_registered = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_missing = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 1.0},))
        compose = RecipeCompose(branches=(lora_registered, lora_missing))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        class MockLoader:
            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

        class MockWIDEN:
            t_factor = 1.0

            def filter_delta_batched(self, lora_applied, backbone):
                return lora_applied

            def merge_weights_batched(self, weights_list, backbone):
                return torch.stack(weights_list).mean(dim=0)

        # Only register one of the two LoRAs
        set_id_map = {id(lora_registered): "set1"}

        with pytest.raises(RuntimeError, match="RecipeLoRA has no set_id mapping"):
            evaluate_recipe(
                keys=["k0"],
                base_batch=base_batch,
                recipe_node=merge,
                loader=MockLoader(),
                widen=MockWIDEN(),
                set_id_map=set_id_map,
                device="cpu",
                dtype=torch.float32,
            )


# ---------------------------------------------------------------------------
# AC-3: get_delta_specs scoped by set_id
# ---------------------------------------------------------------------------


class TestGetDeltaSpecsScoping:
    """AC: @fix-lora-set-scoping ac-3

    Given: a loader with multiple sets loaded
    When: get_delta_specs is called with a specific set_id
    Then: only deltas from that set are returned
    """

    def test_nonexistent_set_returns_empty(
        self, overlapping_sdxl_lora_a: str
    ):
        """Querying a non-existent set returns no specs.

        # AC: @fix-lora-set-scoping ac-3
        """
        loader = SDXLLoader()
        loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")

        key = "diffusion_model.input_blocks.0.0.weight"
        keys = [key]
        key_indices = {key: 0}

        specs = loader.get_delta_specs(keys, key_indices, set_id="nonexistent")
        assert len(specs) == 0

        loader.cleanup()

    def test_affected_keys_for_set_returns_correct_keys(
        self, overlapping_sdxl_lora_a: str, overlapping_sdxl_lora_b: str
    ):
        """affected_keys_for_set returns only keys modified by that set.

        # AC: @fix-lora-set-scoping ac-3
        """
        # Create a LoRA B that affects a different key
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "lora_unet_middle_block_0.lora_up.weight": torch.randn(128, 16),
                "lora_unet_middle_block_0.lora_down.weight": torch.randn(16, 64),
            }
            save_file(tensors, f.name)
            path_b_different = f.name

        try:
            loader = SDXLLoader()
            loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")
            loader.load(path_b_different, strength=0.5, set_id="set_b")

            keys_a = loader.affected_keys_for_set("set_a")
            keys_b = loader.affected_keys_for_set("set_b")

            # set_a has input_blocks key, set_b has middle_block key
            assert "diffusion_model.input_blocks.0.0.weight" in keys_a
            assert "diffusion_model.input_blocks.0.0.weight" not in keys_b
            assert "diffusion_model.middle_block.0.weight" in keys_b
            assert "diffusion_model.middle_block.0.weight" not in keys_a

            # Global affected_keys is the union
            assert loader.affected_keys == keys_a | keys_b

            loader.cleanup()
        finally:
            Path(path_b_different).unlink(missing_ok=True)

    def test_cleanup_clears_all_set_data(
        self, overlapping_sdxl_lora_a: str
    ):
        """cleanup() clears all per-set data.

        # AC: @fix-lora-set-scoping ac-3
        """
        loader = SDXLLoader()
        loader.load(overlapping_sdxl_lora_a, strength=1.0, set_id="set_a")
        assert len(loader.affected_keys) > 0
        assert len(loader.affected_keys_for_set("set_a")) > 0

        loader.cleanup()

        assert len(loader.affected_keys) == 0
        assert len(loader.affected_keys_for_set("set_a")) == 0

    def test_default_set_id_when_none_provided(
        self, overlapping_sdxl_lora_a: str
    ):
        """load() with set_id=None uses __default__ internally.

        # AC: @fix-lora-set-scoping ac-3
        Backward compatibility: existing code that doesn't pass set_id
        still works, data goes into __default__ set.
        """
        loader = SDXLLoader()
        loader.load(overlapping_sdxl_lora_a, strength=1.0)  # No set_id

        key = "diffusion_model.input_blocks.0.0.weight"
        keys = [key]
        key_indices = {key: 0}

        # Without set_id filter, returns all (backward compatible)
        specs = loader.get_delta_specs(keys, key_indices)
        assert len(specs) == 1

        # With __default__ set_id filter
        specs_default = loader.get_delta_specs(keys, key_indices, set_id="__default__")
        assert len(specs_default) == 1

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-4: analyze_recipe wires set_id correctly
# ---------------------------------------------------------------------------


class TestAnalysisSetWiring:
    """AC: @fix-lora-set-scoping ac-4

    Given: analyze_recipe loads multiple LoRA sets
    When: set_affected is computed
    Then: each set_id maps to only the keys from its LoRA files
    """

    def test_analysis_set_affected_is_per_set(
        self, overlapping_sdxl_lora_a: str, overlapping_sdxl_lora_b: str
    ):
        """analyze_recipe's set_affected correctly scopes keys per set.

        # AC: @fix-lora-set-scoping ac-4
        """
        from lib.analysis import analyze_recipe
        from tests.conftest import MockModelPatcher

        temp_dir = str(Path(overlapping_sdxl_lora_a).parent)

        mock_patcher = MockModelPatcher()
        base = RecipeBase(model_patcher=mock_patcher, arch="sdxl")

        lora_a = RecipeLoRA(
            loras=({"path": Path(overlapping_sdxl_lora_a).name, "strength": 1.0},)
        )
        lora_b = RecipeLoRA(
            loras=({"path": Path(overlapping_sdxl_lora_b).name, "strength": 0.5},)
        )
        compose = RecipeCompose(branches=(lora_a, lora_b))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=1.0)

        import os

        result = analyze_recipe(
            merge,
            lora_path_resolver=lambda name: os.path.join(temp_dir, name),
        )

        # Should have 2 sets
        assert len(result.set_affected) == 2

        # Both sets should contain the overlapping key
        overlapping_key = "diffusion_model.input_blocks.0.0.weight"
        for set_id, keys in result.set_affected.items():
            assert overlapping_key in keys, (
                f"Set {set_id} should include {overlapping_key}"
            )

        # The loader should have set-scoped data
        key_indices = {overlapping_key: 0}
        for set_key in result.set_affected:
            specs = result.loader.get_delta_specs(
                [overlapping_key], key_indices, set_id=set_key
            )
            assert len(specs) == 1, (
                f"Set {set_key} should have exactly 1 spec for the overlapping key"
            )

        result.loader.cleanup()
