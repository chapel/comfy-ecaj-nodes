"""Tests for Incremental Block Recomputation — @incremental-block-recompute.

Covers all 16 acceptance criteria with unit tests for library functions
and integration tests for the exit node cache.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lib.block_classify import (
    compute_changed_blocks,
    filter_changed_keys,
)
from lib.persistence import (
    collect_block_configs,
    compute_structural_fingerprint,
    serialize_recipe,
)
from lib.recipe import (
    BlockConfig,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
)
from lib.recipe_eval import EvalPlan
from nodes.exit import (
    WIDENExitNode,
    _CacheEntry,
    _incremental_cache,
    clear_incremental_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SDXL_KEYS = (
    "diffusion_model.input_blocks.0.0.weight",
    "diffusion_model.input_blocks.5.0.weight",
    "diffusion_model.middle_block.0.weight",
    "diffusion_model.output_blocks.0.0.weight",
    "diffusion_model.output_blocks.3.0.weight",
    "diffusion_model.time_embed.0.weight",
)


def _base_identity() -> str:
    return "base_identity_hash_abc123"


def _lora_stats() -> dict[str, tuple[float, int]]:
    return {"lora_a.safetensors": (1000.0, 5000)}


def _make_recipe(
    block_config_lora=None,
    block_config_merge=None,
    block_config_model=None,
    t_factor=1.0,
    lora_path="lora_a.safetensors",
    lora_strength=1.0,
    include_model=False,
):
    """Build a minimal recipe tree for testing."""
    patcher = MagicMock()
    base = RecipeBase(model_patcher=patcher, arch="sdxl")
    lora = RecipeLoRA(
        loras=({"path": lora_path, "strength": lora_strength},),
        block_config=block_config_lora,
    )
    if include_model:
        model = RecipeModel(
            path="model_b.safetensors", strength=0.8,
            block_config=block_config_model,
        )
        compose = RecipeCompose(branches=(lora, model))
        return RecipeMerge(
            base=base, target=compose, backbone=None,
            t_factor=t_factor, block_config=block_config_merge,
        )
    return RecipeMerge(
        base=base, target=lora, backbone=None,
        t_factor=t_factor, block_config=block_config_merge,
    )


# ===========================================================================
# Unit tests: compute_structural_fingerprint
# ===========================================================================


class TestStructuralFingerprint:
    """Tests for compute_structural_fingerprint."""

    def test_identical_recipes_same_fingerprint(self):
        """Same recipe tree → same fingerprint."""
        r1 = _make_recipe()
        r2 = _make_recipe()
        fp1 = compute_structural_fingerprint(r1, _base_identity(), _lora_stats())
        fp2 = compute_structural_fingerprint(r2, _base_identity(), _lora_stats())
        assert fp1 == fp2

    def test_different_block_config_same_fingerprint(self):
        """Recipes differing only in block_config → same fingerprint."""
        bc_a = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        bc_b = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.9),))
        r1 = _make_recipe(block_config_lora=bc_a)
        r2 = _make_recipe(block_config_lora=bc_b)
        fp1 = compute_structural_fingerprint(r1, _base_identity(), _lora_stats())
        fp2 = compute_structural_fingerprint(r2, _base_identity(), _lora_stats())
        assert fp1 == fp2

    # AC: @incremental-block-recompute ac-4
    def test_different_t_factor_different_fingerprint(self):
        """Different t_factor → different fingerprint."""
        r1 = _make_recipe(t_factor=1.0)
        r2 = _make_recipe(t_factor=0.5)
        fp1 = compute_structural_fingerprint(r1, _base_identity(), _lora_stats())
        fp2 = compute_structural_fingerprint(r2, _base_identity(), _lora_stats())
        assert fp1 != fp2

    # AC: @incremental-block-recompute ac-4
    def test_different_lora_path_different_fingerprint(self):
        """Different LoRA path → different fingerprint."""
        r1 = _make_recipe(lora_path="lora_a.safetensors")
        r2 = _make_recipe(lora_path="lora_b.safetensors")
        fp1 = compute_structural_fingerprint(r1, _base_identity(), _lora_stats())
        fp2 = compute_structural_fingerprint(r2, _base_identity(), _lora_stats())
        assert fp1 != fp2

    # AC: @incremental-block-recompute ac-13
    def test_different_base_identity_different_fingerprint(self):
        """Different base_identity → different fingerprint."""
        r = _make_recipe()
        fp1 = compute_structural_fingerprint(r, "identity_A", _lora_stats())
        fp2 = compute_structural_fingerprint(r, "identity_B", _lora_stats())
        assert fp1 != fp2

    # AC: @incremental-block-recompute ac-14
    def test_different_file_stats_different_fingerprint(self):
        """Different file stats (mtime/size) → different fingerprint."""
        r = _make_recipe()
        stats1 = {"lora_a.safetensors": (1000.0, 5000)}
        stats2 = {"lora_a.safetensors": (2000.0, 5000)}
        fp1 = compute_structural_fingerprint(r, _base_identity(), stats1)
        fp2 = compute_structural_fingerprint(r, _base_identity(), stats2)
        assert fp1 != fp2

    def test_strip_block_config_in_serialization(self):
        """serialize_recipe with strip_block_config excludes block_config."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        r_with = _make_recipe(block_config_lora=bc)
        r_without = _make_recipe()
        s1 = serialize_recipe(
            r_with, _base_identity(), _lora_stats(), strip_block_config=True
        )
        s2 = serialize_recipe(
            r_without, _base_identity(), _lora_stats(), strip_block_config=True
        )
        assert s1 == s2


# ===========================================================================
# Unit tests: collect_block_configs
# ===========================================================================


class TestCollectBlockConfigs:
    """Tests for collect_block_configs."""

    def test_no_block_configs(self):
        """Recipe with no block_config → all entries are (path, None)."""
        recipe = _make_recipe()
        configs = collect_block_configs(recipe)
        assert len(configs) > 0
        for path, bc in configs:
            assert bc is None

    def test_lora_block_config_collected(self):
        """RecipeLoRA block_config is collected."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = _make_recipe(block_config_lora=bc)
        configs = collect_block_configs(recipe)
        lora_configs = [(p, b) for p, b in configs if b is not None]
        assert len(lora_configs) == 1
        assert lora_configs[0][1] is bc

    def test_merge_block_config_collected(self):
        """RecipeMerge block_config is collected."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.3),))
        recipe = _make_recipe(block_config_merge=bc)
        configs = collect_block_configs(recipe)
        # Root merge should have the block_config
        merge_configs = [(p, b) for p, b in configs if b is not None]
        assert len(merge_configs) == 1
        assert merge_configs[0][1] is bc

    # AC: @incremental-block-recompute ac-5
    def test_multiple_block_configs(self):
        """Multiple positions with block_config are all collected."""
        bc_lora = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        bc_merge = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.3),))
        recipe = _make_recipe(block_config_lora=bc_lora, block_config_merge=bc_merge)
        configs = collect_block_configs(recipe)
        non_none = [(p, b) for p, b in configs if b is not None]
        assert len(non_none) == 2

    def test_deterministic_order(self):
        """Pre-order traversal is deterministic."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = _make_recipe(block_config_lora=bc, block_config_merge=None)
        c1 = collect_block_configs(recipe)
        c2 = collect_block_configs(recipe)
        assert c1 == c2

    def test_model_block_config_collected(self):
        """RecipeModel block_config is collected."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("OUT03", 0.5),))
        recipe = _make_recipe(include_model=True, block_config_model=bc)
        configs = collect_block_configs(recipe)
        model_configs = [
            (p, b) for p, b in configs if b is not None and "OUT03" in str(b)
        ]
        assert len(model_configs) == 1


# ===========================================================================
# Unit tests: compute_changed_blocks
# ===========================================================================


class TestComputeChangedBlocks:
    """Tests for compute_changed_blocks."""

    def test_identical_configs_no_changes(self):
        """Identical configs → empty changed sets."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        configs = [("root", bc), ("root.target", None)]
        result = compute_changed_blocks(configs, configs, "sdxl")
        assert result is not None
        changed_blocks, changed_layer_types = result
        assert changed_blocks == set()
        assert changed_layer_types == set()

    # AC: @incremental-block-recompute ac-3
    def test_single_block_change(self):
        """One block override value changed → that block in changed set."""
        bc_old = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5), ("MID", 1.0)))
        bc_new = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.7), ("MID", 1.0)))
        old_configs = [("root", bc_old)]
        new_configs = [("root", bc_new)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, changed_layer_types = result
        assert changed_blocks == {"IN00"}
        assert changed_layer_types == set()

    # AC: @incremental-block-recompute ac-6
    def test_layer_type_change(self):
        """Layer type override value changed → that type in changed set."""
        bc_old = BlockConfig(
            arch="sdxl",
            block_overrides=(),
            layer_type_overrides=(("attention", 0.5), ("feed_forward", 1.0)),
        )
        bc_new = BlockConfig(
            arch="sdxl",
            block_overrides=(),
            layer_type_overrides=(("attention", 0.8), ("feed_forward", 1.0)),
        )
        old_configs = [("root", bc_old)]
        new_configs = [("root", bc_new)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, changed_layer_types = result
        assert changed_blocks == set()
        assert changed_layer_types == {"attention"}

    # AC: @incremental-block-recompute ac-7
    def test_none_to_present_returns_none(self):
        """Block config added (None → present) → structural mismatch."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        old_configs = [("root", None)]
        new_configs = [("root", bc)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is None

    # AC: @incremental-block-recompute ac-8
    def test_present_to_none_returns_none(self):
        """Block config removed (present → None) → structural mismatch."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        old_configs = [("root", bc)]
        new_configs = [("root", None)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is None

    def test_different_path_count_returns_none(self):
        """Different number of config positions → structural mismatch."""
        bc = BlockConfig(arch="sdxl", block_overrides=())
        old_configs = [("root", bc)]
        new_configs = [("root", bc), ("root.target", None)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is None

    def test_different_paths_returns_none(self):
        """Different position paths → structural mismatch."""
        bc = BlockConfig(arch="sdxl", block_overrides=())
        old_configs = [("root.target", bc)]
        new_configs = [("root.base", bc)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is None

    def test_block_added_to_overrides(self):
        """New block override added → that block in changed set."""
        bc_old = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        bc_new = BlockConfig(
            arch="sdxl",
            block_overrides=(("IN00", 0.5), ("MID", 0.3)),
        )
        old_configs = [("root", bc_old)]
        new_configs = [("root", bc_new)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, _ = result
        assert changed_blocks == {"MID"}

    def test_both_none_no_change(self):
        """Both configs None → no change."""
        old_configs = [("root", None)]
        new_configs = [("root", None)]
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, changed_layer_types = result
        assert changed_blocks == set()
        assert changed_layer_types == set()


# ===========================================================================
# Unit tests: filter_changed_keys
# ===========================================================================


class TestFilterChangedKeys:
    """Tests for filter_changed_keys."""

    # AC: @incremental-block-recompute ac-3
    def test_filter_by_block_group(self):
        """Keys in changed block group are included."""
        keys = set(_SDXL_KEYS)
        changed_blocks = {"IN00"}
        result = filter_changed_keys(keys, changed_blocks, set(), "sdxl")
        # Only IN00 keys + unclassified keys
        for k in result:
            from lib.block_classify import classify_key
            block = classify_key(k, "sdxl")
            assert block == "IN00" or block is None

    # AC: @incremental-block-recompute ac-6
    def test_filter_by_layer_type(self):
        """Keys matching changed layer type are included."""
        keys = {
            "diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_q.weight",
            "diffusion_model.input_blocks.0.1.transformer_blocks.0.ff.net.0.proj.weight",
            "diffusion_model.input_blocks.0.0.weight",
        }
        result = filter_changed_keys(keys, set(), {"attention"}, "sdxl")
        assert "diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_q.weight" in result

    # AC: @incremental-block-recompute ac-11
    def test_unclassified_keys_included(self):
        """Keys with classify_key→None are included conservatively."""
        keys = set(_SDXL_KEYS) | {"diffusion_model.input_blocks.9.0.weight"}
        changed_blocks = {"IN05"}
        result = filter_changed_keys(keys, changed_blocks, set(), "sdxl")
        # input_blocks.9 is unclassified (SDXL only covers 0-8), should be included
        assert "diffusion_model.input_blocks.9.0.weight" in result

    def test_classified_structural_keys_not_spuriously_included(self):
        """Structural keys are classified; only included when their block changes."""
        keys = set(_SDXL_KEYS)
        changed_blocks = {"IN05"}
        result = filter_changed_keys(keys, changed_blocks, set(), "sdxl")
        # time_embed is now classified as TIME_EMBED, not in changed_blocks → excluded
        assert "diffusion_model.time_embed.0.weight" not in result

        # But if TIME_EMBED is in changed_blocks, it IS included
        result2 = filter_changed_keys(keys, {"TIME_EMBED"}, set(), "sdxl")
        assert "diffusion_model.time_embed.0.weight" in result2

    def test_no_changes_empty_result(self):
        """No changed blocks or layer types → only unclassified keys."""
        keys = {
            "diffusion_model.input_blocks.0.0.weight",
            "diffusion_model.middle_block.0.weight",
        }
        result = filter_changed_keys(keys, set(), set(), "sdxl")
        # No unclassified keys in this set, so result should be empty
        assert result == set()

    def test_combined_block_and_layer_type(self):
        """Both block and layer type changes filter correctly."""
        keys = {
            "diffusion_model.input_blocks.0.0.weight",  # IN00
            "diffusion_model.middle_block.0.weight",  # MID
            "diffusion_model.output_blocks.0.0.weight",  # OUT00
        }
        changed_blocks = {"IN00"}
        changed_layer_types = set()  # no layer type changes
        result = filter_changed_keys(keys, changed_blocks, changed_layer_types, "sdxl")
        assert "diffusion_model.input_blocks.0.0.weight" in result
        assert "diffusion_model.middle_block.0.weight" not in result


# ===========================================================================
# Integration tests: _CacheEntry and _incremental_cache
# ===========================================================================


class TestCacheEntry:
    """Tests for _CacheEntry and module-level cache."""

    def setup_method(self):
        _incremental_cache.clear()

    def teardown_method(self):
        _incremental_cache.clear()

    # AC: @incremental-block-recompute ac-12
    def test_clear_incremental_cache(self):
        """clear_incremental_cache empties the cache."""
        _incremental_cache["test"] = _CacheEntry(
            structural_fingerprint="test",
            block_configs=[],
            merged_state={},
            storage_dtype=torch.float32,
        )
        assert len(_incremental_cache) == 1
        clear_incremental_cache()
        assert len(_incremental_cache) == 0

    # AC: @incremental-block-recompute ac-9
    def test_cache_lru_1(self):
        """Cache stores at most one entry."""
        entry1 = _CacheEntry(
            structural_fingerprint="fp1",
            block_configs=[],
            merged_state={},
            storage_dtype=torch.float32,
        )
        entry2 = _CacheEntry(
            structural_fingerprint="fp2",
            block_configs=[],
            merged_state={},
            storage_dtype=torch.float32,
        )
        _incremental_cache["fp1"] = entry1
        # Simulate LRU-1 eviction (as done in exit.py)
        _incremental_cache.clear()
        _incremental_cache["fp2"] = entry2
        assert len(_incremental_cache) == 1
        assert "fp2" in _incremental_cache
        assert "fp1" not in _incremental_cache


# ===========================================================================
# Integration tests: exit node with incremental cache
# ===========================================================================


def _make_exit_mocks(
    mock_model_patcher,
    keys_to_process,
    recipe=None,
    arch="sdxl",
):
    """Create all the mocks needed for WIDENExitNode.execute()."""
    mock_loader = MagicMock()
    mock_loader.affected_keys = set(keys_to_process)
    mock_loader.affected_keys_for_set = MagicMock(return_value=set(keys_to_process))
    mock_loader.cleanup = MagicMock()
    mock_loader.loaded_bytes = 0

    # Build set_affected from actual RecipeLoRA objects in the recipe tree
    set_affected = {}
    if recipe is not None:
        def _find_loras(n):
            if isinstance(n, RecipeLoRA):
                key = str(id(n))
                set_affected[key] = set(keys_to_process)
            elif isinstance(n, RecipeCompose):
                for b in n.branches:
                    _find_loras(b)
            elif isinstance(n, RecipeMerge):
                _find_loras(n.base)
                _find_loras(n.target)
                if n.backbone is not None:
                    _find_loras(n.backbone)
        _find_loras(recipe)
    if not set_affected:
        set_affected = {str(id(None)): set(keys_to_process)}

    mock_analyze = MagicMock(
        model_patcher=mock_model_patcher,
        arch=arch,
        loader=mock_loader,
        set_affected=set_affected,
        affected_keys=set(keys_to_process),
    )

    mock_model_analysis = MagicMock()
    mock_model_analysis.model_loaders = {}
    mock_model_analysis.model_affected = {}
    mock_model_analysis.all_model_keys = frozenset()

    # Create a dummy EvalPlan
    dummy_plan = EvalPlan(ops=(), result_reg=0, dead_after=())

    return mock_analyze, mock_model_analysis, mock_loader, dummy_plan


class TestExitNodeIncrementalCache:
    """Integration tests for incremental cache in WIDENExitNode.execute()."""

    def setup_method(self):
        _incremental_cache.clear()

    def teardown_method(self):
        _incremental_cache.clear()

    # AC: @incremental-block-recompute ac-1
    def test_first_execution_stores_cache(self, mock_model_patcher):
        """First execution stores result in incremental cache."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )

        merged = {k: torch.randn(4, 4) for k in keys}
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        assert len(_incremental_cache) == 1
        entry = next(iter(_incremental_cache.values()))
        assert len(entry.merged_state) == len(keys)

    # AC: @incremental-block-recompute ac-2
    def test_identical_reexecution_uses_cache(self, mock_model_patcher):
        """Re-execution with identical recipe uses cache, no GPU compute.

        Verifies both control flow (no GPU call) and value correctness
        (output patches contain cached tensor values).
        """
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with known tensor values
        fp = "test_fingerprint"
        cached_state = {k: torch.randn(4, 4) for k in keys}
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe),
            merged_state={k: v.clone() for k, v in cached_state.items()},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )

        chunked_eval_mock = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.chunked_evaluation", chunked_eval_mock),
            patch("nodes.exit.compile_batch_groups", return_value={}),
        ):
            node = WIDENExitNode()
            (result,) = node.execute(recipe)

        # chunked_evaluation should NOT have been called (full cache hit)
        chunked_eval_mock.assert_not_called()

        # Value check: output patches contain cached tensor values
        for key in keys:
            assert key in result.patches, f"Missing patch for {key}"
            # Patches format: [(strength, ("set", (tensor,)), ...)]
            patch_entry = result.patches[key][-1]
            set_tuple = patch_entry[1]  # ("set", (tensor,))
            output_tensor = set_tuple[1][0]
            expected = cached_state[key].to(torch.float32)
            assert torch.equal(output_tensor, expected), (
                f"Value mismatch for {key}"
            )

    # AC: @incremental-block-recompute ac-3
    def test_single_block_change_partial_recompute(self, mock_model_patcher):
        """Changing one block override → only that block's keys recomputed."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc_old = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5), ("MID", 1.0)))
        bc_new = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.7), ("MID", 1.0)))

        recipe_old = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc_old,
            ),
            backbone=None,
            t_factor=1.0,
        )
        recipe_new = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc_new,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache
        fp = "test_fingerprint"
        cached_state = {k: torch.randn(4, 4) for k in keys}
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe_old),
            merged_state={k: v.clone() for k, v in cached_state.items()},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe_new,
        )

        batch_groups_calls = []

        def track_batch_groups(key_list, *args, **kwargs):
            batch_groups_calls.append(list(key_list))
            from lib.batch_groups import OpSignature
            sig = OpSignature(shape=(4, 4), ndim=2)
            return {sig: key_list}

        new_results = {k: torch.randn(4, 4) for k in keys}

        def mock_chunked_eval(keys, **kwargs):
            return {k: new_results[k] for k in keys}

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", side_effect=track_batch_groups),
            patch("nodes.exit.chunked_evaluation", side_effect=mock_chunked_eval),
        ):
            node = WIDENExitNode()
            node.execute(recipe_new)

        # compile_batch_groups is called twice: first with all keys, then with filtered
        # The second call (incremental) should only have IN00 keys + unclassified
        assert len(batch_groups_calls) == 2
        recomputed_keys = batch_groups_calls[1]  # second call = filtered
        from lib.block_classify import classify_key
        for k in recomputed_keys:
            block = classify_key(k, "sdxl")
            assert block in ("IN00", None), f"Key {k} classified as {block}, expected IN00 or None"

        # Value check: unchanged keys in cache should be preserved in output
        entry = next(iter(_incremental_cache.values()))
        for k in keys:
            block = classify_key(k, "sdxl")
            if block not in ("IN00", None):
                # Unchanged key — should match original cached value
                assert torch.equal(
                    entry.merged_state[k], cached_state[k]
                ), f"Unchanged key {k} should match cache"

    # AC: @incremental-block-recompute ac-4
    def test_structural_change_full_recompute(self, mock_model_patcher):
        """Structural recipe change → full recomputation, cache replaced."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with DIFFERENT fingerprint
        old_fp = "old_fingerprint"
        new_fp = "new_fingerprint"
        _incremental_cache[old_fp] = _CacheEntry(
            structural_fingerprint=old_fp,
            block_configs=[("root", bc)],
            merged_state={k: torch.randn(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )

        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=new_fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation",
                  return_value={k: torch.randn(4, 4) for k in keys}),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        # Old fingerprint should be evicted, new one stored
        assert old_fp not in _incremental_cache
        assert new_fp in _incremental_cache

    # AC: @incremental-block-recompute ac-9
    def test_cache_bounded_to_one_entry(self, mock_model_patcher):
        """Cache has at most one entry after any execution."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate with entry F1
        _incremental_cache["F1"] = _CacheEntry(
            structural_fingerprint="F1",
            block_configs=[],
            merged_state={},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value="F2"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation",
                  return_value={k: torch.randn(4, 4) for k in keys}),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        assert len(_incremental_cache) == 1
        assert "F2" in _incremental_cache

    # AC: @incremental-block-recompute ac-16
    def test_interrupt_preserves_cache(self, mock_model_patcher):
        """Exception during GPU eval preserves previous cache entry."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        fp = "preserved_fp"
        old_state = {k: torch.randn(4, 4) for k in keys}
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe),
            merged_state={k: v.clone() for k, v in old_state.items()},
            storage_dtype=torch.float32,
        )

        # Use a different fingerprint to force full recompute
        new_fp = "different_fp"
        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=new_fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", side_effect=RuntimeError("GPU OOM")),
        ):
            node = WIDENExitNode()
            with pytest.raises(RuntimeError, match="GPU OOM"):
                node.execute(recipe)

        # Old cache entry should still be there because the exception happens
        # before cache clear/store
        assert fp in _incremental_cache
        entry = _incremental_cache[fp]
        for k in keys:
            assert torch.equal(entry.merged_state[k], old_state[k])

    # AC: @incremental-block-recompute ac-10
    def test_save_model_with_partial_recompute(self, mock_model_patcher):
        """save_model=True with partial recompute saves complete state."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc_old = BlockConfig(
            arch="sdxl", block_overrides=(("IN00", 0.5), ("MID", 1.0)),
        )
        bc_new = BlockConfig(
            arch="sdxl", block_overrides=(("IN00", 0.7), ("MID", 1.0)),
        )

        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc_new,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with old block config
        fp = "test_fingerprint"
        cached_state = {k: torch.randn(4, 4) for k in keys}
        old_recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc_old,
            ),
            backbone=None,
            t_factor=1.0,
        )
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(old_recipe),
            merged_state={k: v.clone() for k, v in cached_state.items()},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )

        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        new_results = {k: torch.randn(4, 4) for k in keys}
        atomic_save_mock = MagicMock()

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models",
                  return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint",
                  return_value=fp),
            patch("nodes.exit.compute_base_identity",
                  return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups",
                  return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation",
                  return_value=new_results),
            patch("nodes.exit.validate_model_name",
                  return_value="test.safetensors"),
            patch("nodes.exit._resolve_checkpoints_path",
                  return_value="/tmp/test.safetensors"),
            patch("nodes.exit.serialize_recipe",
                  return_value='{"test": true}'),
            patch("nodes.exit.compute_recipe_hash",
                  return_value="recipe_hash"),
            patch("nodes.exit.check_cache", return_value=None),
            patch("nodes.exit.build_metadata",
                  return_value={"__ecaj_version__": "1"}),
            patch("nodes.exit.atomic_save", atomic_save_mock),
        ):
            node = WIDENExitNode()
            node.execute(
                recipe, save_model=True, model_name="test",
            )

        # atomic_save should have been called
        atomic_save_mock.assert_called_once()
        saved_state = atomic_save_mock.call_args[0][0]

        # Saved state should contain ALL keys (complete merged state)
        for k in keys:
            assert k in saved_state, (
                f"Key {k} missing from saved state"
            )


# ===========================================================================
# Unit tests for edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_collect_block_configs_recipe_base_only(self):
        """RecipeBase alone → empty list (no block_config fields)."""
        patcher = MagicMock()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        configs = collect_block_configs(base)
        assert configs == []

    def test_compute_changed_blocks_empty_configs(self):
        """Empty config lists → no changes."""
        result = compute_changed_blocks([], [], "sdxl")
        assert result is not None
        changed_blocks, changed_layer_types = result
        assert changed_blocks == set()
        assert changed_layer_types == set()

    def test_filter_changed_keys_empty_keys(self):
        """No keys → empty result."""
        result = filter_changed_keys(set(), {"IN00"}, set(), "sdxl")
        assert result == set()

    # AC: @incremental-block-recompute ac-15
    def test_recipe_model_block_config_change(self):
        """RecipeModel block_config change → affected block keys changed."""
        bc_old = BlockConfig(arch="sdxl", block_overrides=(("OUT03", 0.5),))
        bc_new = BlockConfig(arch="sdxl", block_overrides=(("OUT03", 0.8),))

        patcher = MagicMock()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        model_old = RecipeModel(path="m.safetensors", block_config=bc_old)
        model_new = RecipeModel(path="m.safetensors", block_config=bc_new)
        recipe_old = RecipeMerge(
            base=base, target=model_old, backbone=None, t_factor=1.0,
        )
        recipe_new = RecipeMerge(
            base=base, target=model_new, backbone=None, t_factor=1.0,
        )

        old_configs = collect_block_configs(recipe_old)
        new_configs = collect_block_configs(recipe_new)
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, _ = result
        assert changed_blocks == {"OUT03"}

    # AC: @incremental-block-recompute ac-5
    def test_multiple_block_config_positions_only_one_changes(self):
        """Multiple positions, only one changes → only affected blocks."""
        bc_lora_old = BlockConfig(arch="sdxl", block_overrides=(("IN05", 0.5),))
        bc_lora_new = BlockConfig(arch="sdxl", block_overrides=(("IN05", 0.9),))
        bc_merge = BlockConfig(arch="sdxl", block_overrides=(("MID", 0.3),))

        patcher = MagicMock()
        base = RecipeBase(model_patcher=patcher, arch="sdxl")
        recipe_old = RecipeMerge(
            base=base,
            target=RecipeLoRA(
                loras=({"path": "lora.safetensors", "strength": 1.0},),
                block_config=bc_lora_old,
            ),
            backbone=None, t_factor=1.0, block_config=bc_merge,
        )
        recipe_new = RecipeMerge(
            base=base,
            target=RecipeLoRA(
                loras=({"path": "lora.safetensors", "strength": 1.0},),
                block_config=bc_lora_new,
            ),
            backbone=None, t_factor=1.0, block_config=bc_merge,
        )

        old_configs = collect_block_configs(recipe_old)
        new_configs = collect_block_configs(recipe_new)
        result = compute_changed_blocks(old_configs, new_configs, "sdxl")
        assert result is not None
        changed_blocks, _ = result
        assert changed_blocks == {"IN05"}

    # AC: @incremental-block-recompute ac-17
    # AC: @memory-management ac-6
    def test_cache_entry_stores_references_not_clones(self):
        """Cache stores tensor references — no clone, no memory duplication."""
        tensor = torch.randn(4, 4)
        state = {"key": tensor}
        entry = _CacheEntry(
            structural_fingerprint="fp",
            block_configs=[],
            merged_state=state,
            storage_dtype=torch.float32,
        )
        # Cache holds the same tensor object — no clone
        assert entry.merged_state["key"] is tensor

    # AC: @incremental-block-recompute ac-17
    # AC: @memory-management ac-6
    def test_install_merged_patches_does_not_mutate_cached_tensors(self):
        """install_merged_patches is read-only — safe to alias with cache."""
        from nodes.exit import install_merged_patches

        state = {"diffusion_model.key": torch.randn(4, 4)}
        expected = state["diffusion_model.key"].clone()

        # Simulate cache + patch install sharing the same tensors
        entry = _CacheEntry(
            structural_fingerprint="fp",
            block_configs=[],
            merged_state=state,
            storage_dtype=torch.float32,
        )

        patcher = MagicMock()
        patcher.clone.return_value = MagicMock()
        install_merged_patches(patcher, state, torch.float32)

        # Cached tensor must be unmodified
        assert torch.equal(entry.merged_state["diffusion_model.key"], expected)

    # AC: @incremental-block-recompute ac-18
    def test_enable_cache_false_skips_lookup(self, mock_model_patcher):
        """enable_cache=False skips cache lookup, does full GPU compute."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache — should be ignored when enable_cache=False
        fp = "test_fingerprint"
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe),
            merged_state={k: torch.randn(4, 4) for k in keys},
            storage_dtype=torch.float32,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        chunked_eval_mock = MagicMock(
            return_value={k: torch.randn(4, 4) for k in keys}
        )

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", chunked_eval_mock),
        ):
            node = WIDENExitNode()
            node.execute(recipe, enable_cache=False)

        # GPU compute SHOULD have been called (cache was not used)
        chunked_eval_mock.assert_called_once()
        # Cache should be empty (evicted, not populated)
        assert len(_incremental_cache) == 0

    # AC: @incremental-block-recompute ac-18
    def test_enable_cache_false_evicts_existing(self, mock_model_patcher):
        """enable_cache=False evicts existing cache entries."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate with unrelated entry
        _incremental_cache["old_fp"] = _CacheEntry(
            structural_fingerprint="old_fp",
            block_configs=[],
            merged_state={},
            storage_dtype=torch.float32,
        )
        assert len(_incremental_cache) == 1

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value="new_fp"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation",
                  return_value={k: torch.randn(4, 4) for k in keys}),
        ):
            node = WIDENExitNode()
            node.execute(recipe, enable_cache=False)

        # All cache entries should be evicted
        assert len(_incremental_cache) == 0

    # AC: @incremental-block-recompute ac-18
    def test_enable_cache_true_populates_cache(self, mock_model_patcher):
        """enable_cache=True (default) populates cache as normal."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
            ),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value="fp"),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation",
                  return_value={k: torch.randn(4, 4) for k in keys}),
        ):
            node = WIDENExitNode()
            node.execute(recipe, enable_cache=True)

        assert len(_incremental_cache) == 1

    # AC: @incremental-block-recompute ac-17
    # AC: @memory-management ac-6
    def test_persistence_overlay_does_not_mutate_cached_tensors(self):
        """Persistence save path is read-only — aliased cache tensors survive."""
        key = "diffusion_model.key"
        merged_tensor = torch.randn(4, 4)
        merged_state = {key: merged_tensor}
        expected = merged_tensor.clone()

        # Cache stores a reference to the same tensor
        entry = _CacheEntry(
            structural_fingerprint="fp",
            block_configs=[],
            merged_state=dict(merged_state),
            storage_dtype=torch.float32,
        )

        # Simulate the persistence overlay (exit.py: base_state[key] = tensor)
        base_state = {"diffusion_model.other": torch.randn(4, 4)}
        for k, tensor in merged_state.items():
            base_state[k] = tensor  # reference assignment, no copy

        # base_state now aliases the same tensor — verify cache is unmodified
        assert torch.equal(entry.merged_state[key], expected)
        assert entry.merged_state[key] is merged_tensor


# ===========================================================================
# Cache Loader Metadata — @cache-loader-metadata
# ===========================================================================


class TestCacheLoaderMetadata:
    """Tests for loader_bytes metadata stored in _CacheEntry."""

    def setup_method(self):
        _incremental_cache.clear()

    def teardown_method(self):
        _incremental_cache.clear()

    # AC: @cache-loader-metadata ac-1
    def test_cache_entry_stores_loader_bytes(self, mock_model_patcher):
        """_CacheEntry.loader_bytes contains measured sum of all loader loaded_bytes."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        mock_loader.loaded_bytes = 1024 * 1024  # 1 MB

        merged = {k: torch.randn(4, 4) for k in keys}
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        assert len(_incremental_cache) == 1
        entry = next(iter(_incremental_cache.values()))
        assert entry.loader_bytes == 1024 * 1024

    # AC: @cache-loader-metadata ac-2
    def test_cache_hit_logs_loader_bytes(self, mock_model_patcher, caplog):
        """Incremental cache hit logs cached loader_bytes."""
        import logging

        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with loader_bytes=2048
        fp = "test_fingerprint"
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe),
            merged_state={k: torch.randn(4, 4) for k in keys},
            storage_dtype=torch.float32,
            loader_bytes=2048,
        )

        mock_analyze, mock_model_analysis, _, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compute_structural_fingerprint", return_value=fp),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.chunked_evaluation", MagicMock()),
            patch("nodes.exit.compile_batch_groups", return_value={}),
            caplog.at_level(logging.INFO, logger="ecaj.exit"),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        assert any(
            "loader_bytes=2048" in record.message
            for record in caplog.records
        ), f"Expected loader_bytes=2048 in log, got: {[r.message for r in caplog.records]}"

    # AC: @cache-loader-metadata ac-3
    def test_new_run_stores_new_loader_bytes(self, mock_model_patcher):
        """New run with different loaders stores new loader_bytes, not old."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with old loader_bytes
        old_fp = "old_fingerprint"
        _incremental_cache[old_fp] = _CacheEntry(
            structural_fingerprint=old_fp,
            block_configs=collect_block_configs(recipe),
            merged_state={k: torch.randn(4, 4) for k in keys},
            storage_dtype=torch.float32,
            loader_bytes=999,
        )

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        mock_loader.loaded_bytes = 5000  # Different loader size

        merged = {k: torch.randn(4, 4) for k in keys}
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compute_structural_fingerprint", return_value="new_fp"),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        assert len(_incremental_cache) == 1
        entry = next(iter(_incremental_cache.values()))
        assert entry.loader_bytes == 5000, f"Expected 5000, got {entry.loader_bytes}"

    # AC: @cache-loader-metadata ac-4
    def test_loader_bytes_not_in_preflight_estimate(self, mock_model_patcher):
        """Cached loader_bytes does not affect preflight RAM estimate."""
        keys = list(mock_model_patcher.model_state_dict().keys())
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.5),))
        recipe = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc,
            ),
            backbone=None,
            t_factor=1.0,
        )

        # Pre-populate cache with LARGE loader_bytes — should not affect preflight
        fp = "test_fingerprint"
        bc_changed = BlockConfig(arch="sdxl", block_overrides=(("IN00", 0.9),))
        recipe_changed = RecipeMerge(
            base=RecipeBase(model_patcher=mock_model_patcher, arch="sdxl"),
            target=RecipeLoRA(
                loras=({"path": "lora_a.safetensors", "strength": 1.0},),
                block_config=bc_changed,
            ),
            backbone=None,
            t_factor=1.0,
        )
        _incremental_cache[fp] = _CacheEntry(
            structural_fingerprint=fp,
            block_configs=collect_block_configs(recipe_changed),
            merged_state={k: torch.randn(4, 4) for k in keys},
            storage_dtype=torch.float32,
            loader_bytes=999_999_999_999,  # ~1 TB — absurd value
        )

        mock_analyze, mock_model_analysis, mock_loader, dummy_plan = _make_exit_mocks(
            mock_model_patcher, keys, recipe=recipe,
        )
        mock_loader.loaded_bytes = 100

        merged = {k: torch.randn(4, 4) for k in keys}
        from lib.batch_groups import OpSignature
        sig = OpSignature(shape=(4, 4), ndim=2)

        preflight_calls = []

        def capture_preflight(**kwargs):
            preflight_calls.append(kwargs)

        with (
            patch("nodes.exit.analyze_recipe", return_value=mock_analyze),
            patch("nodes.exit.analyze_recipe_models", return_value=mock_model_analysis),
            patch("nodes.exit.compile_plan", return_value=dummy_plan),
            patch("nodes.exit.compile_batch_groups", return_value={sig: keys}),
            patch("nodes.exit.chunked_evaluation", return_value=merged),
            patch("nodes.exit.compute_base_identity", return_value="base_id"),
            patch("nodes.exit.compute_lora_stats", return_value={}),
            patch("nodes.exit.compute_structural_fingerprint", return_value="different_fp"),
            patch("nodes.exit.check_ram_preflight", side_effect=capture_preflight),
        ):
            node = WIDENExitNode()
            node.execute(recipe)

        # Preflight was called with loader_bytes for logging, but the estimate
        # formula does not include loader memory (it's already in MemAvailable).
        assert len(preflight_calls) == 1
        call_kwargs = preflight_calls[0]
        # loader_bytes is passed for diagnostic logging, not added to estimate
        assert "loader_bytes" in call_kwargs
        # merged_state_bytes should be reasonable (based on key sizes, not loader data)
        assert call_kwargs["merged_state_bytes"] < 999_999_999_999
        # n_models is no longer passed (loaders already reflected in MemAvailable)
        assert "n_models" not in call_kwargs
