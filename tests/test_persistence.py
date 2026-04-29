"""Tests for lib/persistence.py — AC coverage for @exit-model-persistence spec."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from lib.persistence import (
    atomic_save,
    build_metadata,
    check_cache,
    check_checkpoint_cache,
    check_full_model_cache,
    compute_base_identity,
    compute_lora_stats,
    compute_recipe_hash,
    load_affected_keys,
    serialize_recipe,
    validate_model_name,
)
from lib.recipe import (
    BlockConfig,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
)

# =============================================================================
# AC-5, AC-11, AC-12: validate_model_name
# =============================================================================


class TestValidateModelName:
    """AC: @exit-model-persistence ac-5, ac-11, ac-12"""

    # AC: @exit-model-persistence ac-5
    def test_empty_name_raises(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_model_name("")

    # AC: @exit-model-persistence ac-5
    def test_whitespace_only_raises(self):
        """Whitespace-only name should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_model_name("   ")

    # AC: @exit-model-persistence ac-12
    def test_path_traversal_dotdot_raises(self):
        """Name with '..' should raise ValueError."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_model_name("../evil")

    # AC: @exit-model-persistence ac-12
    def test_forward_slash_raises(self):
        """Name with '/' should raise ValueError."""
        with pytest.raises(ValueError, match="path separators"):
            validate_model_name("subdir/model")

    # AC: @exit-model-persistence ac-12
    def test_backslash_raises(self):
        """Name with backslash should raise ValueError."""
        with pytest.raises(ValueError, match="path separators"):
            validate_model_name("subdir\\model")

    # AC: @exit-model-persistence ac-11
    def test_auto_appends_safetensors(self):
        """Name without extension should get .safetensors appended."""
        assert validate_model_name("my_model") == "my_model.safetensors"

    # AC: @exit-model-persistence ac-11
    def test_preserves_existing_extension(self):
        """Name already ending in .safetensors should be preserved."""
        assert validate_model_name("my_model.safetensors") == "my_model.safetensors"

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        assert validate_model_name("  my_model  ") == "my_model.safetensors"


# =============================================================================
# AC-6, AC-7: serialize_recipe
# =============================================================================


class TestSerializeRecipe:
    """AC: @exit-model-persistence ac-6, ac-7"""

    # AC: @exit-model-persistence ac-6
    def test_recipe_base_serialization(self):
        """RecipeBase should serialize with base_identity, not model_patcher."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        result = serialize_recipe(base, "abc123", {})
        parsed = json.loads(result)
        assert parsed["type"] == "RecipeBase"
        assert parsed["arch"] == "sdxl"
        assert parsed["base_identity"] == "abc123"
        assert "model_patcher" not in result

    # AC: @exit-model-persistence ac-6
    def test_recipe_lora_serialization(self):
        """RecipeLoRA should serialize loras with path and strength."""
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 0.8},))
        result = serialize_recipe(lora, "abc", {})
        parsed = json.loads(result)
        assert parsed["type"] == "RecipeLoRA"
        assert len(parsed["loras"]) == 1
        assert parsed["loras"][0]["path"] == "test.safetensors"
        assert parsed["loras"][0]["strength"] == 0.8

    # AC: @exit-model-persistence ac-7
    def test_lora_stats_included(self):
        """LoRA file stats should be included in serialization."""
        lora = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        stats = {"a.safetensors": (1234.5, 67890)}
        result = serialize_recipe(lora, "abc", stats)
        parsed = json.loads(result)
        assert parsed["loras"][0]["mtime"] == 1234.5
        assert parsed["loras"][0]["size"] == 67890

    # AC: @exit-model-persistence ac-6
    def test_recipe_compose_serialization(self):
        """RecipeCompose should serialize all branches."""
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.5},))
        compose = RecipeCompose(branches=(lora_a, lora_b))
        result = serialize_recipe(compose, "abc", {})
        parsed = json.loads(result)
        assert parsed["type"] == "RecipeCompose"
        assert len(parsed["branches"]) == 2

    # AC: @exit-model-persistence ac-6
    def test_recipe_merge_serialization(self):
        """RecipeMerge should serialize base, target, and t_factor."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "x.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=0.7)
        result = serialize_recipe(merge, "abc", {})
        parsed = json.loads(result)
        assert parsed["type"] == "RecipeMerge"
        assert parsed["t_factor"] == 0.7
        assert parsed["base"]["type"] == "RecipeBase"
        assert "backbone" not in parsed  # None backbone omitted

    # AC: @exit-model-persistence ac-6
    def test_block_config_serialization(self):
        """BlockConfig should be serialized when present."""
        bc = BlockConfig(arch="sdxl", block_overrides=(("IN00-02", 0.5),))
        lora = RecipeLoRA(
            loras=({"path": "x.safetensors", "strength": 1.0},),
            block_config=bc,
        )
        result = serialize_recipe(lora, "abc", {})
        parsed = json.loads(result)
        assert "block_config" in parsed
        assert parsed["block_config"]["arch"] == "sdxl"
        assert parsed["block_config"]["block_overrides"] == [["IN00-02", 0.5]]

    # AC: @exit-model-persistence ac-6
    def test_deterministic_output(self):
        """Same recipe should always produce the same JSON."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "x.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=0.5)
        stats = {"x.safetensors": (100.0, 200)}

        r1 = serialize_recipe(merge, "abc", stats)
        r2 = serialize_recipe(merge, "abc", stats)
        assert r1 == r2

    # AC: @exit-model-persistence ac-6
    def test_merge_with_backbone(self):
        """RecipeMerge with backbone should include it."""
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "x.safetensors", "strength": 1.0},))
        backbone = RecipeLoRA(loras=({"path": "y.safetensors", "strength": 0.5},))
        merge = RecipeMerge(base=base, target=lora, backbone=backbone, t_factor=0.5)
        result = serialize_recipe(merge, "abc", {})
        parsed = json.loads(result)
        assert "backbone" in parsed
        assert parsed["backbone"]["type"] == "RecipeLoRA"

    # AC: @diffusion-model-path-resolution ac-6
    def test_recipe_model_includes_source_dir(self):
        """RecipeModel serialization should include source_dir."""
        model = RecipeModel(path="test.safetensors", strength=0.8, source_dir="checkpoints")
        result = serialize_recipe(model, "abc", {})
        parsed = json.loads(result)
        assert parsed["type"] == "RecipeModel"
        assert parsed["source_dir"] == "checkpoints"

    # AC: @diffusion-model-path-resolution ac-6
    def test_recipe_model_diffusion_models_source_dir(self):
        """RecipeModel with diffusion_models source_dir should serialize correctly."""
        model = RecipeModel(
            path="flux_dev.safetensors", strength=1.0, source_dir="diffusion_models"
        )
        result = serialize_recipe(model, "abc", {})
        parsed = json.loads(result)
        assert parsed["source_dir"] == "diffusion_models"

    # AC: @diffusion-model-path-resolution ac-6
    def test_recipe_model_source_dir_affects_hash(self):
        """Same path with different source_dir should produce different serialization."""
        model_ckpt = RecipeModel(
            path="model.safetensors", strength=1.0, source_dir="checkpoints"
        )
        model_diff = RecipeModel(
            path="model.safetensors", strength=1.0, source_dir="diffusion_models"
        )
        result_ckpt = serialize_recipe(model_ckpt, "abc", {})
        result_diff = serialize_recipe(model_diff, "abc", {})
        assert result_ckpt != result_diff


# =============================================================================
# AC-6: compute_base_identity
# =============================================================================


class TestComputeBaseIdentity:
    """AC: @exit-model-persistence ac-6"""

    # AC: @exit-model-persistence ac-6
    def test_same_state_same_identity(self):
        """Same state dict should produce identical identity."""
        state = {"key_a": torch.ones(4, 4), "key_b": torch.zeros(4, 4)}
        id1 = compute_base_identity(state)
        id2 = compute_base_identity(state)
        assert id1 == id2

    # AC: @exit-model-persistence ac-6
    def test_different_keys_different_identity(self):
        """Different keys should produce different identity."""
        state1 = {"key_a": torch.ones(4, 4)}
        state2 = {"key_b": torch.ones(4, 4)}
        assert compute_base_identity(state1) != compute_base_identity(state2)

    # AC: @exit-model-persistence ac-6
    def test_different_shapes_different_identity(self):
        """Different shapes should produce different identity."""
        state1 = {"key_a": torch.ones(4, 4)}
        state2 = {"key_a": torch.ones(8, 8)}
        assert compute_base_identity(state1) != compute_base_identity(state2)

    # AC: @exit-model-persistence ac-6
    def test_different_dtypes_different_identity(self):
        """Different dtypes should produce different identity."""
        state1 = {"key_a": torch.ones(4, 4, dtype=torch.float32)}
        state2 = {"key_a": torch.ones(4, 4, dtype=torch.float16)}
        assert compute_base_identity(state1) != compute_base_identity(state2)

    # AC: @exit-model-persistence ac-6
    def test_different_values_different_identity(self):
        """Same architecture but different weights should produce different identity."""
        state1 = {"key_a": torch.ones(4, 4)}
        state2 = {"key_a": torch.zeros(4, 4)}
        assert compute_base_identity(state1) != compute_base_identity(state2)


# =============================================================================
# AC-6: compute_recipe_hash
# =============================================================================


class TestComputeRecipeHash:
    """AC: @exit-model-persistence ac-6"""

    # AC: @exit-model-persistence ac-6
    def test_deterministic(self):
        """Same input should produce same hash."""
        s = '{"type":"RecipeBase","arch":"sdxl","base_identity":"abc"}'
        assert compute_recipe_hash(s) == compute_recipe_hash(s)

    # AC: @exit-model-persistence ac-6
    def test_different_inputs_different_hash(self):
        """Different inputs should produce different hashes."""
        s1 = '{"type":"RecipeBase","arch":"sdxl","base_identity":"abc"}'
        s2 = '{"type":"RecipeBase","arch":"sdxl","base_identity":"def"}'
        assert compute_recipe_hash(s1) != compute_recipe_hash(s2)


# =============================================================================
# AC-7: compute_lora_stats
# =============================================================================


class TestComputeLoraStats:
    """AC: @exit-model-persistence ac-7"""

    # AC: @exit-model-persistence ac-7
    def test_collects_stats_from_lora(self, tmp_path):
        """Should collect mtime and size from LoRA files."""
        lora_file = tmp_path / "test.safetensors"
        lora_file.write_bytes(b"x" * 100)

        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        def resolver(name):
            return str(tmp_path / name)

        stats = compute_lora_stats(lora, resolver)
        assert "test.safetensors" in stats
        mtime, size = stats["test.safetensors"]
        assert size == 100
        assert mtime > 0

    # AC: @exit-model-persistence ac-7
    def test_missing_lora_gets_sentinel(self):
        """Missing LoRA file should get (0.0, 0) sentinel values."""
        lora = RecipeLoRA(loras=({"path": "missing.safetensors", "strength": 1.0},))

        def resolver(name):
            return f"/nonexistent/{name}"

        stats = compute_lora_stats(lora, resolver)
        assert stats["missing.safetensors"] == (0.0, 0)

    # AC: @exit-model-persistence ac-7
    def test_walks_merge_tree(self, tmp_path):
        """Should walk through merge tree to find all LoRAs."""
        for name in ("a.safetensors", "b.safetensors"):
            (tmp_path / name).write_bytes(b"x" * 50)

        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora_a = RecipeLoRA(loras=({"path": "a.safetensors", "strength": 1.0},))
        lora_b = RecipeLoRA(loras=({"path": "b.safetensors", "strength": 0.5},))
        merge = RecipeMerge(base=base, target=lora_a, backbone=lora_b, t_factor=0.5)

        def resolver(name):
            return str(tmp_path / name)

        stats = compute_lora_stats(merge, resolver)
        assert "a.safetensors" in stats
        assert "b.safetensors" in stats

    # AC: @diffusion-model-path-resolution ac-8
    def test_model_resolver_receives_source_dir(self, tmp_path):
        """Model resolver should receive source_dir from RecipeModel."""
        model_file = tmp_path / "checkpoints" / "test.safetensors"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"x" * 200)

        model = RecipeModel(
            path="test.safetensors", strength=1.0, source_dir="checkpoints"
        )

        received_args = []

        def lora_resolver(name):
            return None

        def model_resolver(name, source_dir):
            received_args.append((name, source_dir))
            if source_dir == "checkpoints":
                return str(tmp_path / "checkpoints" / name)
            return None

        stats = compute_lora_stats(model, lora_resolver, model_resolver)
        assert received_args == [("test.safetensors", "checkpoints")]
        assert "test.safetensors" in stats
        assert stats["test.safetensors"][1] == 200

    # AC: @diffusion-model-path-resolution ac-8
    def test_diffusion_models_source_dir(self, tmp_path):
        """Model with diffusion_models source_dir should resolve correctly."""
        diff_dir = tmp_path / "diffusion_models"
        diff_dir.mkdir(parents=True, exist_ok=True)
        (diff_dir / "flux.safetensors").write_bytes(b"y" * 300)

        model = RecipeModel(
            path="flux.safetensors", strength=1.0, source_dir="diffusion_models"
        )

        def lora_resolver(name):
            return None

        def model_resolver(name, source_dir):
            return str(tmp_path / source_dir / name)

        stats = compute_lora_stats(model, lora_resolver, model_resolver)
        assert stats["flux.safetensors"][1] == 300

    # AC: @diffusion-model-path-resolution ac-8
    def test_mixed_lora_and_model_stats(self, tmp_path):
        """Should collect stats from both LoRAs and models with correct resolvers."""
        # Setup files
        (tmp_path / "lora.safetensors").write_bytes(b"L" * 100)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model.safetensors").write_bytes(b"M" * 500)

        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "lora.safetensors", "strength": 1.0},))
        model = RecipeModel(
            path="model.safetensors", strength=0.5, source_dir="checkpoints"
        )
        compose = RecipeCompose(branches=(lora, model))
        merge = RecipeMerge(base=base, target=compose, backbone=None, t_factor=0.5)

        def lora_resolver(name):
            return str(tmp_path / name)

        def model_resolver(name, source_dir):
            return str(tmp_path / source_dir / name)

        stats = compute_lora_stats(merge, lora_resolver, model_resolver)
        assert stats["lora.safetensors"][1] == 100
        assert stats["model.safetensors"][1] == 500


# =============================================================================
# AC-3, AC-4, AC-9: check_cache
# =============================================================================


class TestCheckCache:
    """AC: @exit-model-persistence ac-3, ac-4, ac-9"""

    def _make_cached_file(self, path, recipe_hash="abc123"):
        """Helper to create a valid ecaj-cached safetensors file."""
        tensors = {"key_a": torch.randn(4, 4)}
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": recipe_hash,
            "__ecaj_affected_keys__": '["key_a"]',
        }
        save_file(tensors, str(path), metadata=metadata)

    # AC: @exit-model-persistence ac-3
    def test_cache_hit(self, tmp_path):
        """Matching hash should return metadata."""
        path = tmp_path / "model.safetensors"
        self._make_cached_file(path, "abc123")
        result = check_cache(str(path), "abc123")
        assert result is not None
        assert result["__ecaj_recipe_hash__"] == "abc123"

    # AC: @exit-model-persistence ac-4
    def test_cache_mismatch(self, tmp_path):
        """Non-matching hash should return None."""
        path = tmp_path / "model.safetensors"
        self._make_cached_file(path, "abc123")
        result = check_cache(str(path), "different_hash")
        assert result is None

    def test_file_not_found(self, tmp_path):
        """Missing file should return None."""
        result = check_cache(str(tmp_path / "nonexistent.safetensors"), "abc123")
        assert result is None

    # AC: @exit-model-persistence ac-9
    def test_non_ecaj_file_raises(self, tmp_path):
        """File without ecaj metadata should raise ValueError."""
        path = tmp_path / "model.safetensors"
        save_file({"key_a": torch.randn(4, 4)}, str(path))
        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_cache(str(path), "abc123")

    # AC: @exit-model-persistence ac-9
    def test_empty_metadata_raises(self, tmp_path):
        """File with empty metadata should raise ValueError."""
        path = tmp_path / "model.safetensors"
        save_file({"key_a": torch.randn(4, 4)}, str(path), metadata={})
        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_cache(str(path), "abc123")


# =============================================================================
# AC-3: load_affected_keys
# =============================================================================


class TestLoadAffectedKeys:
    """AC: @exit-model-persistence ac-3"""

    # AC: @exit-model-persistence ac-3
    def test_selective_load(self, tmp_path):
        """Should load only the requested keys."""
        path = tmp_path / "model.safetensors"
        tensors = {
            "key_a": torch.randn(4, 4),
            "key_b": torch.randn(4, 4),
            "key_c": torch.randn(4, 4),
        }
        save_file(tensors, str(path))

        result = load_affected_keys(str(path), ["key_a", "key_c"])
        assert set(result.keys()) == {"key_a", "key_c"}
        assert torch.allclose(result["key_a"], tensors["key_a"])
        assert torch.allclose(result["key_c"], tensors["key_c"])


# =============================================================================
# AC-6, AC-13, AC-14: build_metadata
# =============================================================================


class TestBuildMetadata:
    """AC: @exit-model-persistence ac-6, ac-13, ac-14"""

    # AC: @exit-model-persistence ac-6
    def test_core_fields_present(self):
        """Should include version, recipe, hash, and affected keys."""
        metadata = build_metadata('{"test": true}', "abc123", ["key_a", "key_b"])
        assert metadata["__ecaj_version__"] == "1"
        assert metadata["__ecaj_recipe__"] == '{"test": true}'
        assert metadata["__ecaj_recipe_hash__"] == "abc123"
        assert json.loads(metadata["__ecaj_affected_keys__"]) == ["key_a", "key_b"]

    # AC: @exit-model-persistence ac-13
    def test_workflow_included_when_provided(self):
        """Workflow JSON should be included when provided."""
        workflow = '{"nodes": []}'
        metadata = build_metadata("{}", "abc", ["k"], workflow_json=workflow)
        assert metadata["__ecaj_workflow__"] == workflow

    # AC: @exit-model-persistence ac-14
    def test_workflow_excluded_when_none(self):
        """Workflow should not be in metadata when None."""
        metadata = build_metadata("{}", "abc", ["k"], workflow_json=None)
        assert "__ecaj_workflow__" not in metadata


# =============================================================================
# AC-8, AC-10: atomic_save
# =============================================================================


class TestAtomicSave:
    """AC: @exit-model-persistence ac-8, ac-10"""

    # AC: @exit-model-persistence ac-10
    def test_atomic_replace(self, tmp_path):
        """Should atomically write the file."""
        path = tmp_path / "model.safetensors"
        tensors = {"key_a": torch.randn(4, 4)}
        metadata = {"__ecaj_version__": "1"}

        atomic_save(tensors, str(path), metadata)
        assert path.exists()

        # Verify no temp file remains
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @exit-model-persistence ac-10
    def test_no_partial_on_failure(self, tmp_path):
        """Failed save should not leave a partial file."""
        path = tmp_path / "model.safetensors"

        # Force a failure by passing non-tensor data
        with pytest.raises(Exception):
            atomic_save({"key_a": "not_a_tensor"}, str(path), {})

        assert not path.exists()
        # Temp file should be cleaned up
        tmp_files = list(tmp_path.glob(".ecaj_tmp_*"))
        assert len(tmp_files) == 0

    # AC: @exit-model-persistence ac-8
    def test_all_keys_saved(self, tmp_path):
        """Saved file should contain all provided keys."""
        path = tmp_path / "model.safetensors"
        tensors = {
            "key_a": torch.randn(4, 4),
            "key_b": torch.randn(8, 8),
            "key_c": torch.randn(2, 2),
        }
        metadata = {"__ecaj_version__": "1"}

        atomic_save(tensors, str(path), metadata)

        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            saved_keys = set(f.keys())
        assert saved_keys == {"key_a", "key_b", "key_c"}

    # AC: @exit-model-persistence ac-8
    def test_correct_dtypes(self, tmp_path):
        """Saved tensors should preserve dtypes."""
        path = tmp_path / "model.safetensors"
        tensors = {
            "fp32": torch.randn(4, 4, dtype=torch.float32),
            "bf16": torch.randn(4, 4, dtype=torch.bfloat16),
        }

        atomic_save(tensors, str(path), {"__ecaj_version__": "1"})

        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            assert f.get_tensor("fp32").dtype == torch.float32
            assert f.get_tensor("bf16").dtype == torch.bfloat16

    # AC: @exit-model-persistence ac-10
    def test_overwrites_existing(self, tmp_path):
        """Should atomically overwrite an existing file."""
        path = tmp_path / "model.safetensors"

        # Write initial
        tensors1 = {"key_a": torch.ones(4, 4)}
        atomic_save(tensors1, str(path), {"__ecaj_version__": "1"})

        # Overwrite
        tensors2 = {"key_b": torch.zeros(4, 4)}
        atomic_save(tensors2, str(path), {"__ecaj_version__": "1"})

        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            assert set(f.keys()) == {"key_b"}

    # AC: @exit-model-persistence ac-10
    def test_metadata_preserved(self, tmp_path):
        """Metadata should be readable after save."""
        path = tmp_path / "model.safetensors"
        tensors = {"key_a": torch.randn(4, 4)}
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe_hash__": "test_hash",
        }

        atomic_save(tensors, str(path), metadata)

        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            saved_meta = f.metadata()
        assert saved_meta["__ecaj_version__"] == "1"
        assert saved_meta["__ecaj_recipe_hash__"] == "test_hash"


# =============================================================================
# AC: @exit-model-persistence ac-6 — build_metadata artifact_kind fields
# =============================================================================


class TestBuildMetadataArtifactKind:
    """AC: @exit-model-persistence ac-6 — artifact kind and checkpoint metadata."""

    # AC: @exit-model-persistence ac-6
    def test_artifact_kind_included(self):
        """artifact_kind should be stored in metadata."""
        metadata = build_metadata("{}", "abc", [], artifact_kind="checkpoint")
        assert metadata["__ecaj_artifact_kind__"] == "checkpoint"

    # AC: @exit-model-persistence ac-6
    def test_base_identity_included(self):
        """base_identity should be stored in metadata."""
        metadata = build_metadata("{}", "abc", [], base_identity="sha256hash")
        assert metadata["__ecaj_base_identity__"] == "sha256hash"

    # AC: @exit-model-persistence ac-6
    def test_dependency_fingerprints_included(self):
        """dependency_fingerprints should be stored in metadata."""
        deps = json.dumps({"lora.safetensors": [1234.5, 100]})
        metadata = build_metadata("{}", "abc", [], dependency_fingerprints=deps)
        assert metadata["__ecaj_dependency_fingerprints__"] == deps

    # AC: @exit-model-persistence ac-6
    def test_checkpoint_components_flag(self):
        """checkpoint_components=True should set classification metadata."""
        metadata = build_metadata("{}", "abc", [], checkpoint_components=True)
        assert metadata["__ecaj_checkpoint_components__"] == "true"

    # AC: @exit-model-persistence ac-6
    def test_checkpoint_components_false_omitted(self):
        """checkpoint_components=False should not include the key."""
        metadata = build_metadata("{}", "abc", [], checkpoint_components=False)
        assert "__ecaj_checkpoint_components__" not in metadata

    # AC: @exit-model-persistence ac-6
    def test_omitted_optional_fields(self):
        """Optional fields should not appear when not provided."""
        metadata = build_metadata("{}", "abc", [])
        assert "__ecaj_artifact_kind__" not in metadata
        assert "__ecaj_base_identity__" not in metadata
        assert "__ecaj_dependency_fingerprints__" not in metadata
        assert "__ecaj_checkpoint_components__" not in metadata

    # AC: @exit-model-persistence ac-6
    def test_full_checkpoint_metadata(self):
        """All checkpoint fields should be present together."""
        deps = json.dumps({"lora.safetensors": [1234.5, 100]})
        metadata = build_metadata(
            '{"recipe": true}', "hash123", ["key_a"],
            artifact_kind="checkpoint",
            base_identity="base_sha256",
            dependency_fingerprints=deps,
            checkpoint_components=True,
        )
        assert metadata["__ecaj_version__"] == "1"
        assert metadata["__ecaj_recipe__"] == '{"recipe": true}'
        assert metadata["__ecaj_recipe_hash__"] == "hash123"
        assert metadata["__ecaj_artifact_kind__"] == "checkpoint"
        assert metadata["__ecaj_base_identity__"] == "base_sha256"
        assert metadata["__ecaj_dependency_fingerprints__"] == deps
        assert metadata["__ecaj_checkpoint_components__"] == "true"


# =============================================================================
# AC: @saved-model-artifact-safety, @exit-model-persistence — check_checkpoint_cache
# =============================================================================


class TestCheckCheckpointCache:
    """AC: @saved-model-artifact-safety ac-missing-metadata-not-reused,
    ac-wrong-artifact-kind-not-reused, ac-internal-format-not-checkpoint-cache;
    @exit-model-persistence ac-3, ac-4, ac-6, ac-9."""

    # Checkpoint-style tensors include non-diffusion keys (CLIP/VAE components)
    _CHECKPOINT_TENSORS = {
        "diffusion_model.input.weight": torch.randn(4, 4),
        "first_stage_model.decoder.weight": torch.randn(4, 4),  # VAE
        "cond_stage_model.transformer.weight": torch.randn(4, 4),  # CLIP
    }

    _BASE_IDENTITY = "base_sha256_abc"
    _DEPS = json.dumps({"lora.safetensors": [1234.5, 100]})

    def _make_checkpoint_file(self, path, *, recipe_hash="abc123",
                              artifact_kind="checkpoint",
                              base_identity=None,
                              dependency_fingerprints=None,
                              checkpoint_components=True,
                              version="1",
                              tensors=None,
                              extra_metadata=None):
        """Helper to create a valid checkpoint-style safetensors file."""
        if base_identity is None:
            base_identity = self._BASE_IDENTITY
        if dependency_fingerprints is None:
            dependency_fingerprints = self._DEPS
        if tensors is None:
            tensors = self._CHECKPOINT_TENSORS

        metadata = {
            "__ecaj_version__": version,
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": recipe_hash,
            "__ecaj_affected_keys__": '["diffusion_model.input.weight"]',
            "__ecaj_artifact_kind__": artifact_kind,
            "__ecaj_base_identity__": base_identity,
            "__ecaj_dependency_fingerprints__": dependency_fingerprints,
        }
        if checkpoint_components:
            metadata["__ecaj_checkpoint_components__"] = "true"
        if extra_metadata:
            metadata.update(extra_metadata)
        save_file(tensors, str(path), metadata=metadata)

    # AC: @exit-model-persistence ac-3
    def test_valid_checkpoint_cache_hit(self, tmp_path):
        """Matching checkpoint artifact kind and matching metadata are accepted."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is True

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_missing_artifact_kind_is_cache_miss(self, tmp_path):
        """Missing artifact kind metadata causes cache miss."""
        path = tmp_path / "model.safetensors"
        tensors = self._CHECKPOINT_TENSORS
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            "__ecaj_dependency_fingerprints__": self._DEPS,
            "__ecaj_checkpoint_components__": "true",
            # No __ecaj_artifact_kind__
        }
        save_file(tensors, str(path), metadata=metadata)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
    def test_wrong_artifact_kind_is_cache_miss(self, tmp_path):
        """Wrong artifact kind (e.g. 'diffusion' instead of 'checkpoint') is cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path, artifact_kind="diffusion")
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_missing_checkpoint_components_is_cache_miss(self, tmp_path):
        """Missing checkpoint component classification is cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path, checkpoint_components=False)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_unsupported_ecaj_version_is_cache_miss(self, tmp_path):
        """Unsupported ecaj version causes cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path, version="999")
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_internal_format_diffusion_only_rejected(self, tmp_path):
        """Older internal-format with diffusion_model-only keys is rejected."""
        path = tmp_path / "model.safetensors"
        internal_tensors = {
            "diffusion_model.input_blocks.0.0.weight": torch.randn(4, 4),
            "diffusion_model.middle_block.0.weight": torch.randn(4, 4),
            "diffusion_model.output_blocks.0.0.weight": torch.randn(4, 4),
        }
        self._make_checkpoint_file(path, tensors=internal_tensors)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_internal_format_noise_augmentor_only_rejected(self, tmp_path):
        """Internal artifact with noise_augmentor/model_sampling keys only is rejected."""
        path = tmp_path / "model.safetensors"
        internal_tensors = {
            "diffusion_model.layers.0.weight": torch.randn(4, 4),
            "noise_augmentor.weight": torch.randn(4, 4),
            "model_sampling.sigmas": torch.randn(4),
        }
        self._make_checkpoint_file(path, tensors=internal_tensors)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
    def test_model_prefix_diffusion_only_rejected(self, tmp_path):
        """Internal artifact with model.diffusion_model.* prefix only is rejected."""
        path = tmp_path / "model.safetensors"
        internal_tensors = {
            "model.diffusion_model.input_blocks.0.weight": torch.randn(4, 4),
            "model.diffusion_model.output_blocks.0.weight": torch.randn(4, 4),
        }
        self._make_checkpoint_file(path, tensors=internal_tensors)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-9
    def test_non_ecaj_file_raises(self, tmp_path):
        """Existing non-ecaj file raises/refuses overwrite."""
        path = tmp_path / "model.safetensors"
        save_file({"key_a": torch.randn(4, 4)}, str(path))
        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_checkpoint_cache(
                str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
            )

    # AC: @exit-model-persistence ac-9
    def test_non_ecaj_empty_metadata_raises(self, tmp_path):
        """Existing file with empty metadata raises ValueError."""
        path = tmp_path / "model.safetensors"
        save_file({"key_a": torch.randn(4, 4)}, str(path), metadata={})
        with pytest.raises(ValueError, match="not an ecaj-saved model"):
            check_checkpoint_cache(
                str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
            )

    # AC: @exit-model-persistence ac-4
    def test_recipe_hash_mismatch_is_cache_miss(self, tmp_path):
        """Different recipe hash causes cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path, recipe_hash="abc123")
        result = check_checkpoint_cache(
            str(path), "different_hash", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_base_identity_mismatch_is_cache_miss(self, tmp_path):
        """Different base model identity causes cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path)
        result = check_checkpoint_cache(
            str(path), "abc123", "different_base_identity", self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_dependency_fingerprints_mismatch_is_cache_miss(self, tmp_path):
        """Different dependency fingerprints cause cache miss."""
        path = tmp_path / "model.safetensors"
        self._make_checkpoint_file(path)
        different_deps = json.dumps({"lora.safetensors": [9999.9, 200]})
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, different_deps,
        )
        assert result is False

    def test_missing_file_returns_false(self, tmp_path):
        """Non-existent file returns False (not an error)."""
        result = check_checkpoint_cache(
            str(tmp_path / "nonexistent.safetensors"),
            "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_missing_base_identity_metadata_is_cache_miss(self, tmp_path):
        """Artifact without base_identity metadata causes cache miss."""
        path = tmp_path / "model.safetensors"
        tensors = self._CHECKPOINT_TENSORS
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_checkpoint_components__": "true",
            "__ecaj_dependency_fingerprints__": self._DEPS,
            # No __ecaj_base_identity__
        }
        save_file(tensors, str(path), metadata=metadata)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_missing_dependency_fingerprints_metadata_is_cache_miss(self, tmp_path):
        """Artifact without dependency_fingerprints metadata causes cache miss."""
        path = tmp_path / "model.safetensors"
        tensors = self._CHECKPOINT_TENSORS
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_checkpoint_components__": "true",
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            # No __ecaj_dependency_fingerprints__
        }
        save_file(tensors, str(path), metadata=metadata)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_checkpoint_components_false_is_cache_miss(self, tmp_path):
        """Artifact with __ecaj_checkpoint_components__='false' is cache miss."""
        path = tmp_path / "model.safetensors"
        tensors = self._CHECKPOINT_TENSORS
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            "__ecaj_dependency_fingerprints__": self._DEPS,
            "__ecaj_checkpoint_components__": "false",
        }
        save_file(tensors, str(path), metadata=metadata)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_missing_recipe_metadata_is_cache_miss(self, tmp_path):
        """Artifact without __ecaj_recipe__ metadata is cache miss."""
        path = tmp_path / "model.safetensors"
        tensors = self._CHECKPOINT_TENSORS
        metadata = {
            "__ecaj_version__": "1",
            # No __ecaj_recipe__
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": "[]",
            "__ecaj_artifact_kind__": "checkpoint",
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            "__ecaj_dependency_fingerprints__": self._DEPS,
            "__ecaj_checkpoint_components__": "true",
        }
        save_file(tensors, str(path), metadata=metadata)
        result = check_checkpoint_cache(
            str(path), "abc123", self._BASE_IDENTITY, self._DEPS,
        )
        assert result is False


# AC: @exit-model-persistence ac-4, ac-6 — check_full_model_cache artifact classification
# =============================================================================


class TestCheckFullModelCacheClassification:
    """AC: @exit-model-persistence ac-4, ac-6;
    @saved-model-artifact-safety ac-missing-metadata-not-reused,
    ac-wrong-artifact-kind-not-reused."""

    _TENSORS = {
        "diffusion_model.input.weight": torch.randn(4, 4),
        "diffusion_model.output.weight": torch.randn(4, 4),
    }
    _BASE_IDENTITY = "base_sha256_xyz"
    _DEPS = json.dumps({"lora.safetensors": [5678.0, 200]})

    def _make_full_file(self, path, *, recipe_hash="abc123",
                        artifact_kind="diffusion",
                        base_identity=None,
                        dependency_fingerprints=None):
        """Helper to create a full-model safetensors file with classification metadata."""
        if base_identity is None:
            base_identity = self._BASE_IDENTITY
        if dependency_fingerprints is None:
            dependency_fingerprints = self._DEPS
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": recipe_hash,
            "__ecaj_affected_keys__": '["diffusion_model.input.weight"]',
            "__ecaj_output_mode__": "full",
            "__ecaj_artifact_kind__": artifact_kind,
            "__ecaj_base_identity__": base_identity,
            "__ecaj_dependency_fingerprints__": dependency_fingerprints,
        }
        save_file(self._TENSORS, str(path), metadata=metadata)

    # AC: @exit-model-persistence ac-3
    def test_valid_full_model_with_classification_hit(self, tmp_path):
        """Matching artifact kind, base identity, and deps are accepted."""
        path = tmp_path / "model.safetensors"
        self._make_full_file(path)
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is True

    # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
    def test_missing_artifact_kind_is_miss(self, tmp_path):
        """Full artifact without __ecaj_artifact_kind__ fails when caller expects it."""
        path = tmp_path / "model.safetensors"
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": '["diffusion_model.input.weight"]',
            "__ecaj_output_mode__": "full",
            # No __ecaj_artifact_kind__
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            "__ecaj_dependency_fingerprints__": self._DEPS,
        }
        save_file(self._TENSORS, str(path), metadata=metadata)
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is False

    # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
    def test_wrong_artifact_kind_is_miss(self, tmp_path):
        """Full artifact with wrong artifact kind fails."""
        path = tmp_path / "model.safetensors"
        self._make_full_file(path, artifact_kind="checkpoint")
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_missing_base_identity_is_miss(self, tmp_path):
        """Full artifact without __ecaj_base_identity__ fails when caller expects it."""
        path = tmp_path / "model.safetensors"
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": '["diffusion_model.input.weight"]',
            "__ecaj_output_mode__": "full",
            "__ecaj_artifact_kind__": "diffusion",
            # No __ecaj_base_identity__
            "__ecaj_dependency_fingerprints__": self._DEPS,
        }
        save_file(self._TENSORS, str(path), metadata=metadata)
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_missing_dependency_fingerprints_is_miss(self, tmp_path):
        """Full artifact without __ecaj_dependency_fingerprints__ fails when caller expects it."""
        path = tmp_path / "model.safetensors"
        metadata = {
            "__ecaj_version__": "1",
            "__ecaj_recipe__": "{}",
            "__ecaj_recipe_hash__": "abc123",
            "__ecaj_affected_keys__": '["diffusion_model.input.weight"]',
            "__ecaj_output_mode__": "full",
            "__ecaj_artifact_kind__": "diffusion",
            "__ecaj_base_identity__": self._BASE_IDENTITY,
            # No __ecaj_dependency_fingerprints__
        }
        save_file(self._TENSORS, str(path), metadata=metadata)
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_base_identity_mismatch_is_miss(self, tmp_path):
        """Full artifact with wrong base identity fails."""
        path = tmp_path / "model.safetensors"
        self._make_full_file(path)
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity="different_base",
            expected_dependency_fingerprints=self._DEPS,
        )
        assert result is False

    # AC: @exit-model-persistence ac-4
    def test_dependency_fingerprints_mismatch_is_miss(self, tmp_path):
        """Full artifact with wrong dependency fingerprints fails."""
        path = tmp_path / "model.safetensors"
        self._make_full_file(path)
        different_deps = json.dumps({"lora.safetensors": [9999.0, 999]})
        result = check_full_model_cache(
            str(path), "abc123",
            expected_artifact_kind="diffusion",
            expected_base_identity=self._BASE_IDENTITY,
            expected_dependency_fingerprints=different_deps,
        )
        assert result is False
