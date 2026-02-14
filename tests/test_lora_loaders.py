"""Tests for architecture-specific LoRA loaders.

Covers all 4 acceptance criteria:
- AC-1: Architecture-specific loader selection and key mapping
- AC-2: Produces DeltaSpec objects compatible with batched executor
- AC-3: Pluggable design (new loaders integrate without modifying existing)
- AC-4: Loader interface (load, affected_keys, get_delta_specs, cleanup)
"""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.executor import DeltaSpec
from lib.lora import (
    LOADER_REGISTRY,
    LoRALoader,
    QwenLoader,
    SDXLLoader,
    ZImageLoader,
    get_loader,
)

# ---------------------------------------------------------------------------
# Fixtures: Create temporary LoRA files for testing
# ---------------------------------------------------------------------------


@pytest.fixture
def sdxl_lora_file() -> str:
    """Create a temporary SDXL-format LoRA file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # Standard linear LoRA
            "lora_unet_input_blocks_0_0_proj_in.lora_up.weight": torch.randn(64, 8),
            "lora_unet_input_blocks_0_0_proj_in.lora_down.weight": torch.randn(8, 32),
            # Another layer
            "lora_unet_middle_block_0_proj.lora_up.weight": torch.randn(128, 16),
            "lora_unet_middle_block_0_proj.lora_down.weight": torch.randn(16, 64),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def zimage_lora_file() -> str:
    """Create a temporary Z-Image format LoRA file with QKV components."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # QKV LoRA components (to_q, to_k, to_v)
            "transformer.layers.0.attention.to_q.lora_A.weight": torch.randn(8, 3840),
            "transformer.layers.0.attention.to_q.lora_B.weight": torch.randn(3840, 8),
            "transformer.layers.0.attention.to_k.lora_A.weight": torch.randn(8, 3840),
            "transformer.layers.0.attention.to_k.lora_B.weight": torch.randn(3840, 8),
            "transformer.layers.0.attention.to_v.lora_A.weight": torch.randn(8, 3840),
            "transformer.layers.0.attention.to_v.lora_B.weight": torch.randn(3840, 8),
            # Standard feed-forward LoRA
            "transformer.layers.0.ff.linear_1.lora_A.weight": torch.randn(16, 3840),
            "transformer.layers.0.ff.linear_1.lora_B.weight": torch.randn(15360, 16),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def qwen_diffusers_lora_file() -> str:
    """Create a temporary Qwen Diffusers-format LoRA file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # Diffusers format: transformer.transformer_blocks.N.*.lora_A/B.weight
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(8, 3072),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(3072, 8),
            "transformer.transformer_blocks.0.attn.to_k.lora_A.weight": torch.randn(8, 3072),
            "transformer.transformer_blocks.0.attn.to_k.lora_B.weight": torch.randn(3072, 8),
            "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": torch.randn(8, 3072),
            "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": torch.randn(3072, 8),
            # Feed-forward
            "transformer.transformer_blocks.0.mlp.gate_proj.lora_A.weight": torch.randn(16, 3072),
            "transformer.transformer_blocks.0.mlp.gate_proj.lora_B.weight": torch.randn(12288, 16),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def qwen_kohya_lora_file() -> str:
    """Create a temporary Qwen A1111/kohya-format LoRA file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # Kohya format: lora_unet_transformer_blocks_N_*.lora_up/down.weight
            "lora_unet_transformer_blocks_5_attn_to_q.lora_down.weight": torch.randn(8, 3072),
            "lora_unet_transformer_blocks_5_attn_to_q.lora_up.weight": torch.randn(3072, 8),
            "lora_unet_transformer_blocks_5_ff_gate_proj.lora_down.weight": torch.randn(8, 3072),
            "lora_unet_transformer_blocks_5_ff_gate_proj.lora_up.weight": torch.randn(12288, 8),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def qwen_lycoris_lora_file() -> str:
    """Create a temporary Qwen LyCORIS-format LoRA file."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tensors = {
            # LyCORIS format: lycoris_transformer_blocks_N_*.lora_down/up.weight
            "lycoris_transformer_blocks_10_attn_to_q.lora_down.weight": torch.randn(8, 3072),
            "lycoris_transformer_blocks_10_attn_to_q.lora_up.weight": torch.randn(3072, 8),
            "lycoris_transformer_blocks_10_mlp_down_proj.lora_down.weight": torch.randn(16, 12288),
            "lycoris_transformer_blocks_10_mlp_down_proj.lora_up.weight": torch.randn(3072, 16),
        }
        save_file(tensors, f.name)
        return f.name


@pytest.fixture
def cleanup_lora_files(
    sdxl_lora_file: str,
    zimage_lora_file: str,
    qwen_diffusers_lora_file: str,
    qwen_kohya_lora_file: str,
    qwen_lycoris_lora_file: str,
):
    """Clean up temporary files after tests."""
    yield
    Path(sdxl_lora_file).unlink(missing_ok=True)
    Path(zimage_lora_file).unlink(missing_ok=True)
    Path(qwen_diffusers_lora_file).unlink(missing_ok=True)
    Path(qwen_kohya_lora_file).unlink(missing_ok=True)
    Path(qwen_lycoris_lora_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC-1: Architecture-specific loader selection and key mapping
# ---------------------------------------------------------------------------


class TestAC1ArchitectureSelection:
    """AC-1: Given a LoRA file and architecture tag, the correct loader handles key mapping."""

    def test_sdxl_loader_selected_for_sdxl_arch(self):
        """SDXL architecture tag returns SDXLLoader."""
        # AC: @lora-loaders ac-1
        loader = get_loader("sdxl")
        assert isinstance(loader, SDXLLoader)

    def test_zimage_loader_selected_for_zimage_arch(self):
        """Z-Image architecture tag returns ZImageLoader."""
        # AC: @lora-loaders ac-1
        loader = get_loader("zimage")
        assert isinstance(loader, ZImageLoader)

    def test_unsupported_arch_raises_value_error(self):
        """Unsupported architecture raises helpful ValueError."""
        # AC: @lora-loaders ac-1
        with pytest.raises(ValueError, match="Unsupported architecture 'flux'"):
            get_loader("flux")

    def test_sdxl_key_mapping(self, sdxl_lora_file: str, cleanup_lora_files):
        """SDXL loader maps LoRA keys to model keys correctly."""
        # AC: @lora-loaders ac-1
        loader = SDXLLoader()
        loader.load(sdxl_lora_file)

        # Check that keys are mapped to diffusion_model.* format
        for key in loader.affected_keys:
            assert key.startswith("diffusion_model."), f"Key {key} missing prefix"
            assert key.endswith(".weight"), f"Key {key} missing suffix"

        loader.cleanup()

    def test_zimage_qkv_key_mapping(self, zimage_lora_file: str, cleanup_lora_files):
        """Z-Image loader maps QKV keys to fused qkv.weight format."""
        # AC: @lora-loaders ac-1
        loader = ZImageLoader()
        loader.load(zimage_lora_file)

        affected = loader.affected_keys
        # QKV components should map to single fused key
        qkv_key = "diffusion_model.layers.0.attention.qkv.weight"
        assert qkv_key in affected, f"Expected {qkv_key} in affected keys"

        # Standard FF key should also be present
        ff_key = "diffusion_model.layers.0.ff.linear_1.weight"
        assert ff_key in affected, f"Expected {ff_key} in affected keys"

        loader.cleanup()


# ---------------------------------------------------------------------------
# AC-2: Produces DeltaSpec objects compatible with batched executor
# ---------------------------------------------------------------------------


class TestAC2DeltaSpecProduction:
    """AC-2: Architecture loaders produce DeltaSpec objects for batched executor."""

    def test_sdxl_produces_deltaspec_objects(
        self, sdxl_lora_file: str, cleanup_lora_files
    ):
        """SDXL loader produces DeltaSpec with correct fields."""
        # AC: @lora-loaders ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_file, strength=0.8)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert len(specs) > 0, "Should produce at least one DeltaSpec"

        for spec in specs:
            assert isinstance(spec, DeltaSpec)
            assert spec.kind in ("standard", "lokr", "qkv_q", "qkv_k", "qkv_v")
            assert spec.key_index in key_indices.values()
            assert spec.up is not None
            assert spec.down is not None
            assert isinstance(spec.scale, float)

        loader.cleanup()

    def test_zimage_produces_qkv_deltaspec(
        self, zimage_lora_file: str, cleanup_lora_files
    ):
        """Z-Image loader produces qkv_* kind DeltaSpecs for QKV layers."""
        # AC: @lora-loaders ac-2
        loader = ZImageLoader()
        loader.load(zimage_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Should have QKV specs (q, k, v) and standard specs
        qkv_kinds = {s.kind for s in specs if s.kind.startswith("qkv_")}
        assert qkv_kinds == {"qkv_q", "qkv_k", "qkv_v"}, f"Missing QKV kinds: {qkv_kinds}"

        # Should also have standard spec for FF layer
        standard_specs = [s for s in specs if s.kind == "standard"]
        assert len(standard_specs) > 0, "Should have standard specs for FF layer"

        loader.cleanup()

    def test_deltaspec_tensors_are_valid(
        self, sdxl_lora_file: str, cleanup_lora_files
    ):
        """DeltaSpec up/down tensors have valid shapes for bmm."""
        # AC: @lora-loaders ac-2
        loader = SDXLLoader()
        loader.load(sdxl_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        for spec in specs:
            if spec.kind == "standard":
                # up: (out, rank), down: (rank, in) for bmm compatibility
                assert spec.up.dim() == 2, f"Up should be 2D: {spec.up.shape}"
                assert spec.down.dim() == 2, f"Down should be 2D: {spec.down.shape}"
                # up columns == down rows (rank dimension)
                assert spec.up.shape[1] == spec.down.shape[0], (
                    f"Rank mismatch: up {spec.up.shape} vs down {spec.down.shape}"
                )

        loader.cleanup()

    def test_strength_affects_scale(self, sdxl_lora_file: str, cleanup_lora_files):
        """LoRA strength multiplier affects DeltaSpec scale."""
        # AC: @lora-loaders ac-2
        loader1 = SDXLLoader()
        loader1.load(sdxl_lora_file, strength=1.0)
        keys = list(loader1.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs1 = loader1.get_delta_specs(keys, key_indices)
        loader1.cleanup()

        loader2 = SDXLLoader()
        loader2.load(sdxl_lora_file, strength=0.5)
        specs2 = loader2.get_delta_specs(keys, key_indices)
        loader2.cleanup()

        # Same key should have half the scale
        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2, strict=True):
            assert abs(s1.scale - 2 * s2.scale) < 1e-6, (
                f"Scale mismatch: {s1.scale} vs {s2.scale}"
            )


# ---------------------------------------------------------------------------
# AC-3: Pluggable design (new loaders integrate without modifying existing)
# ---------------------------------------------------------------------------


class TestAC3PluggableDesign:
    """AC-3: New architecture loaders integrate without modifying existing ones."""

    def test_registry_is_dict_of_loader_classes(self):
        """Registry maps arch tags to LoRALoader subclasses."""
        # AC: @lora-loaders ac-3
        assert isinstance(LOADER_REGISTRY, dict)
        for arch, loader_cls in LOADER_REGISTRY.items():
            assert isinstance(arch, str)
            assert issubclass(loader_cls, LoRALoader)

    def test_loaders_are_independent_modules(self):
        """Each loader is in its own module (no cross-dependencies)."""
        # AC: @lora-loaders ac-3
        from lib.lora import sdxl, zimage

        # Modules should exist separately
        assert hasattr(sdxl, "SDXLLoader")
        assert hasattr(zimage, "ZImageLoader")

        # Neither imports the other
        import inspect

        sdxl_source = inspect.getsource(sdxl)
        zimage_source = inspect.getsource(zimage)

        # Check that sdxl doesn't import zimage
        assert "zimage" not in sdxl_source.lower()
        # Check that zimage doesn't import sdxl
        assert "sdxl" not in zimage_source.lower()

    def test_adding_new_arch_only_requires_registry_entry(self):
        """New architecture can be added by just updating the registry."""
        # AC: @lora-loaders ac-3
        # Create a mock loader
        class MockLoader(LoRALoader):
            def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
                pass

            @property
            def affected_keys(self) -> set[str]:
                return set()

            def affected_keys_for_set(self, set_id: str) -> set[str]:
                return set()

            def get_delta_specs(self, keys, key_indices, set_id=None) -> list[DeltaSpec]:
                return []

            def cleanup(self) -> None:
                pass

        # Add to registry
        original_registry = LOADER_REGISTRY.copy()
        LOADER_REGISTRY["mock_arch"] = MockLoader

        try:
            # Can now get the loader
            loader = get_loader("mock_arch")
            assert isinstance(loader, MockLoader)

            # Original loaders still work
            assert isinstance(get_loader("sdxl"), SDXLLoader)
            assert isinstance(get_loader("zimage"), ZImageLoader)
        finally:
            # Restore registry
            LOADER_REGISTRY.clear()
            LOADER_REGISTRY.update(original_registry)


# ---------------------------------------------------------------------------
# AC-4: Loader interface (load, affected_keys, get_delta_specs, cleanup)
# ---------------------------------------------------------------------------


class TestAC4LoaderInterface:
    """AC-4: Loaders implement the full interface contract."""

    def test_loader_has_load_method(self):
        """Loaders have load(path, strength) method."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        assert callable(getattr(loader, "load", None))

    def test_loader_has_affected_keys_property(self):
        """Loaders have affected_keys property returning set-like type."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        assert hasattr(loader, "affected_keys")
        assert isinstance(loader.affected_keys, (set, frozenset))

    def test_loader_has_get_delta_specs_method(self):
        """Loaders have get_delta_specs(keys, key_indices) method."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        assert callable(getattr(loader, "get_delta_specs", None))

    def test_loader_has_cleanup_method(self):
        """Loaders have cleanup() method."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        assert callable(getattr(loader, "cleanup", None))

    def test_cleanup_clears_state(self, sdxl_lora_file: str, cleanup_lora_files):
        """cleanup() releases loaded tensors."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        loader.load(sdxl_lora_file)
        assert len(loader.affected_keys) > 0, "Should have affected keys"

        loader.cleanup()
        assert len(loader.affected_keys) == 0, "cleanup should clear affected keys"

    def test_context_manager_calls_cleanup(self, sdxl_lora_file: str, cleanup_lora_files):
        """Context manager (__enter__/__exit__) calls cleanup automatically."""
        # AC: @lora-loaders ac-4
        with SDXLLoader() as loader:
            loader.load(sdxl_lora_file)
            assert len(loader.affected_keys) > 0

        # After context exit, cleanup should have been called
        assert len(loader.affected_keys) == 0

    def test_multiple_loads_accumulate(self, sdxl_lora_file: str, cleanup_lora_files):
        """Multiple load() calls accumulate LoRA data."""
        # AC: @lora-loaders ac-4
        loader = SDXLLoader()
        loader.load(sdxl_lora_file, strength=0.5)

        # Loading same file again should add more data
        loader.load(sdxl_lora_file, strength=0.3)
        # Same keys, but more entries
        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Should have 2 specs per key (loaded twice)
        specs_per_key = len(specs) / len(keys)
        assert specs_per_key == 2, f"Expected 2 specs per key, got {specs_per_key}"

        loader.cleanup()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the loader system."""

    def test_full_workflow_sdxl(self, sdxl_lora_file: str, cleanup_lora_files):
        """Full workflow: get loader, load, get specs, cleanup."""
        # Get architecture-appropriate loader
        loader = get_loader("sdxl")

        # Load LoRA file
        loader.load(sdxl_lora_file, strength=0.75)

        # Check affected keys
        affected = loader.affected_keys
        assert len(affected) > 0

        # Get delta specs for batched execution
        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Verify specs are executor-compatible
        assert all(isinstance(s, DeltaSpec) for s in specs)
        assert all(s.kind in ("standard", "lokr", "qkv_q", "qkv_k", "qkv_v") for s in specs)

        # Cleanup
        loader.cleanup()
        assert len(loader.affected_keys) == 0

    def test_full_workflow_zimage_qkv(self, zimage_lora_file: str, cleanup_lora_files):
        """Full workflow for Z-Image with QKV fusing."""
        loader = get_loader("zimage")
        loader.load(zimage_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0

        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Should have both QKV and standard specs
        kinds = {s.kind for s in specs}
        assert "qkv_q" in kinds
        assert "qkv_k" in kinds
        assert "qkv_v" in kinds
        assert "standard" in kinds

        loader.cleanup()


# ---------------------------------------------------------------------------
# Qwen-specific tests
# ---------------------------------------------------------------------------


class TestQwenLoader:
    """Tests for Qwen architecture LoRA loader."""

    def test_qwen_loader_selected_for_qwen_arch(self):
        """Qwen architecture tag returns QwenLoader."""
        # AC: @qwen-lora-loader ac-4
        loader = get_loader("qwen")
        assert isinstance(loader, QwenLoader)

    def test_qwen_loader_in_registry(self):
        """QwenLoader is registered in LOADER_REGISTRY."""
        # AC: @lora-loaders ac-3
        assert "qwen" in LOADER_REGISTRY
        assert LOADER_REGISTRY["qwen"] is QwenLoader

    def test_qwen_diffusers_format_loads(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """Qwen loader handles diffusers format LoRA files."""
        # AC: @qwen-lora-loader ac-4
        loader = QwenLoader()
        loader.load(qwen_diffusers_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        # Keys should be in diffusion_model.transformer_blocks.N format
        for key in affected:
            assert key.startswith("diffusion_model."), f"Key {key} missing prefix"
            assert "transformer_blocks" in key, f"Key {key} missing transformer_blocks"
            assert key.endswith(".weight"), f"Key {key} missing suffix"

        loader.cleanup()

    def test_qwen_kohya_format_loads(
        self, qwen_kohya_lora_file: str, cleanup_lora_files
    ):
        """Qwen loader handles A1111/kohya format LoRA files."""
        # AC: @qwen-lora-loader ac-4
        loader = QwenLoader()
        loader.load(qwen_kohya_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        # Verify key mapping: lora_unet_transformer_blocks_5 -> transformer_blocks.5
        assert any("transformer_blocks.5" in k for k in affected), (
            f"Expected transformer_blocks.5 in keys: {affected}"
        )

        loader.cleanup()

    def test_qwen_lycoris_format_loads(
        self, qwen_lycoris_lora_file: str, cleanup_lora_files
    ):
        """Qwen loader handles LyCORIS format LoRA files."""
        # AC: @qwen-lora-loader ac-4, ac-5
        loader = QwenLoader()
        loader.load(qwen_lycoris_lora_file)

        affected = loader.affected_keys
        assert len(affected) > 0, "Should have affected keys"

        # Verify key mapping: lycoris_transformer_blocks_10 -> transformer_blocks.10
        assert any("transformer_blocks.10" in k for k in affected), (
            f"Expected transformer_blocks.10 in keys: {affected}"
        )

        loader.cleanup()

    def test_qwen_produces_deltaspec_objects(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """Qwen loader produces DeltaSpec with correct fields."""
        # AC: @qwen-lora-loader ac-6
        loader = QwenLoader()
        loader.load(qwen_diffusers_lora_file, strength=0.8)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        assert len(specs) > 0, "Should produce at least one DeltaSpec"

        for spec in specs:
            assert isinstance(spec, DeltaSpec)
            # Qwen uses standard specs only (no QKV fusing)
            assert spec.kind == "standard", f"Expected standard kind, got {spec.kind}"
            assert spec.key_index in key_indices.values()
            assert spec.up is not None
            assert spec.down is not None
            assert isinstance(spec.scale, float)

        loader.cleanup()

    def test_qwen_no_qkv_fusing(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """Qwen loader does NOT fuse QKV weights (unlike Z-Image)."""
        # AC: @qwen-lora-loader ac-6
        loader = QwenLoader()
        loader.load(qwen_diffusers_lora_file)

        keys = list(loader.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # All specs should be 'standard', not qkv_*
        kinds = {s.kind for s in specs}
        assert kinds == {"standard"}, f"Expected only standard kind, got {kinds}"

        # to_q, to_k, to_v should be separate keys
        qkv_keys = [k for k in keys if any(p in k for p in ["to_q", "to_k", "to_v"])]
        assert len(qkv_keys) == 3, f"Expected 3 separate QKV keys, got {qkv_keys}"

        loader.cleanup()

    def test_qwen_compound_names_preserved(
        self, qwen_lycoris_lora_file: str, cleanup_lora_files
    ):
        """Compound names like to_q, mlp, down_proj are preserved during normalization."""
        # AC: @qwen-lora-loader ac-5
        loader = QwenLoader()
        loader.load(qwen_lycoris_lora_file)

        affected = loader.affected_keys

        # to_q should be preserved (not split into to.q)
        to_q_keys = [k for k in affected if "to_q" in k]
        assert len(to_q_keys) > 0, f"Expected to_q in keys: {affected}"

        # down_proj should be preserved
        down_proj_keys = [k for k in affected if "down_proj" in k]
        assert len(down_proj_keys) > 0, f"Expected down_proj in keys: {affected}"

        loader.cleanup()

    def test_qwen_strength_affects_scale(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """LoRA strength multiplier affects DeltaSpec scale."""
        # AC: @qwen-lora-loader ac-6
        loader1 = QwenLoader()
        loader1.load(qwen_diffusers_lora_file, strength=1.0)
        keys = list(loader1.affected_keys)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs1 = loader1.get_delta_specs(keys, key_indices)
        loader1.cleanup()

        loader2 = QwenLoader()
        loader2.load(qwen_diffusers_lora_file, strength=0.5)
        specs2 = loader2.get_delta_specs(keys, key_indices)
        loader2.cleanup()

        # Same key should have half the scale
        assert len(specs1) == len(specs2)
        for s1, s2 in zip(specs1, specs2, strict=True):
            assert abs(s1.scale - 2 * s2.scale) < 1e-6, (
                f"Scale mismatch: {s1.scale} vs {s2.scale}"
            )

    def test_qwen_cleanup_clears_state(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """cleanup() releases loaded tensors."""
        # AC: @lora-loaders ac-4
        loader = QwenLoader()
        loader.load(qwen_diffusers_lora_file)
        assert len(loader.affected_keys) > 0, "Should have affected keys"

        loader.cleanup()
        assert len(loader.affected_keys) == 0, "cleanup should clear affected keys"

    def test_qwen_full_workflow(
        self, qwen_diffusers_lora_file: str, cleanup_lora_files
    ):
        """Full workflow: get loader, load, get specs, cleanup."""
        # Get architecture-appropriate loader
        loader = get_loader("qwen")

        # Load LoRA file
        loader.load(qwen_diffusers_lora_file, strength=0.75)

        # Check affected keys
        affected = loader.affected_keys
        assert len(affected) > 0

        # Get delta specs for batched execution
        keys = list(affected)
        key_indices = {k: i for i, k in enumerate(keys)}
        specs = loader.get_delta_specs(keys, key_indices)

        # Verify specs are executor-compatible
        assert all(isinstance(s, DeltaSpec) for s in specs)
        assert all(s.kind == "standard" for s in specs)

        # Cleanup
        loader.cleanup()
        assert len(loader.affected_keys) == 0
