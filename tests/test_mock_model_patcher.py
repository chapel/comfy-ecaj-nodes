"""MockModelPatcher fidelity tests — validates it matches real ModelPatcher API.
# AC: @testing-infrastructure ac-2
"""

import uuid

import torch

from tests.conftest import MockModelPatcher


class TestStateDict:
    """model_state_dict access and filtering."""

    def test_returns_all_keys(self):
        mp = MockModelPatcher()
        sd = mp.model_state_dict()
        assert len(sd) == 4
        assert all(k.startswith("diffusion_model.") for k in sd)

    def test_filter_prefix(self):
        mp = MockModelPatcher()
        sd = mp.model_state_dict(filter_prefix="diffusion_model.input_blocks")
        assert len(sd) == 2
        assert all("input_blocks" in k for k in sd)

    def test_filter_no_match(self):
        mp = MockModelPatcher()
        sd = mp.model_state_dict(filter_prefix="nonexistent.")
        assert len(sd) == 0

    def test_tensors_are_4x4_float32(self):
        mp = MockModelPatcher()
        for t in mp.model_state_dict().values():
            assert t.shape == (4, 4)
            assert t.dtype == torch.float32


class TestClone:
    """clone() produces independent copies with shared underlying tensors."""

    def test_clone_shares_state_dict(self):
        mp = MockModelPatcher()
        cl = mp.clone()
        # Same underlying dict object (shallow clone)
        assert cl._state_dict is mp._state_dict

    def test_clone_independent_patches(self):
        mp = MockModelPatcher()
        cl = mp.clone()
        key = "diffusion_model.input_blocks.0.0.weight"
        cl.add_patches({key: torch.zeros(4, 4)})
        assert key in cl.patches
        assert key not in mp.patches

    def test_clone_copies_uuid(self):
        """clone() copies patches_uuid from source, matching real ComfyUI behavior."""
        mp = MockModelPatcher()
        cl = mp.clone()
        assert cl.patches_uuid == mp.patches_uuid

    def test_clone_uuid_diverges_after_add_patches(self):
        """After add_patches, clone UUID diverges from source."""
        mp = MockModelPatcher()
        cl = mp.clone()
        assert cl.patches_uuid == mp.patches_uuid
        key = "diffusion_model.input_blocks.0.0.weight"
        cl.add_patches({key: torch.zeros(4, 4)})
        assert cl.patches_uuid != mp.patches_uuid


class TestAddPatches:
    """add_patches registers only for existing keys and updates UUID."""

    def test_adds_existing_key(self):
        mp = MockModelPatcher()
        key = "diffusion_model.middle_block.0.weight"
        added = mp.add_patches({key: torch.ones(4, 4)})
        assert added == [key]
        assert len(mp.patches[key]) == 1

    def test_ignores_nonexistent_key(self):
        mp = MockModelPatcher()
        added = mp.add_patches({"nonexistent.key": torch.ones(4, 4)})
        assert added == []
        assert "nonexistent.key" not in mp.patches

    def test_updates_uuid(self):
        mp = MockModelPatcher()
        old_uuid = mp.patches_uuid
        mp.add_patches({"diffusion_model.middle_block.0.weight": torch.ones(4, 4)})
        assert mp.patches_uuid != old_uuid

    def test_patch_tuple_format(self):
        mp = MockModelPatcher()
        key = "diffusion_model.middle_block.0.weight"
        patch_data = torch.ones(4, 4)
        mp.add_patches({key: patch_data}, strength_patch=0.8, strength_model=0.5)
        entry = mp.patches[key][0]
        assert entry == (0.8, patch_data, 0.5, None, None)


class TestGetKeyPatches:
    """get_key_patches returns expected format with original weight."""

    def test_unpatched_format(self):
        mp = MockModelPatcher()
        kp = mp.get_key_patches()
        assert len(kp) == 4
        for k, v in kp.items():
            assert len(v) == 1  # just the (weight, convert_func) base entry
            weight, convert_fn = v[0]
            assert isinstance(weight, torch.Tensor)
            assert callable(convert_fn)

    def test_patched_includes_patches(self):
        mp = MockModelPatcher()
        key = "diffusion_model.middle_block.0.weight"
        mp.add_patches({key: torch.ones(4, 4)})
        kp = mp.get_key_patches()
        assert len(kp[key]) == 2  # base + 1 patch

    def test_filter_prefix(self):
        mp = MockModelPatcher()
        kp = mp.get_key_patches(filter_prefix="diffusion_model.output_blocks")
        assert len(kp) == 1


class TestDiffusionModelAccess:
    """model.diffusion_model state dict access matching real ModelPatcher."""

    def test_model_diffusion_model_exists(self):
        mp = MockModelPatcher()
        assert hasattr(mp, "model")
        assert hasattr(mp.model, "diffusion_model")

    def test_diffusion_model_state_dict(self):
        mp = MockModelPatcher()
        sd = mp.model.diffusion_model.state_dict()
        assert len(sd) == 4
        # Keys should have diffusion_model. prefix stripped
        for k in sd:
            assert not k.startswith("diffusion_model.")

    def test_diffusion_model_state_dict_values_match(self):
        mp = MockModelPatcher()
        full_sd = mp.model_state_dict()
        dm_sd = mp.model.diffusion_model.state_dict()
        for short_key, tensor in dm_sd.items():
            full_key = f"diffusion_model.{short_key}"
            assert full_key in full_sd
            assert torch.equal(tensor, full_sd[full_key])

    def test_clone_has_diffusion_model(self):
        mp = MockModelPatcher()
        cl = mp.clone()
        assert hasattr(cl.model, "diffusion_model")
        assert len(cl.model.diffusion_model.state_dict()) == 4


class TestPatchesUUID:
    """patches_uuid is a proper UUID."""

    def test_uuid_type(self):
        mp = MockModelPatcher()
        assert isinstance(mp.patches_uuid, uuid.UUID)
