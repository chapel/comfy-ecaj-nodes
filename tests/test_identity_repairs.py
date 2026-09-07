"""Bounded CPU regressions for persistence identity and recipe snapshots."""

import hashlib
import json
import os
from types import MappingProxyType

import pytest
import torch

from lib.persistence import (
    compute_base_identity,
    compute_lora_stats,
    compute_recipe_hash,
    serialize_recipe,
)
from lib.recipe import CheckpointComponents, RecipeBase, RecipeCompose, RecipeLoRA, RecipeModel


# AC: @exit-model-persistence ac-4 ac-6
@pytest.mark.parametrize("mutation", ["local", "late", "numpy"])
def test_untrusted_content_mutations_never_reuse_identity(mutation):
    state = {str(i): torch.zeros(128) for i in range(5)}
    before = compute_base_identity(state)
    if mutation == "local":
        state["1"][0] = 1
    elif mutation == "late":
        state["0"][100] = 1
    else:
        version = state["1"]._version
        state["1"].numpy()[0] = 1
        assert state["1"]._version == version
    assert compute_base_identity(state) != before


# AC: @exit-model-persistence ac-4 ac-6
def test_offset_views_do_not_collide():
    storage = torch.arange(200, dtype=torch.float32)
    assert compute_base_identity({"w": storage[:64]}) != compute_base_identity(
        {"w": storage[64:128]}
    )


# AC: @exit-model-persistence ac-bounded-identity-preparation
def test_chunked_transfers_do_not_export_full_storage(monkeypatch):
    real_to = torch.Tensor.to
    transfers = []

    def bounded_to(tensor, *args, **kwargs):
        transfers.append(tensor.numel() * tensor.element_size())
        assert transfers[-1] <= 64
        assert "dtype" not in kwargs
        return real_to(tensor, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not export the backing storage")

    monkeypatch.setattr(torch.Tensor, "to", bounded_to)
    monkeypatch.setattr(torch.Tensor, "untyped_storage", forbidden)
    for tensor in (torch.arange(1024.0), torch.arange(1024.0).reshape(32, 32).T):
        assert len(compute_base_identity({"w": tensor}, chunk_bytes=64)) == 64
    assert sum(transfers) == 8192


# AC: @exit-model-persistence ac-4 ac-6
def test_unknown_companion_identity_conservatively_misses():
    base = RecipeBase(
        object(), "sdxl", checkpoint_components=CheckpointComponents(object(), object())
    )
    first = serialize_recipe(base, "same-diffusion", {})
    second = serialize_recipe(base, "same-diffusion", {})
    assert compute_recipe_hash(first) != compute_recipe_hash(second)


# AC: @exit-model-persistence ac-6
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64])
def test_all_logical_content_matches_independent_digest(dtype):
    tensor = torch.arange(120).to(dtype).reshape(4, 5, 6).transpose(0, 2)[1:5:2, :, ::2]
    expected = hashlib.sha256(b"ecaj-base-v2\0")
    signature = json.dumps(["w", list(tensor.shape), str(dtype)], separators=(",", ":")).encode()
    expected.update(len(signature).to_bytes(8, "big"))
    expected.update(signature)
    expected.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert compute_base_identity({"w": tensor}, chunk_bytes=32) == expected.hexdigest()
    assert compute_base_identity({"w": tensor.clone()}, chunk_bytes=1024) == expected.hexdigest()


# AC: @exit-model-persistence ac-6
@pytest.mark.parametrize("kind", ["expanded", "scalar", "empty", "conjugate", "float8"])
def test_special_strided_tensor_layouts_hash_logical_values(kind):
    if kind == "expanded":
        tensor = torch.arange(5.0).expand(7, 5)
    elif kind == "scalar":
        tensor = torch.tensor(2.0)
    elif kind == "empty":
        tensor = torch.empty(0, 4)
    elif kind == "conjugate":
        tensor = torch.tensor([1 + 2j, 3 + 4j]).conj()
    else:
        tensor = torch.arange(8.0).to(torch.float8_e4m3fn)
    expected = tensor.resolve_conj().resolve_neg().contiguous().clone()
    assert compute_base_identity({"w": tensor}, chunk_bytes=16) == compute_base_identity(
        {"w": expected}
    )


# AC: @exit-model-persistence ac-6
def test_float64_precision_is_not_lost_to_float32_conversion():
    left = torch.tensor([1.0], dtype=torch.float64)
    right = torch.nextafter(left, torch.tensor([2.0], dtype=torch.float64))
    assert torch.equal(left.float(), right.float())
    assert compute_base_identity({"w": left}) != compute_base_identity({"w": right})


# AC: @exit-model-persistence ac-6
@pytest.mark.parametrize("kind", ["meta", "sparse", "quantized"])
def test_unsupported_identity_tensors_fail_closed(kind):
    if kind == "meta":
        tensor = torch.empty(2, device="meta")
    elif kind == "sparse":
        tensor = torch.ones(2).to_sparse()
    else:
        tensor = torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8)
    with pytest.raises(ValueError, match="Unsupported identity tensor"):
        compute_base_identity({"w": tensor})


# AC: @exit-model-persistence ac-4 ac-6
def test_companion_only_revision_invalidates_recipe():
    base = RecipeBase(
        object(), "sdxl", checkpoint_components=CheckpointComponents(object(), object())
    )
    first = serialize_recipe(
        base, "diffusion", {}, companion_identities={"clip": "c1", "vae": "v1"}
    )
    assert first == serialize_recipe(
        base, "diffusion", {}, companion_identities={"clip": "c1", "vae": "v1"}
    )
    for ids in ({"clip": "c2", "vae": "v1"}, {"clip": "c1", "vae": "v2"}):
        assert compute_recipe_hash(first) != compute_recipe_hash(
            serialize_recipe(base, "diffusion", {}, companion_identities=ids)
        )


# AC: @exit-model-persistence ac-3 ac-4 ac-6
@pytest.mark.parametrize("component", ["clip", "vae"])
def test_companion_only_change_rejects_real_saved_checkpoint(tmp_path, component):
    from safetensors.torch import save_file

    from lib.persistence import build_metadata, check_checkpoint_cache

    base = RecipeBase(
        object(), "sdxl", checkpoint_components=CheckpointComponents(object(), object())
    )
    companions = {"clip": {"w": torch.ones(4)}, "vae": {"w": torch.ones(4)}}
    ids = {name: compute_base_identity(state) for name, state in companions.items()}
    serialized = serialize_recipe(base, "diffusion", {}, companion_identities=ids)
    original_hash = compute_recipe_hash(serialized)
    metadata = build_metadata(
        serialized,
        original_hash,
        [],
        output_mode="full",
        artifact_kind="checkpoint",
        base_identity="diffusion",
        dependency_fingerprints="{}",
        checkpoint_components=True,
    )
    path = tmp_path / "checkpoint.safetensors"
    save_file(
        {
            "model.diffusion_model.w": torch.ones(1),
            "conditioner.w": torch.ones(1),
            "first_stage_model.w": torch.ones(1),
        },
        str(path),
        metadata=metadata,
    )
    assert check_checkpoint_cache(str(path), original_hash, "diffusion", "{}")
    companions[component]["w"][2] = 5.0
    ids[component] = compute_base_identity(companions[component])
    changed_hash = compute_recipe_hash(
        serialize_recipe(base, "diffusion", {}, companion_identities=ids)
    )
    assert not check_checkpoint_cache(str(path), changed_hash, "diffusion", "{}")


# AC: @exit-model-persistence ac-4
def test_v1_hash_does_not_reuse_real_persisted_cache(tmp_path):
    from safetensors.torch import save_file

    from lib.persistence import build_metadata, check_cache

    serialized = serialize_recipe(RecipeBase(object(), "sdxl"), "base", {})
    old_hash = hashlib.sha256(serialized.encode()).hexdigest()
    path = tmp_path / "old.safetensors"
    save_file(
        {"w": torch.ones(1)}, str(path), metadata=build_metadata(serialized, old_hash, ["w"])
    )
    assert check_cache(str(path), compute_recipe_hash(serialized)) is None


# AC: @recipe-system ac-1
@pytest.mark.parametrize("proxy", [False, True])
def test_lora_snapshot_does_not_alias_callers_mapping(proxy):
    source = {"path": "a.safetensors", "strength": 1.0}
    recipe = RecipeLoRA((MappingProxyType(source) if proxy else source,))
    before = serialize_recipe(recipe, "base", {})
    source["strength"] = 9.0
    assert recipe.loras[0]["strength"] == 1.0
    assert serialize_recipe(recipe, "base", {}) == before
    with pytest.raises(TypeError):
        recipe.loras[0]["strength"] = 2.0


# AC: @diffusion-model-path-resolution ac-3 ac-8
# AC: @exit-model-persistence ac-7
def test_same_filename_dependencies_have_independent_fingerprints(tmp_path):
    for directory, size in [("loras", 1), ("checkpoints", 2), ("diffusion_models", 3)]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "same").write_bytes(b"x" * size)
    recipe = RecipeCompose(
        (
            RecipeLoRA(({"path": "same", "strength": 1.0},)),
            RecipeModel("same"),
            RecipeModel("same", source_dir="diffusion_models"),
        )
    )
    stats = compute_lora_stats(
        recipe, lambda p: str(tmp_path / "loras" / p), lambda p, d: str(tmp_path / d / p)
    )
    parsed = json.loads(serialize_recipe(recipe, "base", stats))
    branches = parsed["branches"]
    assert [branches[0]["loras"][0]["size"], branches[1]["size"], branches[2]["size"]] == [1, 2, 3]


# AC: @diffusion-model-path-resolution ac-8
def test_resolver_none_never_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "same").write_bytes(b"unintended")
    recipe = RecipeModel("same")
    stats = compute_lora_stats(recipe, lambda p: None, lambda p, d: None)
    parsed = json.loads(serialize_recipe(recipe, "base", stats))
    assert parsed["size"] == 0


# AC: @exit-model-persistence ac-7
def test_same_size_content_change_with_restored_mtime_changes_fingerprint(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"before")
    recipe = RecipeLoRA(({"path": "weights", "strength": 1.0},))
    original_stat = path.stat()
    before = compute_lora_stats(recipe, lambda p: str(path))
    path.write_bytes(b"after!")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = compute_lora_stats(recipe, lambda p: str(path))
    fields = next(iter(after.values()))
    assert fields["sha256"] == hashlib.sha256(b"after!").hexdigest()
    assert serialize_recipe(recipe, "base", before) != serialize_recipe(recipe, "base", after)


# AC: @exit-model-persistence ac-7
def test_resolved_location_is_part_of_dependency_identity(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"x")

    stamp = (tmp_path / "a").stat().st_mtime_ns
    os.utime(tmp_path / "b", ns=(stamp, stamp))
    recipe = RecipeLoRA(({"path": "same", "strength": 1.0},))
    first = compute_lora_stats(recipe, lambda p: str(tmp_path / "a"))
    second = compute_lora_stats(recipe, lambda p: str(tmp_path / "b"))
    assert serialize_recipe(recipe, "base", first) != serialize_recipe(recipe, "base", second)
