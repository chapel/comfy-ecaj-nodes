"""IR1/IR2: exact strided bytes and unavailable dependency cache misses."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import lib.persistence as persistence
from lib.recipe import RecipeLoRA, RecipeModel


# AC: @exit-model-persistence ac-6 ac-bounded-identity-preparation
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int16,
        torch.int64,
        torch.complex64,
    ],
)
@pytest.mark.parametrize("length", [0, 1, 4])
def test_degenerate_strides_use_bounded_packed_bytes(dtype, length, monkeypatch):
    tensor = torch.arange(16).to(dtype)[1::3][:length]
    packed = torch.empty(tensor.shape, dtype=dtype)
    packed.copy_(tensor)
    signature = json.dumps(["w", list(tensor.shape), str(dtype)], separators=(",", ":")).encode()
    expected = hashlib.sha256(b"ecaj-base-v2\0")
    expected.update(len(signature).to_bytes(8, "big"))
    expected.update(signature)
    expected.update(packed.view(torch.uint8).numpy().tobytes())
    real_clone, real_numpy = torch.Tensor.clone, torch.Tensor.numpy
    bound = tensor.element_size() if length == 4 else 1024 * 1024
    exports = []

    def bounded_clone(value, *args, **kwargs):
        assert value.numel() * value.element_size() <= bound
        return real_clone(value, *args, **kwargs)

    def bounded_numpy(value, *args, **kwargs):
        assert 0 < value.numel() * value.element_size() <= bound
        exports.append(value.numel() * value.element_size())
        return real_numpy(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", bounded_clone)
    monkeypatch.setattr(torch.Tensor, "numpy", bounded_numpy)
    assert (
        persistence.compute_base_identity({"w": tensor}, chunk_bytes=bound) == expected.hexdigest()
    )
    assert sum(exports) == tensor.numel() * tensor.element_size()
    assert not torch.cuda.is_initialized()


# AC: @exit-model-persistence ac-3 ac-4 ac-bounded-identity-preparation
@pytest.mark.parametrize(
    "failure", ["read", "unresolved", "missing_model_resolver", "legacy", "omitted"]
)
def test_unavailable_dependency_never_reuses_old_payload(tmp_path, monkeypatch, failure):
    dependency = tmp_path / "dependency"
    dependency.write_bytes(b"aaaa")
    recipe = RecipeLoRA(({"path": "dependency", "strength": 1.0},))
    if failure == "missing_model_resolver":
        recipe = RecipeModel(path="dependency", strength=1.0)
        monkeypatch.chdir(tmp_path)  # No implicit CWD model resolver.
    if failure == "read":

        def fail_read(*args, **kwargs):
            raise OSError("injected identity read failure")

        monkeypatch.setattr(persistence, "open", fail_read, raising=False)

    def collect():
        if failure == "legacy":
            return {"dependency": (1.0, 4)}
        if failure == "omitted":
            return {}
        return persistence.compute_lora_stats(
            recipe, lambda _: None if failure == "unresolved" else str(dependency)
        )

    first_stats = collect()
    if failure in {"read", "unresolved", "missing_model_resolver"}:
        fields = next(iter(first_stats.values()))
        assert fields["cacheable"] is False
        assert "sha256" not in fields
        if failure == "missing_model_resolver":
            assert fields["resolved_path"] is None
    first = persistence.serialize_recipe(recipe, "base", first_stats)
    if failure == "legacy":
        entry = json.loads(first)["loras"][0]
        assert entry["mtime"] == 1.0 and entry["size"] == 4
        assert entry["cacheable"] is False
    original_hash = persistence.compute_recipe_hash(first)
    artifact = tmp_path / "output.safetensors"
    save_file(
        {"w": torch.tensor([1.0])},
        str(artifact),
        metadata=persistence.build_metadata(first, original_hash, ["w"]),
    )
    assert persistence.check_cache(str(artifact), original_hash) is not None
    dependency.write_bytes(b"bbbb")
    second = persistence.serialize_recipe(recipe, "base", collect())
    assert persistence.check_cache(str(artifact), persistence.compute_recipe_hash(second)) is None
    # Even reusing the collector result cannot turn unavailable evidence into equality.
    repeated = persistence.serialize_recipe(recipe, "base", first_stats)
    assert (
        persistence.check_cache(str(artifact), persistence.compute_recipe_hash(repeated)) is None
    )
    assert torch.equal(load_file(str(artifact))["w"], torch.tensor([1.0]))
    save_file(
        {"w": torch.tensor([2.0])},
        str(artifact),
        metadata=persistence.build_metadata(
            second, persistence.compute_recipe_hash(second), ["w"]
        ),
    )
    assert torch.equal(load_file(str(artifact))["w"], torch.tensor([2.0]))


# AC: @exit-model-persistence ac-3 ac-4 ac-6
def test_readable_dependency_is_reusable_and_content_change_misses(tmp_path):
    dependency = tmp_path / "dependency"
    dependency.write_bytes(b"aaaa")
    recipe = RecipeLoRA(({"path": "dependency", "strength": 1.0},))

    def serialize():
        return persistence.serialize_recipe(
            recipe, "base", persistence.compute_lora_stats(recipe, lambda _: str(dependency))
        )

    first = serialize()
    artifact = tmp_path / "output.safetensors"
    save_file(
        {"w": torch.tensor([1.0])},
        str(artifact),
        metadata=persistence.build_metadata(first, persistence.compute_recipe_hash(first), ["w"]),
    )
    assert (
        persistence.check_cache(str(artifact), persistence.compute_recipe_hash(serialize()))
        is not None
    )
    dependency.write_bytes(b"bbbb")
    assert (
        persistence.check_cache(str(artifact), persistence.compute_recipe_hash(serialize()))
        is None
    )
    assert torch.equal(load_file(str(artifact))["w"], torch.tensor([1.0]))
