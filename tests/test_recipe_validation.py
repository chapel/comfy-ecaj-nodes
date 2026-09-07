"""CPU characterization of the Exit validators' shared structural contract."""

from functools import partial

import pytest

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel
from lib.recipe_validation import validate_recipe_tree
from nodes.clip_exit import _validate_clip_recipe_tree
from nodes.exit import _validate_recipe_tree

BASE = RecipeBase(object(), "sdxl", domain="clip")
LORA = RecipeLoRA(())
MODEL = RecipeModel("not-loaded.safetensors")
COMPOSE = RecipeCompose((LORA, MODEL))
MERGE = RecipeMerge(BASE, COMPOSE, None, 1.0)
NODES = [BASE, LORA, MODEL, COMPOSE, MERGE, None, 42]
BRANCH_TYPES = "RecipeLoRA, RecipeModel, RecipeCompose, or RecipeMerge"


@pytest.fixture(
    params=[
        _validate_recipe_tree,
        _validate_clip_recipe_tree,
        validate_recipe_tree,
        partial(validate_recipe_tree, expected_domain="clip"),
    ]
)
def validate(request):
    return request.param


def assert_result(validate, node, message=None, path="root"):
    if message is None:
        assert validate(node, path) is None
    else:
        with pytest.raises(ValueError) as exc:
            validate(node, path)
        assert str(exc.value) == message


# AC: @exit-node ac-2
# AC: @clip-exit-node ac-6
@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("position", ["root", "base", "target", "branch", "backbone"])
def test_position_type_matrix(validate, node, position):
    """Backbones allow any valid recipe; root restrictions belong to execute()."""
    name = type(node).__name__
    message = None
    if position == "root":
        tree = node
        if node is None or node == 42:
            message = f"Unknown recipe node type at root: {name}"
    elif position == "base":
        tree = RecipeMerge(node, LORA, None, 1.0)
        if not isinstance(node, (RecipeBase, RecipeMerge)):
            message = (
                f"Invalid base type at root.base: expected RecipeBase or RecipeMerge, got {name}"
            )
    elif position == "target":
        tree = RecipeMerge(BASE, node, None, 1.0)
        if not isinstance(node, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            message = f"Invalid target type at root.target: expected {BRANCH_TYPES}, got {name}"
    elif position == "branch":
        tree = RecipeCompose((LORA, node))
        if not isinstance(node, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            message = (
                f"Invalid branch type at root.branches[1]: expected {BRANCH_TYPES}, got {name}"
            )
    else:
        tree = RecipeMerge(BASE, LORA, node, 1.0)
        if node == 42:
            message = "Unknown recipe node type at root.backbone: int"
    assert_result(validate, tree, message)


# AC: @exit-node ac-2
# AC: @clip-exit-node ac-6
# AC: @node-graph-testing ac-5
@pytest.mark.parametrize(
    ("bad", "suffix", "message"),
    [
        (RecipeCompose(()), "", "RecipeCompose at {path} has no branches"),
        (
            RecipeCompose((LORA, BASE)),
            ".branches[1]",
            "Invalid branch type at {path}: expected " + BRANCH_TYPES + ", got RecipeBase",
        ),
        (
            RecipeMerge(LORA, MODEL, None, 1.0),
            ".base",
            "Invalid base type at {path}: expected RecipeBase or RecipeMerge, got RecipeLoRA",
        ),
        (
            RecipeMerge(BASE, BASE, None, 1.0),
            ".target",
            "Invalid target type at {path}: expected " + BRANCH_TYPES + ", got RecipeBase",
        ),
        (RecipeMerge(BASE, LORA, 42, 1.0), ".backbone", "Unknown recipe node type at {path}: int"),
    ],
)
def test_nested_error_paths(validate, bad, suffix, message):
    tree = RecipeMerge(BASE, RecipeCompose((MERGE, RecipeCompose((MODEL, bad)))), None, 1.0)
    path = "custom.target.branches[1].branches[1]" + suffix
    assert_result(validate, tree, message.format(path=path), path="custom")


# AC: @exit-node ac-2
# AC: @clip-exit-node ac-6
def test_shared_subtrees_and_unchecked_leaf_payloads(validate):
    """Do not introduce payload, numeric, tuple-only, or unique-owner checks."""
    leaf = RecipeModel(None, strength="unchecked", block_config=object())
    shared = RecipeMerge(BASE, RecipeCompose([LORA, leaf]), LORA, "unchecked")
    tree = RecipeMerge(shared, RecipeCompose((shared, shared)), shared, None)
    assert validate(tree) is None


# AC: @exit-node ac-2
# AC: @clip-exit-node ac-6
def test_first_error_order(validate):
    bad_base = RecipeMerge(BASE, RecipeCompose(()), None, 1.0)
    tree = RecipeMerge(bad_base, 42, 42, 1.0)
    assert_result(validate, tree, "RecipeCompose at root.base.target has no branches")
    tree = RecipeMerge(BASE, RecipeCompose((RecipeCompose(()), 42)), 42, 1.0)
    assert_result(validate, tree, "RecipeCompose at root.target.branches[0] has no branches")


# AC: @exit-node ac-2
# AC: @clip-exit-node ac-6
@pytest.mark.parametrize("domain", ["clip", "diffusion", "unknown", None])
@pytest.mark.parametrize("position", ["root", "base", "target-base", "branch-base", "backbone"])
def test_domain_matrix(domain, position):
    base = RecipeBase(object(), "unchecked-arch", domain=domain)
    trees = {
        "root": (base, "custom"),
        "base": (RecipeMerge(base, LORA, None, 1.0), "custom.base"),
        "target-base": (
            RecipeMerge(BASE, RecipeMerge(base, LORA, None, 1.0), None, 1.0),
            "custom.target.base",
        ),
        "branch-base": (
            RecipeCompose((LORA, RecipeMerge(base, MODEL, None, 1.0))),
            "custom.branches[1].base",
        ),
        "backbone": (RecipeMerge(BASE, LORA, base, 1.0), "custom.backbone"),
    }
    tree, path = trees[position]
    assert _validate_recipe_tree(tree, "custom") is None
    assert validate_recipe_tree(tree, "custom") is None
    message = (
        None
        if domain == "clip"
        else (
            f"RecipeBase at {path} has domain='{domain}', expected domain='clip'. "
            "Use CLIP Entry node to create CLIP recipes."
        )
    )
    assert_result(_validate_clip_recipe_tree, tree, message, path="custom")
    assert_result(
        partial(validate_recipe_tree, expected_domain="clip"), tree, message, path="custom"
    )


# AC: @clip-exit-node ac-6
def test_domain_error_precedes_invalid_target():
    tree = RecipeMerge(RecipeBase(object(), "sdxl"), 42, 42, 1.0)
    message = (
        "RecipeBase at root.base has domain='diffusion', expected domain='clip'. "
        "Use CLIP Entry node to create CLIP recipes."
    )
    for validate in (
        _validate_clip_recipe_tree,
        partial(validate_recipe_tree, expected_domain="clip"),
    ):
        assert_result(validate, tree, message)


def test_explicit_expected_domain():
    """The optional domain parameter compares exactly; it is not a domain registry."""
    assert validate_recipe_tree(BASE, expected_domain="clip") is None
    assert_result(
        partial(validate_recipe_tree, expected_domain="diffusion"),
        BASE,
        "RecipeBase at root has domain='clip', expected domain='diffusion'.",
    )
