"""Structural recipe validation shared by diffusion and CLIP Exit nodes."""

from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel, RecipeNode


def validate_recipe_tree(
    node: RecipeNode, path: str = "root", *, expected_domain: str | None = None
) -> None:
    """Raise ValueError naming the invalid type/domain and its tree position.

    AC: @exit-node ac-2
    AC: @clip-exit-node ac-6
    None leaves domains unchecked (diffusion Exit's existing behavior).
    Root execution restrictions and leaf payload validation belong to callers.
    """
    if isinstance(node, RecipeBase):
        if expected_domain is not None and getattr(node, "domain", "diffusion") != expected_domain:
            guidance = (
                " Use CLIP Entry node to create CLIP recipes." if expected_domain == "clip" else ""
            )
            raise ValueError(
                f"RecipeBase at {path} has domain='{getattr(node, 'domain', 'diffusion')}', "
                f"expected domain='{expected_domain}'.{guidance}"
            )
    elif isinstance(node, (RecipeLoRA, RecipeModel)):
        return
    elif isinstance(node, RecipeCompose):
        if not node.branches:
            raise ValueError(f"RecipeCompose at {path} has no branches")
        for i, branch in enumerate(node.branches):
            branch_path = f"{path}.branches[{i}]"
            if not isinstance(branch, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
                raise ValueError(
                    f"Invalid branch type at {branch_path}: expected RecipeLoRA, "
                    f"RecipeModel, RecipeCompose, or RecipeMerge, got {type(branch).__name__}"
                )
            validate_recipe_tree(branch, branch_path, expected_domain=expected_domain)
    elif isinstance(node, RecipeMerge):
        base_path = f"{path}.base"
        if not isinstance(node.base, (RecipeBase, RecipeMerge)):
            raise ValueError(
                f"Invalid base type at {base_path}: expected RecipeBase or "
                f"RecipeMerge, got {type(node.base).__name__}"
            )
        validate_recipe_tree(node.base, base_path, expected_domain=expected_domain)

        target_path = f"{path}.target"
        if not isinstance(node.target, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            raise ValueError(
                f"Invalid target type at {target_path}: expected RecipeLoRA, "
                f"RecipeModel, RecipeCompose, or RecipeMerge, got {type(node.target).__name__}"
            )
        validate_recipe_tree(node.target, target_path, expected_domain=expected_domain)

        if node.backbone is not None:
            validate_recipe_tree(
                node.backbone, f"{path}.backbone", expected_domain=expected_domain
            )
    else:
        raise ValueError(f"Unknown recipe node type at {path}: {type(node).__name__}")
