"""WIDEN Compose Node — Accumulates branches for simultaneous merge."""


class WIDENComposeNode:
    """Appends branch to compose list. Pure data manipulation, no GPU work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "branch": ("WIDEN",),
            },
            "optional": {
                "compose": ("WIDEN",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "compose"
    CATEGORY = "ecaj/merge"

    def compose(self, branch, compose=None):
        raise NotImplementedError("Compose node not yet implemented")
