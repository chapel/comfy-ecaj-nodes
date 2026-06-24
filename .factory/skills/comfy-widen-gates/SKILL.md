---
name: comfy-widen-gates
description: Project-specific validation and review gates for comfy-ecaj-nodes
  WIDEN/ComfyUI work.
---
<!-- kspec-managed -->
# Comfy WIDEN Gates

Project-specific gates for comfy-ecaj-nodes task work and review.

## When to use

Use this alongside `task-work` for implementation agents and alongside `review`/`merge` for review agents in this repository.

## Environment assumptions

- Use `uv` for local Python commands. If dependencies are missing, run `uv sync --extra dev` first rather than installing packages globally.
- Default tests must be CPU-safe. GPU/ComfyUI smoke tests are optional/local evidence unless a task explicitly scopes them.
- Do not require a running ComfyUI process for normal task completion or review gates.

## Required project checks

Run from the repository root unless the task explicitly scopes a smaller check.

1. Python syntax/import sanity:
   - `uv run python -m compileall lib nodes tests`
2. Unit tests:
   - `uv run pytest`
   - For narrow fixes, run the focused pytest slice first, then the broader suite before submission when feasible.
3. Lint/format checks:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
4. kspec validation after spec/task/meta changes:
   - `kspec validate --warnings-ok`

## Comfy/WIDEN-specific review points

- Entry/LoRA/Compose/Merge nodes must remain deferred: no tensor/GPU work before Exit.
- Architecture support belongs in architecture-specific loader/classifier modules, not ad hoc branching in generic executor code.
- LoRA loaders must fail or report unmatched groups explicitly; never silently skip shape/key mismatches.
- Tests must assert observable behavior. Do not add `pass`-only, `assert True`, or assertion-free test bodies.
- Tests covering acceptance criteria should include `# AC: @spec-ref ac-N` annotations where applicable.
- Do not commit local machine paths, model files, generated checkpoints, or artifacts from `/mnt/big-data`.
- Treat FP8/quantized checkpoint handling as explicit scope; do not silently merge quantization sidecars as normal weights.

## Submission/review evidence

Task notes or review summaries should name the focused tests and broad gates actually run, plus any skipped optional GPU/Comfy smoke with the reason.
