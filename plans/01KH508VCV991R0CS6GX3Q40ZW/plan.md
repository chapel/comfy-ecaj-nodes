# Testing Strategy

Restructure @testing-infrastructure into a parent feature with three sub-requirements,
add a CI spec, and establish a project convention requiring AC-annotated test coverage
for every implementation task.

## Specs

```yaml
- title: Testing Infrastructure
  slug: testing-infrastructure
  type: feature
  parent: "@foundation"
  description: |
    Comprehensive testing strategy for comfy-ecaj-nodes. Covers pytest
    configuration, ComfyUI mocking, recipe fixtures, node graph validation
    via mock entry/exit, and CI pipeline. Parent feature for all testing
    sub-requirements.
  acceptance_criteria:
    - id: ac-1
      given: a developer runs pytest from the project root
      when: tests execute
      then: all tests pass without requiring a running ComfyUI instance
    - id: ac-2
      given: any implementation task is completed
      when: its tests are inspected
      then: each spec AC has a corresponding test annotated with AC @spec-ref ac-N

- title: ComfyUI Mocking and Fixtures
  slug: comfyui-mocking
  type: requirement
  parent: "@testing-infrastructure"
  description: |
    pytest conftest.py with sys.modules mocking for ComfyUI imports,
    MockModelPatcher class, and recipe tree fixtures. The base layer that
    all other tests depend on.
  acceptance_criteria:
    - id: ac-1
      given: a test needs a ModelPatcher-like object
      when: it uses the mock_model_patcher fixture
      then: |
        the mock provides model_state_dict(filter_prefix) returning a dict
        of small fake tensors keyed like diffusion_model.input_blocks.0.0.weight,
        clone() returning a new MockModelPatcher, add_patches() storing patches,
        get_key_patches() returning patch data, and patches_uuid property
    - id: ac-2
      given: a test needs a recipe tree
      when: it uses recipe fixtures
      then: |
        pre-built recipe trees are available for single-LoRA, multi-LoRA set,
        compose (2 branches), chain (2 sequential merges), and full
        (compose + chain) patterns
    - id: ac-3
      given: tests for nodes that import ComfyUI modules like folder_paths
      when: they run without ComfyUI installed
      then: |
        ComfyUI modules are mocked via sys.modules patching in conftest.py
        before any node module is imported
    - id: ac-4
      given: a test needs fake SDXL or Z-Image state dict keys
      when: it uses arch-specific fixtures
      then: |
        fixture provides a dict with representative key patterns for each
        supported architecture (input_blocks for SDXL, layers + noise_refiner
        for Z-Image)
  implementation_notes: |
    Use ComfyUI's own pattern from tests-unit/ and ComfyUI_Selectors:
    sys.modules patching in conftest.py before node imports. MockModelPatcher
    should use small tensors (4x4 float32) for speed. Recipe fixtures build
    on lib/recipe.py dataclasses. Arch fixtures provide representative state
    dict key sets for detection testing.
    Files: tests/conftest.py, tests/mocks/__init__.py, tests/mocks/mock_comfy.py

- title: Node Graph Testing
  slug: node-graph-testing
  type: requirement
  parent: "@testing-infrastructure"
  description: |
    Integration tests that validate the recipe graph building pipeline.
    Uses mock entry node to feed RecipeBase into the node chain, validates
    recipe tree structure through LoRA/Compose/Merge, and uses a mock
    executor path in Exit to verify the tree would produce correct operation
    sequences (filter_delta vs merge_weights) without GPU execution.
  acceptance_criteria:
    - id: ac-1
      given: a mock Entry node producing a RecipeBase with arch sdxl
      when: wired to LoRA node then to Merge node
      then: |
        the resulting RecipeMerge contains the correct base (RecipeBase)
        and target (RecipeLoRA with the specified LoRA) and t_factor
    - id: ac-2
      given: a recipe graph with compose target containing 3 branches
      when: the mock executor analyzes the tree
      then: it identifies this as a merge_weights operation (not filter_delta)
    - id: ac-3
      given: a recipe graph with single LoRA target
      when: the mock executor analyzes the tree
      then: it identifies this as a filter_delta operation
    - id: ac-4
      given: a chain of two Merge nodes (inner merge feeds outer base)
      when: the mock executor walks the tree
      then: it identifies inner merge must evaluate first and feeds into outer
    - id: ac-5
      given: an invalid recipe graph (e.g. RecipeBase wired to compose branch)
      when: validation runs
      then: a clear error is raised naming the invalid type and position
    - id: ac-6
      given: a complete graph matching the hyphoria workflow from design doc 6.5
      when: built and validated through the node chain
      then: the recipe tree structure matches the expected compose-merge-chain pattern
  implementation_notes: |
    Create tests/test_graph.py with helper functions that instantiate node
    classes and call their FUNCTION methods directly to build recipe trees.
    The mock executor is a lightweight tree walker (separate from the real
    executor) that returns an operation plan (list of {op: filter_delta|merge_weights,
    keys: ...}) without touching GPU. This validates the Exit node's recipe
    analysis logic independently.
    Files: tests/test_graph.py, tests/helpers/graph_builder.py

- title: CI Pipeline
  slug: ci-pipeline
  type: requirement
  parent: "@testing-infrastructure"
  description: |
    GitHub Actions workflow for automated testing. Initial scope is
    unit tests with CPU-only torch and ruff linting on push/PR. Designed
    to be extended later with comfy-test registration and workflow smoke tests.
  acceptance_criteria:
    - id: ac-1
      given: a push to any branch or a PR is opened
      when: GitHub Actions runs
      then: |
        pytest executes with CPU-only PyTorch on ubuntu-latest and all
        tests pass
    - id: ac-2
      given: the CI workflow
      when: linting step runs
      then: ruff check passes with no errors
    - id: ac-3
      given: CI completes
      when: results are reported
      then: PR shows green check for both test and lint jobs
  implementation_notes: |
    Follow ComfyUI's test-unit.yml pattern: install CPU-only torch via
    --index-url https://download.pytorch.org/whl/cpu, install project deps,
    run pytest. Add ruff for linting. Single ubuntu-latest runner to start
    (extend to matrix later). Add pyproject.toml [tool.ruff] config.
    Files: .github/workflows/test.yml, pyproject.toml (ruff + pytest config)
```

## Tasks

derive_from_specs: true

## Implementation Notes

The existing @testing-infrastructure spec and task need to be restructured:
the current spec becomes the parent feature, its ACs are redistributed into
the new sub-requirements (comfyui-mocking absorbs the original ACs), and the
existing task is updated to reflect the new scope.

Additionally, a project convention should be established requiring AC-annotated
test coverage (# AC: @spec ac-N) for every implementation task. This is enforced
by the local-review workflow which checks for AC annotations in test files.
