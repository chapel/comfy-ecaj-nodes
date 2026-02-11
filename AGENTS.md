# Agent Guide

## What This Project Is

**comfy-ecaj-nodes** is a ComfyUI custom node pack for advanced model merging. The first (and flagship) feature set implements **WIDEN-based merging** — weight disentanglement for intelligent parameter-level model composition. Unlike simple linear interpolation, WIDEN analyzes per-parameter importance across models and routes each parameter to the most-relevant contributor.

The node pack uses a **deferred execution architecture**: recipe-building nodes (Entry, LoRA, Compose, Merge) construct a lightweight recipe tree with zero GPU work. The Exit node receives the complete recipe and runs the full batched GPU pipeline in one shot, preserving optimal `OpSignature` batching and `torch.bmm` LoRA application.

The core algorithm is ported from the **merge-router** project (`~/Projects/merge-router`). The design document lives at `docs/design.md`.

It uses **kspec** (Kynetic Spec) for specification and task management. Spec files in `.kspec/` define what to build; tasks track the work.

## Finding Information

AGENTS.md provides **project architecture, gotchas, and decision frameworks**. For detailed workflows and command syntax, use skills and CLI help:

| Need | Where to look |
|------|---------------|
| CLI command syntax | `kspec help <command>` or invoke `/kspec` skill |
| Task lifecycle (start → submit → PR → complete) | `/task-work` skill |
| Creating PRs | `/pr` skill, then `/pr-review` for merge gates |
| Spec authoring (items, ACs, traits) | `/spec` skill |
| Plan-to-spec translation | `/spec-plan` skill |
| Session context (focus, threads, observations) | `/meta` skill |
| Inbox/observation processing | `/triage` skill |
| Pre-PR quality checks | `/local-review` skill |
| Session reflection | `/reflect` skill |
| Comprehensive audit | `/audit` skill |
| Creating workflows | `/create-workflow` skill |
| WIDEN algorithm (source) | `~/Projects/merge-router/src/core/widen.py` |
| Design document | `docs/design.md` |

Skills inject their full documentation when invoked — you don't need to memorize their contents.

## Quick Start

```bash
# Install kspec (if not installed globally)
npm install -g kynetic-spec

# Initialize kspec (creates shadow branch)
kspec init

# Configure agent environment
kspec setup

# Get session context
kspec session start
```

### Development Environment

- **Python 3.12** with `uv` package manager
- **PyTorch** with CUDA (RTX 4090, 24GB VRAM)
- **ComfyUI** installation required for testing nodes

```bash
# Install dependencies
uv pip install -r requirements.txt

# Link into ComfyUI custom_nodes/ for testing
ln -s $(pwd) /path/to/ComfyUI/custom_nodes/comfy-ecaj-nodes
```

## Essential Rules

1. **Use CLI, not manual YAML edits** — Never manually edit files in `.kspec/`. CLI auto-commits to shadow branch.
2. **Spec before code** — If changing behavior, check spec coverage. Update spec first if needed.
3. **Add notes** — Document what you do in task notes for audit trail.
4. **Check dependencies** — Tasks have `depends_on` relationships; complete prerequisites first.
5. **Always confirm** — Ask before creating or modifying spec items.
6. **Batch mutations** — Use `kspec batch` for 2+ sequential write operations (one atomic commit).

## Project Structure

```
comfy-ecaj-nodes/
├── .kspec/                      # Spec/task state (shadow branch worktree)
│   ├── comfy-ecaj-nodes.yaml    # Root manifest
│   ├── comfy-ecaj-nodes.tasks.yaml
│   └── modules/                 # Spec items by domain
├── .claude/skills/              # 13 skill definitions
├── nodes/                       # ComfyUI node definitions
│   ├── entry.py                 # Entry node (MODEL → WIDEN)
│   ├── lora.py                  # LoRA node (file + strength → WIDEN)
│   ├── compose.py               # Compose node (branch accumulator)
│   ├── merge.py                 # Merge node (recipe builder, t_factor)
│   └── exit.py                  # Exit node (recipe executor → MODEL)
├── lib/                         # Core algorithm library
│   ├── widen.py                 # WIDEN algorithm (ported from merge-router)
│   ├── divergence.py            # Divergence metrics
│   ├── ranking.py               # Ranking mechanisms
│   ├── numerical_config.py      # Numerical stability config
│   ├── recipe.py                # Recipe tree dataclasses
│   ├── executor.py              # Batched pipeline executor
│   └── lora/                    # Architecture-specific LoRA handling
│       ├── base.py              # Loader interface
│       ├── zimage.py            # Z-Image QKV fusing, key mapping
│       └── sdxl.py              # SDXL key mapping
├── examples/                    # Example ComfyUI workflow JSONs
├── tests/                       # Test suite
├── __init__.py                  # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
├── pyproject.toml               # ComfyUI registry metadata
└── requirements.txt             # Python dependencies
```

## Node Architecture

### Five Nodes, One Custom Type

All WIDEN nodes communicate via a single custom ComfyUI type: **`WIDEN`**. This type wraps lightweight recipe dataclasses — no GPU tensors, pure data.

| Node | Inputs | Output | Purpose |
|------|--------|--------|---------|
| **Entry** | `MODEL` | `WIDEN` | Boundary in: snapshot base model, auto-detect architecture |
| **LoRA** | file selector, strength, optional `prev` | `WIDEN` | Declare LoRA spec (chains via `prev` to form sets) |
| **Compose** | `branch`, optional `compose` | `WIDEN` | Accumulate branches for simultaneous merge |
| **Merge** | `base`, `target`, optional `backbone`, `t_factor` | `WIDEN` | Define a WIDEN merge step in the recipe |
| **Exit** | `WIDEN` | `MODEL` | Execute full batched pipeline, return merged model |

### Deferred Execution

Entry, LoRA, Compose, and Merge do **zero GPU work**. They build a recipe tree. The Exit node receives the complete recipe and runs the full batched GPU pipeline:

1. Walk recipe tree → assign synthetic set IDs → load LoRAs
2. Group parameters by `OpSignature` (same shape, same affecting sets)
3. Batched GPU LoRA apply via `torch.bmm`
4. Batched WIDEN merge (filter_delta for single-target, merge_weights for compose)
5. Install results as `"set"` patches on a `ModelPatcher.clone()`

### Key Design Decisions

- **Architecture auto-detection**: Entry node inspects state dict keys → stores arch tag in recipe
- **LoRA node is our own loader**: Deferred loading enables batched bmm apply at Exit time
- **Merge node is universal**: compose target → `merge_weights`, single target → `filter_delta`
- **Chain is implicit**: Merge output feeds next Merge's base input
- **Backbone override**: Optional Merge input for explicit WIDEN importance reference

### Supported Architectures

- **SDXL** — native ComfyUI support
- **Z-Image** — custom loader (fused QKV attention, Diffusers-style LoRA keys)
- **Flux** — planned
- **Qwen** — planned

## Shadow Branch Architecture

`.kspec/` is NOT a regular directory — it's a **git worktree** on an orphan branch (`kspec-meta`).

```
.kspec/.git → file pointing to worktree
  ↓
gitdir: .git/worktrees/-kspec
  ↓
Shadow branch (kspec-meta): orphan branch with spec/task files
```

**Why:** Spec/task changes don't clutter main branch history. Code PRs and spec changes tracked independently.

**How it works:** Every `kspec` command auto-commits to `kspec-meta`. Main branch gitignores `.kspec/`.

**CRITICAL: Always run kspec from project root, never from inside `.kspec/`.** If you see "Cannot run kspec from inside .kspec/ directory", check `pwd`.

### Shadow Branch Commands

```bash
kspec shadow status   # Verify health
kspec shadow repair   # Fix broken worktree
kspec shadow sync     # Sync with remote
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `.kspec/` doesn't exist | `kspec init` |
| Worktree disconnected | `kspec shadow repair` |
| Sync conflicts | `kspec shadow resolve` |
| Commands seem broken | Check `pwd` — must be project root |

## Key Concepts

### IDs and References

Every item has a ULID (canonical) and slug (human-friendly). References use `@` prefix: `@task-slug` or `@01JHNKAB`.

### Spec Items vs Tasks

- **Spec items** (`.kspec/modules/*.yaml`): Define WHAT to build
- **Tasks** (`.kspec/comfy-ecaj-nodes.tasks.yaml`): Track the WORK of building

Tasks reference specs via `spec_ref`. They don't duplicate spec content.

### Task States

```
pending → in_progress → pending_review → completed
              ↓              ↓
          blocked ←──────────┘
              ↓
          cancelled
```

See `kspec help task` for transition commands and options.

## Working on This Project

```bash
kspec session start   # Get context, check for existing work
```

### Task Workflow

1. **Verify**: Check if work is already done (git history, existing code). If so, complete with reason.
2. **Start**: `kspec task start @ref` — mark in_progress before working.
3. **Note**: `kspec task note @ref "..."` — add notes as you work, not just at end.
4. **Complete**: `kspec task complete @ref --reason "Summary"` — mark done.

### Creating Work

- **Clear scope?** → Create task directly
- **Unclear scope?** → `kspec inbox add "idea"` → triage later with `/triage`
- **Behavior change?** → Check/update spec first, then derive task

### The Full Loop

1. `kspec session start` — get context, inherit existing work
2. `kspec task start @ref` — mark in_progress
3. Implement, add notes as you go
4. `kspec task submit @ref` — mark pending_review
5. `/local-review` → `/pr` → `/pr-review` — quality gates, create PR, merge
6. `kspec task complete @ref` — after PR merged
7. New tasks unblock, repeat

For the full task lifecycle, use `/task-work`.

## Session Context & Observations

Track four dimensions to maintain continuity across sessions: **focus** (current work), **threads** (parallel streams), **questions** (open decisions), **observations** (learnings).

```bash
kspec meta focus "Implementing @task-slug"
kspec meta thread add "Background: investigating performance issue"
kspec meta question add "Should we support Flux architecture in v1?"
```

Four observation types capture what you notice **during work**:
- **friction**: Things that didn't work, gotchas, blockers
- **success**: Patterns that worked well, useful approaches
- **question**: Clarifications needed, process decisions
- **idea**: Thoughts that emerge but aren't actionable yet

```bash
kspec meta observe friction "ComfyUI's LoRA loader doesn't handle Z-Image QKV fusing"
kspec meta observe success "OpSignature batching gives 8x speedup over per-key"
```

**Inbox vs observations**: Observations document what you noticed (learning/reflection). Inbox captures what you might do (potential work).

For full session context workflow, run `/meta`.

## Spec-First Development

**Core principle**: If you're changing behavior and the spec doesn't cover it, update the spec first.

| Situation | Flow |
|-----------|------|
| Clear behavior change | Check spec → Update/create spec → Derive task |
| Vague idea, unclear scope | Capture in inbox → Triage later |
| Infra/internal (no user impact) | Create task directly, no spec needed |
| Bug revealing spec gap | Fix bug → Update spec to match reality |

### Plan Mode Workflow

When a plan is approved, you MUST translate it to specs before implementing:

1. Create spec item: `kspec item add --under @parent --title "Feature" --type feature`
2. Add acceptance criteria: `kspec item ac add @spec --given "..." --when "..." --then "..."`
3. Derive task: `kspec derive @spec`
4. Add implementation notes to task
5. Begin implementation

**Plans without specs are incomplete.** The spec with ACs IS the durable artifact.

After plan approval, run `/spec-plan` for the full translation workflow.

## Staying Aligned During Work

**Watch for scope expansion:**
- Modifying files outside your current task
- Adding functionality the spec doesn't mention
- "While I'm here, I should also..." thoughts

**When you notice something outside your task:** Capture it separately (inbox item, new task, or observation). Add a note to your current task documenting what you found. Don't fix it inline — stay on your task.

## PR Workflow

Before creating a PR, mark the task: `kspec task submit @ref` (transitions to `pending_review`).

The full PR lifecycle has three steps — **all required, in order:**

1. **`/local-review`** — Quality gates: AC coverage, test quality. Run this FIRST.
2. **`/pr`** — Create the pull request.
3. **`/pr-review`** — Review and merge.

**Quality gates (never skip without explicit approval):**
- All CI checks passing
- All review comments addressed
- AC coverage verified

**After merge:** `kspec task complete @ref --reason "Merged in PR #N. Summary..."`

## Commit Convention

```
feat: Feature description

Task: @task-slug
Spec: @spec-ref
```

Trailers enable `kspec log @ref` to find commits by task or spec.

## Code Annotations

Link tests to acceptance criteria:

```python
# AC: @spec-item ac-N
def test_something():
    ...
```

Every AC SHOULD have at least one test with this annotation.

## Porting from merge-router

When porting code from `~/Projects/merge-router`, follow these guidelines:

- **Port, don't symlink** — this is a standalone project
- **Strip CLI/config concerns** — ComfyUI nodes replace JSON configs
- **Keep algorithm code clean** — `lib/` should have no ComfyUI imports
- **Architecture-specific code goes in `lib/lora/`** — one module per arch
- **Recipe dataclasses in `lib/recipe.py`** — frozen, immutable, no tensors
- **Node definitions in `nodes/`** — thin wrappers calling into `lib/`

### Key Source Files to Port

| merge-router | comfy-ecaj-nodes | Notes |
|---|---|---|
| `src/core/widen.py` | `lib/widen.py` | Core algorithm, batched methods |
| `src/core/divergence.py` | `lib/divergence.py` | Divergence metrics |
| `src/core/ranking.py` | `lib/ranking.py` | Ranking mechanisms |
| `src/core/numerical_config.py` | `lib/numerical_config.py` | Stability config |
| `scripts/lora_chain_merge.py` (batched pipeline) | `lib/executor.py` | Adapted for recipe tree |
| `scripts/zimage_lora_merge.py` (LoRA loading) | `lib/lora/zimage.py` | QKV fusing, key mapping |
| `scripts/sdxl_lora_merge.py` (LoRA loading) | `lib/lora/sdxl.py` | Key mapping |

### What NOT to Port

- `merge.py` (JSON config runner — replaced by node graph)
- CLI argument parsing
- Safetensors save logic (ComfyUI handles this)
- Training infrastructure

## Environment

- `KSPEC_AUTHOR` — Attribution for notes (e.g., @claude)
- Run `kspec setup` to configure automatically
