# Spec Reconciliation: Exit Output-Mode Scoping

Task: @task-full-model-update-existing-specs
Spec: @full-saved-model-output
Plan: @plan-full-saved-model-output-plan

## Purpose

Scope existing Exit specs so set-patch behavior is identified as the default
in-memory patch output mode, and full saved model output is a distinct
supported behavior. This prevents contradictions when both output modes coexist.

## Specs Updated

### @exit-node
- **Description**: Updated to mention output modes (in-memory patch default, others possible)
- **ac-1**: Scoped to in-memory patch output mode (the default)
- **ac-7**: Scoped to in-memory patch output mode (downstream LoRA additivity)
- **ac-8**: Scoped to in-memory patch output mode (patch dtype matching)
- ac-2 through ac-6 (validation, evaluation, chaining), ac-9/ac-10 (progress, RAM preflight): Universal — not scoped

### @exit-patch-install
- **Description**: Added, scoping entire spec to in-memory patch output mode
- **ac-1, ac-2, ac-3, ac-4, ac-7**: Scoped to in-memory patch output mode
- ac-5/ac-6 (IS_CHANGED cache validation): Universal — not scoped

### @exit-model-persistence
- **Description**: Updated for both output modes; clarifies cache-hit behavior differs per mode
- **ac-1**: Scoped to in-memory patch mode (no file I/O when save_model disabled)
- **ac-2**: Scoped to in-memory patch mode (save after GPU merge)
- **ac-3**: Updated to follow active output mode for post-cache-hit behavior
- **ac-10**: Scoped to in-memory patch mode (atomic writes)
- ac-4 through ac-9, ac-11 through ac-14: Universal cache/metadata behavior — not scoped

### @memory-management
- **Description**: Updated to reference @streaming-full-model-materialization and @comfy-memory-manager-compatibility
- **ac-7, ac-8, ac-13**: Scoped to in-memory patch mode (save streaming, GPU offload, base_state release)
- ac-1 through ac-6, ac-9 through ac-12, ac-14: Universal GPU evaluation memory — not scoped

## Relationships Added

| From | To | Type |
|------|----|------|
| @full-saved-model-output | @exit-model-persistence | relates_to |
| @full-saved-model-output | @memory-management | relates_to |
| @full-saved-model-output | @exit-patch-install | relates_to |
| @streaming-full-model-materialization | @exit-model-persistence | relates_to |
| @comfy-memory-manager-compatibility | @memory-management | relates_to |

## Tags Added

- @full-saved-model-output: persistence, memory, comfy
- @streaming-full-model-materialization: persistence, memory
- @comfy-memory-manager-compatibility: comfy, memory, compatibility

## Validation

- `kspec validate`: Schema OK, References OK
- 1194 existing tests pass
- No contradictory requirements across output modes
