# Fix Incremental Recompute Memory Estimation

## Problem

The RAM preflight check (`estimate_peak_ram` / `check_ram_preflight`) uses a heuristic
formula that produces false-positive OOM rejections on incremental recompute runs while
passing correctly on full recompute runs after cache clear.

Root cause: the formula estimates **new** RAM allocations but gets two things wrong:

1. **Loader overhead is double-counted.** LoRA loaders are populated by `analyze_recipe()`
   *before* the preflight check runs. Their tensors are already in RAM and reflected in
   `MemAvailable`. The formula adds `0.1 * merged_state_bytes * n_models` as if loaders
   haven't loaded yet — pure double-count.

2. **The heuristic is disconnected from reality.** `0.1 * merged_state_bytes` bears no
   relation to actual loader sizes. A 128-rank SDXL LoRA file is ~600MB-1.2GB regardless
   of how many keys are being recomputed. The formula should use measured data, not a
   percentage of an unrelated quantity.

On full recompute (after cache clear), the double-count doesn't matter because
`MemAvailable` is high (cache freed). On incremental recompute, `MemAvailable` is tight
(cache holds ~model-size of tensors) and the inflated estimate triggers false rejections.

## Correct Memory Model

At the moment `check_ram_preflight` runs, these are **already in RAM** (reflected in
reduced `MemAvailable`):

- LoRA loader tensors (loaded during `analyze_recipe`)
- ModelLoader file handles + key mappings (streaming, negligible RAM)
- Cached `merged_state` tensors (from `_CacheEntry`, if incremental hit)
- `base_state` dict (references `model_patcher` data, not new alloc)
- ComfyUI runtime, OS, other processes

**Genuinely new allocations** during GPU eval:

| Allocation | Size | Lifetime |
|-----------|------|----------|
| Recomputed key tensors in `merged_state` | `recomputed_key_bytes` | Permanent (stored in cache) |
| `pin_memory` temporary copy | `2 * worst_chunk_bytes` | Per-batch, freed after transfer |
| `save_model` streaming overhead | `~0.05 * merged_state_bytes` | During save only |
| DeltaSpec objects | Negligible (tensor refs, ~bytes) | Per-batch |
| `torch.stack` in `execute_plan` | Already captured by `worst_chunk_bytes` | Per-batch |

**Note on the persistence cache-hit path:** When `check_cache()` returns a hit
(exit.py:467-476), the function returns early before loader loading or preflight.
This path involves no GPU work and no memory estimation — it's out of scope.

**Note on DeltaSpec "negligible":** DeltaSpec objects hold references to tensors
already in the loader, plus a few scalar fields. The per-object overhead is ~100 bytes.
For 1000 keys this is ~100KB — negligible vs tensor data. This is an approximation,
not a strict zero-cost invariant.

**Correct formula:**
```
peak_new = merged_state_bytes + 2 * worst_chunk_bytes + save_overhead
```

No loader term — they're already loaded. `merged_state_bytes` is already scoped
by the caller to only the keys being processed (full set on full recompute,
recompute subset on incremental path).

## Design: Measured Loader Bytes

Even though loaders are loaded before preflight (so their cost is in `MemAvailable`),
we want to **measure and record** actual loader memory for:

1. **Verification**: Assert that measured loader bytes match expectations
2. **Future use**: If preflight moves before loader loading, cached measurements enable
   accurate estimation without loading
3. **Diagnostics**: Log actual loader footprint for user troubleshooting

### Approach

- Add `loaded_bytes` property to `LoRALoader` base class (abstract)
- Each subclass implements by summing `tensor.nbytes` for **all** tensor-holding
  structures (e.g., `_lora_data_by_set`, `_qkv_data_by_set` in Flux/Z-Image)
- Add `loaded_bytes` property to `ModelLoader` (returns 0 — streaming/mmap)
- Add `loaded_bytes` property to `CLIPModelLoader` (returns 0 — streaming/mmap)
- Store measured `loader_bytes` in `_CacheEntry` metadata
- Pass measured value to `estimate_peak_ram` for logging/verification (not formula)

## Specs

```yaml
- title: Accurate RAM Preflight Estimation
  slug: accurate-ram-preflight
  type: requirement
  parent: "@memory-management"
  implementation_notes: |
    Core fix: remove heuristic loader_overhead from estimate_peak_ram formula.
    Replace with measured loader_bytes parameter used for verification/logging only.
    The formula becomes: peak = merged_state_bytes + 2*worst_chunk + save_overhead.
    Loaders are already loaded at preflight time so their cost is in MemAvailable.
    Update both exit.py and clip_exit.py call sites.
  acceptance_criteria:
    - id: ac-1
      given: LoRA loaders are populated before preflight check runs
      when: estimate_peak_ram calculates peak new RAM
      then: loader memory is not added to the estimate because it is already reflected in MemAvailable
    - id: ac-2
      given: incremental cache holds a previous merged_state of N bytes
      when: one block config changes requiring recompute of M bytes where M is much less than N
      then: preflight estimates peak new allocation as M bytes plus chunk and save costs, not N bytes
    - id: ac-3
      given: estimate_peak_ram receives measured loader_bytes
      when: the estimate is computed
      then: |
        the ecaj.gpu_ops logger emits a DEBUG message containing the loader_bytes value,
        and the returned peak estimate is identical regardless of whether loader_bytes is 0
        or any positive value
    - id: ac-4
      given: save_model is True
      when: estimate_peak_ram calculates peak
      then: save overhead is computed as 5 percent of merged_state_bytes (streaming one tensor at a time)
    - id: ac-5
      given: save_model is False
      when: estimate_peak_ram calculates peak
      then: no save overhead is added to the estimate
    - id: ac-6
      given: estimate_peak_ram is called with merged_state_bytes=M_full for full recompute and merged_state_bytes=M_sub for incremental recompute where M_sub is less than M_full, with all other parameters identical
      when: both estimates are compared
      then: the incremental estimate is strictly less than or equal to the full recompute estimate
    - id: ac-7
      given: clip_exit node runs preflight
      when: estimate_peak_ram is called
      then: the same corrected formula is used as in the diffusion exit node

- title: Loader Memory Measurement
  slug: loader-memory-measurement
  type: requirement
  parent: "@lora-loaders"
  implementation_notes: |
    Add loaded_bytes property to LoRALoader ABC and all subclasses.
    Each subclass sums tensor.nbytes across ALL tensor-holding data structures:
    - SDXLLoader, SDXLCLIPLoader, QwenLoader: _lora_data_by_set (up, down per key per set)
    - FluxLoader: _lora_data_by_set AND _qkv_data_by_set (QKV-fused up/down tensors)
    - ZImageLoader: _lora_data_by_set AND _qkv_data_by_set (LoKr + standard mixed storage)
    ModelLoader and CLIPModelLoader return 0 (streaming/mmap, no bulk tensor storage).
    This provides ground-truth measurement for verification, logging, and future preflight use.
    If future loaders add new tensor-holding structures (e.g., _lokr_data_by_set),
    loaded_bytes must be updated to include them.
  acceptance_criteria:
    - id: ac-1
      given: a LoRA loader has loaded one or more LoRA files
      when: loaded_bytes property is accessed
      then: returns the sum of tensor.nbytes for all tensors held in memory by the loader
    - id: ac-2
      given: a LoRA loader has not loaded any files
      when: loaded_bytes property is accessed
      then: returns 0
    - id: ac-3
      given: a LoRA loader loads two LoRA files with known tensor sizes
      when: loaded_bytes is accessed after both loads
      then: the value equals the sum of tensor.nbytes for all tensors held across both files including up, down, and any QKV-fused or LoKr component tensors
    - id: ac-4
      given: a ModelLoader is open with a safetensors file
      when: loaded_bytes property is accessed
      then: returns 0 because ModelLoader uses memory-mapped streaming access
    - id: ac-5
      given: a CLIPModelLoader is open with a safetensors file
      when: loaded_bytes property is accessed
      then: returns 0 because CLIPModelLoader uses memory-mapped streaming access
    - id: ac-6
      given: a LoRA loader has loaded files and cleanup is called
      when: loaded_bytes is accessed after cleanup
      then: returns 0 because all tensors have been released
    - id: ac-7
      given: a FluxLoader or ZImageLoader has loaded LoRA files containing both standard up/down pairs and QKV-fused tensors
      when: loaded_bytes is accessed
      then: the value includes tensor.nbytes from both _lora_data_by_set and _qkv_data_by_set structures

- title: Cache Loader Metadata
  slug: cache-loader-metadata
  type: requirement
  parent: "@incremental-block-recompute"
  implementation_notes: |
    Extend _CacheEntry with loader_bytes field (int). Populated from measured
    loader.loaded_bytes + sum of model_loader.loaded_bytes after loading.
    Available for future preflight improvements (e.g., pre-load estimation)
    and diagnostic logging. When cache is read on incremental hit, the stored
    loader_bytes is logged alongside the recompute plan.
  acceptance_criteria:
    - id: ac-1
      given: GPU evaluation completes and cache entry is stored
      when: _CacheEntry is created
      then: loader_bytes field contains the measured sum of all loader loaded_bytes at time of evaluation
    - id: ac-2
      given: an incremental cache hit occurs with stored loader_bytes of L
      when: the recompute plan is logged
      then: L is included in the log message for diagnostic purposes
    - id: ac-4
      given: an incremental cache hit occurs with stored loader_bytes of L
      when: preflight check runs for the recompute subset
      then: L does not affect the peak RAM estimate returned by estimate_peak_ram
    - id: ac-3
      given: cache entry exists with loader_bytes from a previous run
      when: current run loads different LoRA files with different sizes
      then: the new cache entry stores the new measured loader_bytes, not the old value
```

## Tasks

derive_from_specs: true

## Implementation Notes

### Execution order

1. `loader-memory-measurement` first (adds the measurement interface)
2. `cache-loader-metadata` second (stores measurements in cache)
3. `accurate-ram-preflight` last (fixes the formula, uses measurements for logging)

### Files affected

- `lib/lora/base.py` — add abstract `loaded_bytes` property
- `lib/lora/sdxl.py`, `sdxl_clip.py`, `flux.py`, `zimage.py`, `qwen.py` — implement `loaded_bytes`
- `lib/model_loader.py` — add `loaded_bytes` property (returns 0)
- `lib/clip_model_loader.py` — add `loaded_bytes` property (returns 0)
- `lib/gpu_ops.py` — update `estimate_peak_ram` signature and formula
- `nodes/exit.py` — update preflight call, store loader_bytes in cache
- `nodes/clip_exit.py` — update preflight call
- `tests/test_memory_management.py` — update/add tests for new formula
- `tests/test_lora_loaders.py` — add loaded_bytes tests
- `tests/test_incremental_recompute.py` — add cache metadata tests

### Backward compatibility

- `_CacheEntry` gains a new field. Existing cache entries (in-memory only, not
  persisted to disk) will simply miss on structural fingerprint mismatch since
  the cache is LRU-1 and in-process. No migration needed.

### What this does NOT change

- VRAM estimation (`compute_batch_size`) — separate concern, working correctly
- OOM backoff in `chunked_evaluation` — runtime safety net, unchanged
- `pin_memory` gating (`ac-9`) — unchanged, already correct
- Cache eviction under memory pressure (`ac-12`) — unchanged
