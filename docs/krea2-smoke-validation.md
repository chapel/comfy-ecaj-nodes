# Krea 2 Optional Smoke Validation

This workflow is optional evidence for Krea 2 support. Required tests remain CPU-safe and synthetic; real Krea 2 checkpoints, LoRAs, ComfyUI, and GPU capacity are operator-provided inputs.

## Public References

- Krea report: https://www.krea.ai/blog/krea-2-technical-report
- ComfyUI Krea 2 diffusion model source: https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/ldm/krea2/model.py
- ComfyUI Krea 2 text encoder source: https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/text_encoders/krea2.py
- Comfy-Org Krea 2 LoRAs: https://huggingface.co/Comfy-Org/Krea-2/tree/9b05e613f06f5ee45d97b362ba3478fec5488b5a/loras
- Krea RetroAnime LoRA: https://huggingface.co/krea/Krea-2-LoRA-retroanime/tree/23336fcff3bac43028918a2df795fbb63fdc1ff3

## Header Probe

Run the header-only probe before any live Comfy smoke. It reads safetensors headers only and records architecture evidence, Krea LoRA group compatibility, dtype/quantization warnings, and an optional skip reason.

```bash
COMFY_ECAJ_KREA2_PROBE=1 \
python scripts/manual/krea2_header_probe.py \
  --run-krea2-probe \
  --model-path /operator/provided/krea2-model.safetensors \
  --lora-path /operator/provided/krea2-lora.safetensors \
  --skip-reason "Comfy/GPU smoke not run on this machine" \
  --report-output reports/krea2-header-probe.json
```

Do not commit generated reports unless the task explicitly asks for durable evidence artifacts. Do not add local model directories to tests, specs, plans, or docs.

## Live Comfy Smoke

1. Link or install this repository into the operator's ComfyUI `custom_nodes` directory.
2. Place the operator-provided Krea 2 checkpoint and optional supported Krea 2 LoRA in ComfyUI model folders.
3. Start ComfyUI normally and verify the WIDEN nodes load.
4. Build a workflow with `WIDEN Entry -> optional WIDEN LoRA with Krea 2 block config -> WIDEN Merge or Compose -> WIDEN Exit`.
5. Use a small resolution and one or two sampler steps for a bounded smoke.
6. Record the ComfyUI version, model and LoRA filenames, whether header probe passed, whether the workflow queued, whether output completed, and any error text.

If GPU, assets, or ComfyUI are unavailable, record that exact skip reason and use the CPU-safe pytest run plus the header probe report as replacement evidence.
