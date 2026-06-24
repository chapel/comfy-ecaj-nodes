---
title: Krea 2 Technical Report Deep Dive
date: 2026-06-23
tags:
  - image-generation
  - diffusion-transformers
  - krea-2
  - data-curation
  - training
  - ablations
status: reviewed
source: https://www.krea.ai/blog/krea-2-technical-report
review:
  report_review: completed
---

# Krea 2 Technical Report Deep Dive

Primary source: [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report), Sangwu Lee, 2026-06-23.

Companion local source artifacts used for this note:

- `/tmp/krea2_report_text.txt` — extracted full report text.
- `/tmp/krea2_arxiv_selected.json` — selected arXiv metadata/abstracts for key referenced techniques.

## Executive summary

Krea 2 is less a single novel model trick than a full-stack image-generation system optimized for **creative exploration** rather than a narrow polished default aesthetic. The report’s core message is that modern text-to-image quality comes from the interaction of: broad/non-collapsed data curation, dense captioning, staged training, a stable/efficient DiT backbone, stronger text and latent representations, preference/RL post-training, and production-grade infrastructure.

The most transferable technical ideas are:

1. **Data mix > aesthetic monoculture.** Krea argues against blindly selecting only high aesthetic/IQA images. Their pretraining filters mostly remove duplicates, over-represented concepts, caption failures, artifact-inducing samples, excessive low-res complexity, and AI-generated images. They explicitly say synthetic images introduced bias and an upper bound on quality in their setting.
2. **Long captions for dense supervision, plus prompt-length diversity.** Krea uses OCR + metadata + VLM captioning, then a cheaper LLM to reformat captions into different lengths/styles. The model is trained predominantly on long captions while still seeing short/medium prompts.
3. **Midtraining as distribution bridge.** Pretraining is broad and bottom-up; midtraining is top-down by domain/source. Krea uses hierarchical k-means over visual embeddings and Wikipedia/PageRank-style entity coverage to keep rare concepts from disappearing.
4. **Architecture is a DiT with LLM-derived simplifications.** Final choices include SwiGLU, GQA, sigmoid-gated attention, single-stream transformer blocks, lightweight timestep bias modulation, 3D axial RoPE, zero-centered RMSNorm + QKNorm, Qwen 3 VL text encoding with layerwise feature aggregation, and Qwen Image / FLUX 2 autoencoders.
5. **Ablations are classified by stability, performance, efficiency, and simplicity.** Several ideas improved a local metric but were rejected for overhead or longer-horizon/high-resolution degradation: MLA had slight gains but extra overhead, hybrid stream slightly outperformed but single-stream won on simplicity, partial RoPE helped zero-shot 256→512 scaling but later underperformed, DC-AE gave high compression but capped fine detail, and Muon initially looked fast but needed exclusion of first/last linears plus Nesterov momentum to become stable.
6. **Training is staged and post-trained like modern LLMs.** Krea reports 256→512→1024 curriculum pretraining, rectified-flow loss under v-parameterization, iREPA for the first 256px epoch, 8-bit training at 256/512, shifted logit-normal timestep schedules, checkpoint/model merging (PMA), midtraining, SFT, STPO preference optimization, GRPO-style RL, and optional TDM timestep distillation.
7. **Steerability lives outside the base model too.** The prompt expander maps short prompts into model-friendly dense captions with SFT+RL and a diversity reward. The style-reference system uses self-supervised training to transfer style/mood while minimizing content leakage.
8. **Infrastructure is part of model quality.** The report spends substantial space on Kubernetes/Kueue scheduling, FSDP2 + Megatron-style tensor parallelism, torch.compile, Parquet/preshuffled dataloading, Weka checkpointing, GPU/IB observability, and a custom Postgres-sharded metadata/queue system (“krablet”). That is a signal that data/debug iteration speed was central to the research loop.

## Source confidence and non-claims

- The Krea report is the authority for what Krea claims they did. This note avoids inventing hidden implementation details that are not in the report.
- For referenced papers, I independently pulled metadata/abstracts for key sources where possible. Some Krea-cited 2025/2026 references may be very new; where I only have Krea’s summary, I label the statement as Krea’s claim rather than an independently established result.
- The report does **not** disclose full model size, exact dataset size beyond some stage-scale language, full reward formulas, exact STPO formulation, the style-reference method details, or exact final training compute. Any ablation derived from it should be treated as an idea source, not a guaranteed recipe.

## Krea 2 goal: exploratory generation

Krea frames the model family around escaping a “default aesthetic.” The target is not only photorealism or benchmark ranking but a model distribution with many styles, moods, compositions, and visual directions. That framing explains many otherwise odd decisions:

- do not overfit pretraining selection to aesthetic scorers;
- retain diverse/long-tail domains;
- train a prompt expander that preserves intent but increases visual detail;
- use diversity rewards so the expander does not collapse to a house style;
- build style-reference controls so users can navigate the model’s distribution visually.

Krea says Krea 2 is top-10 on the Artificial Analysis text-to-image leaderboard and second among independent labs, but the report’s technical value is mostly in the ablation logic and systems design, not the leaderboard claim.

## Data curation deep dive

### Pretraining principles

Krea’s pretraining curation philosophy is: remove data that harms alignment or induces artifacts, but do not narrow the distribution to only conventionally “beautiful” images. They argue that aesthetic/IQA filters encode biases: motion blur, softness, unusual lighting, or intentional noisy texture can be artistically useful even if a generic quality model dislikes them. They also make a subtler argument: if a caption accurately describes an undesirable image or behavior, that sample can still help the model understand that behavior so later stages can steer away from it.

Krea’s stated removal categories:

- duplicated samples and over-represented concepts;
- images where VLM captions consistently miss important aspects;
- samples that induce undesired biases/artifacts;
- images too visually complex to model at the active low resolution;
- AI-generated samples.

The “no AI-generated images” claim is important: Krea reports that even a small synthetic fraction biased the output distribution because synthetic images were easier to learn, imposing an effective quality ceiling. They used in-house classifiers to filter synthetic images.

Important nuance: Krea is not anti-filtering. As resolution increases, they introduce image-quality and aesthetic filters, but use those scores only to drop extremely poor-quality images rather than oversampling high scorers. They also adjust image-complexity and OCR/text-density thresholds over training progression, because some images/text-heavy samples cannot be meaningfully represented at low resolution but may become usable later.

### Captioning pipeline

Krea’s captioning pipeline has three meaningful pieces:

1. **OCR first.** Extract visible text before captioning.
2. **Context-rich VLM captioning.** Feed OCR plus metadata such as camera settings and known entities into a captioning model so captions include world knowledge and extracted text.
3. **LLM reformatting.** Use a cheaper LLM to rewrite long captions into multiple lengths and prompt styles.

Krea reports that long captions gave dense supervision, faster convergence, and lower training loss. But because real users write short/medium prompts, they intentionally expose the model to shorter prompts too.

A practical interpretation: the long-caption distribution is the training teacher; the prompt expander is the inference-time bridge from user prompts into that teacher distribution.

### Low-resolution pretraining filters

At the billion-image low-resolution stage, Krea emphasizes CPU-cheap filters:

- broken file / resolution / aspect-ratio filters;
- Laplacian filters for extreme texture/noise patterns;
- RGB entropy;
- white/black pixel ratios;
- custom heuristics and in-house classifiers for flat backgrounds and border artifacts.

For task-specific filters, Krea describes using a large VLM to craft a filtering prompt, pseudo-label examples, then train a smaller DINOv3- or SigLIP-2-based classifier under 1B parameters for scale.

### Deduplication

Krea combines md5, perceptual hash, and colorhash. They specifically found default 8×8 phash too color-blind and false-positive-prone, so they combine **12×12 phash + colorhash**.

### SAE-based unsupervised artifact tagging

Krea trains a sparse autoencoder (SAE) over SigLIP-2 embeddings from a sample of the pretraining corpus. Then a VLM annotates each SAE feature using top-k activating images. The resulting features become an unsupervised tagging/filtering system for visual artifacts without having to train one classifier per artifact.

Referenced concept: SAEs are used in interpretability to decompose dense activations into sparse, more human-interpretable features. Krea applies the same idea to dataset embeddings rather than LLM internals.

### Midtraining data

Midtraining is Krea’s bridge between broad pretraining and high-quality SFT. The report distinguishes:

- **pretraining:** broad bottom-up corpus construction;
- **midtraining:** top-down source/domain selection with better style/domain coverage;
- **SFT:** small hand-curated aesthetic/domain datasets.

Midtraining curation uses:

- FAISS hierarchical k-means, inspired by *Automatic Data Curation for Self-Supervised Learning*;
- VLM naming/flagging of cluster centroids;
- human review for problematic clusters;
- semantic dedup inside leaf clusters using SigLIP similarity;
- English Wikipedia PageRank via Danker: retain the top 90% of English Wikipedia articles by rank, filter unrepresentable subjects through Wikidata metadata, then full-text-search captions for the remaining ~5 million representable concepts.

The notable lesson is that **cluster-balanced sampling alone can drop rare named entities** when they sit inside head clusters. Krea adds explicit entity coverage analysis to avoid that, prioritizes rare-concept captions during sampling, and repeats coverage analysis after sampling to confirm concepts present in the initial dataset were not dropped entirely.

### SFT data

SFT uses small, hand-curated datasets focused on visual domains. Krea says quality matters more than scale once there is enough volume. SFT improved checkpoint quality and addressed high saturation/texture issues in earlier checkpoints. They also merge domain-specific SFT checkpoints into a generalist checkpoint, while noting later-stage merging has diminishing returns because directions conflict.

## Architecture deep dive

Krea says they evaluated architecture choices against four goals:

- **Stability:** lower loss/gradient spikes.
- **Performance:** faster convergence and high-resolution robustness.
- **Efficiency:** fewer parameters/FLOPs/memory/communication without quality loss.
- **Simplicity:** remove complexity if quality is unchanged.

### Baseline → ablation → final component table

| Area | Baseline | Ablations | Krea final |
|---|---|---|---|
| Attention | multi-head attention | GQA, MLA, gated sigmoid attention | GQA + gated sigmoid attention |
| MLP | GeLU MLP | SwiGLU | SwiGLU |
| Residual | standard residual | value residual, Laurel | standard residual |
| Text encoder | T5-XXL | T5Gemma, Qwen 2.5 VL, Qwen 3 VL, umT5 | Qwen 3 VL |
| Timestep modulation | per-block MLP modulation | light modulation with bias | light modulation with bias |
| Autoencoder | FLUX AE | Qwen Image VAE, DC-AE, FLUX 2 AE, internal VAE | Qwen Image VAE early; FLUX 2 AE larger models |
| Stream design | single stream | hybrid stream, parallel single stream | single-stream final |
| Norm | LayerNorm, QKNorm | RMSNorm, zero-centered RMSNorm, Derf | zero-centered RMSNorm + QKNorm |
| Position | 3D axial RoPE | Golden Gate RoPE, MRoPE, normalized RoPE, partial RoPE | 3D axial RoPE |

### Transformer block

Krea starts by replacing GeLU MLP with **SwiGLU** at 4× expansion. The referenced GLU variants paper showed gated feed-forward variants can outperform standard ReLU/GELU transformer FFNs. Krea reports consistent performance gains, so all later ablations use SwiGLU.

Attention changes:

- **GQA** reduces KV heads compared with full multi-head attention, improving efficiency with minimal degradation. The GQA paper originally targeted decoder inference efficiency, but Krea uses the efficiency/stability idea in DiTs.
- **MLA** gave slight gains over GQA but introduced overhead, so Krea rejected it. Their MLA variant used up/down projection for KV compression and omitted decoupled RoPE because diffusion inference is pure prefill rather than KV-cache decoding.
- **Gated sigmoid attention** adds a head-specific sigmoid gate after SDPA. The cited paper finds this improves LLM stability, sparsity, and attention-sink behavior. Krea says it did not substantially improve performance but stabilized loss/gradient curves with little overhead.

Stream design:

- **Single-stream:** shared attention/MLP weights for text and image tokens.
- **Dual-stream:** joint attention, separate attention/MLP weights per modality.
- **Hybrid-stream:** dual-stream for the first third, single-stream for the remaining two-thirds.

Krea reports no significant differences except hybrid was slightly better, but they chose single-stream for simplicity. This matters: the report is not saying dual-stream/MMDiT is bad; it says their ablation did not justify complexity for their final model.

### Timestep conditioning

Many MMDiTs use per-block MLPs to generate scale/shift/gates. Krea says those MLPs can be 20–30% of parameters, too large for injecting a scalar condition. They replace them with a **per-block tunable bias term**, freeing parameters for attention and MLP layers without sacrificing performance.

They also tested:

- removing timestep conditioning entirely — underperformed at low-res pretraining;
- in-context timestep tokens — 4–16 tokens replaced AdaLN at 256px, but underperformed at 512/1024 even with more tokens.

This is a direct warning against assuming “time tokens instead of AdaLN” scales automatically.

### Positional encoding

Krea uses **3D axial RoPE** with head dimensions dedicated to frame, height, and width. Text tokens get RoPE index zero.

Ablations:

- Golden Gate RoPE;
- MRoPE;
- normalized RoPE;
- partial RoPE.

Partial RoPE helped zero-shot 256→512 resolution scaling and reduced duplication artifacts, but later underperformed after high-resolution training. Krea kept 3D axial RoPE.

### Autoencoder

Krea starts from FLUX.1-dev AE and benchmarks Qwen Image VAE, DC-AE, FLUX 2 AE, and an internal AE.

Important findings:

- DC-AE’s high spatial compression can improve efficiency but Krea found reconstruction error imposed a hard ceiling on fine detail.
- Qwen Image VAE and FLUX 2 AE converged faster while preserving reconstruction quality.
- An internal VAE trained with DINOv3 semantic alignment and light diffusion loss, similar to REPA-E, performed competitively, but they used Qwen Image / FLUX 2 because those were already validated at scale.

The broad lesson is that latent representation quality can dominate apparent DiT training speed; autoencoder reconstruction ceilings should be measured before blaming the transformer.

### Residual and normalization

Krea kept standard residuals. Laurel did not visibly improve, but they list NOBLE, delta attention residuals, attention residuals, and mHC as future directions.

For normalization, Krea replaces LayerNorm with RMSNorm and uses zero-centered RMSNorm plus QKNorm. RMSNorm removes mean-centering and is computationally simpler; QKNorm normalizes query/key vectors to stabilize attention logits. Krea says Derf was more efficient but degraded quality.

### Text encoder and multilayer feature aggregation

Krea’s baseline text encoder was T5-XXL. They tested T5Gemma, umT5, Qwen 2.5 VL, and Qwen 3 VL. They emphasize that **T5-XXL remained very competitive**, so this should not be read as “T5 is obsolete.” They pick **Qwen 3 VL** because it offers a richer input space (text and image) and stronger multilingual generalization.

The more interesting part is not just picking Qwen 3 VL, but how they use it:

- Last-layer autoregressive LLM features are optimized for next-token prediction, not image generation.
- Inspired by UniFusion, Krea introduces a shallow attention layer over hidden states from multiple VLM layers.
- This lets the image model dynamically select coarse-to-fine text representations.
- They add lightweight bidirectional transformer layers along token positions to reduce autoregressive bias.

This is a strong signal that **text-feature layer aggregation** may be more valuable than simply swapping encoders.

### Optimizer

Krea mostly uses AdamW but reports a positive Muon result after fixing parameter selection and momentum:

- initial Muon on MMDiT: faster early steps, worse longer horizon, frequent loss/grad spikes;
- they use Dion implementation and Moonlight RMS-matched settings;
- crucial fix: exclude first and last linear layers from Muon parameters, analogous to excluding embeddings/LM head in LLM Muon setups;
- add Nesterov momentum;
- after this, Muon outperformed AdamW at low and high resolution;
- not adopted in latest run only due to time constraints.

Author bridge to the companion AsymFlow plan: this lines up with local optimizer notes in [[AsymFlow i1, Pyramid, and Fast Ablation Plan]], where Muon-family comparisons are sensitive to absolute matrix LR, fallback parameter grouping, and long-horizon drift. Krea’s result suggests the optimizer idea is still alive, but only with careful exclusions and long-horizon evals.

## Training and post-training deep dive

### Pretraining

Krea progressively scales resolution from 256px to 512px to 1024px. The low-res stage learns structure, alignment, text rendering, style coverage, and core capabilities; high-res stages add fidelity. Krea explicitly says they dedicate the majority of FLOPs to low-resolution stages to build core capabilities efficiently before spending high-resolution compute.

Final model training uses standard rectified-flow loss under **v-parameterization**. Rectified flow and flow matching learn vector fields that transport noise to data; flow matching provides simulation-free training of continuous normalizing flows. Krea uses these in the modern latent-DiT family rather than classical DDPM-only training.

### iREPA acceleration

Krea uses **iREPA** for the first epoch of 256px pretraining, then removes it so the MMDiT learns its own representations. The cited iREPA/REPA line asks which representation-alignment targets help generation. The key abstract-level lesson is that spatial structure in representation targets can matter more than global classification strength. Krea treats iREPA as an early-convergence accelerator, not a permanent crutch. They also explored TREAD as an alternative acceleration strategy but saw little benefit.

### 8-bit training

Krea reports 8-bit training at 256px and 512px gave 15–20% speedups over bf16 with minimal degradation in loss/eval metrics. They use tensorwise scaling at 256px and rowwise scaling at 512px. From 1024px onward and through RL, they return to bf16.

### Timestep sampling / timeshift

Krea uses a shifted logit-normal timestep schedule for both training and inference, increasing the shift as resolution increases. They sweep the optimal **training** timeshift at each resolution and keep inference shift constant, citing FLUX 2 VAE sensitivity differences.

This suggests timestep distribution is not a one-time global hyperparameter; it should be coupled to resolution and representation.

### PMA / checkpoint merging

During pretraining, Krea uses a warmup-stable-decay LR schedule and PMA following a model-merging-in-pretraining paper. They say PMA reaches EMA-comparable performance without EMA’s memory overhead. Merge interval and number of merged checkpoints can give small downstream gains.

This is relevant where EMA memory is costly or where post-hoc EMA reconstruction is unavailable.

### Preference optimization: STPO

Krea’s PO stage has two stages:

1. large-scale synthetic preference-pair generation, mostly including at least one on-policy sample, similar to delta learning;
2. human-annotation calibration by internal annotators familiar with model quirks.

They identify a DPO failure mode: the model can satisfy margin objectives by reducing likelihood of both preferred and dispreferred samples at different rates, drifting from pretraining and creating high-frequency artifacts. Their STPO variant adds an auxiliary loss and modifies DPO to reduce divergence.

No exact STPO formula is disclosed. The safe takeaway is: preference optimization needs an explicit anti-divergence guard and artifact monitoring.

### RL with GRPO-style multi-reward optimization

Krea’s final RL stage uses a multi-reward GRPO-style method with reward models for:

- general aesthetics;
- prompt following;
- text rendering;
- artifact/structural integrity.

They fine-tune an open-source VLM on PO preference data for the aesthetic reward. For prompt following, they use prompt-specific rubric rewards: decompose a prompt into verifiable requirements rather than ask for one holistic judge score. They separately train an artifact reward model for extra fingers, malformed limbs, distorted text, and similar structural failures that general VLM judges may miss.

They also curate the RL prompt pool as a resource-allocation problem:

- deprioritize prompts that are too easy, too hard, saturated, or low-variance;
- keep diverse styles/concepts/settings/subjects;
- mine hard-but-not-hopeless prompts with useful reward variance.

For CFG, Krea trains the RL stage without CFG to keep rollout/training distributions aligned and avoid unnecessary overhead. CFG remains an inference knob.

### Timestep distillation

Krea optionally distills steps after RL, applying guidance distillation and timestep distillation together. They considered DMD, DMD2, Decoupled DMD, piFlow, APT, and selected **Trajectory Distribution Matching (TDM)** because it is data-free, flexible for multistep students, and simpler to tune than GAN-ish or multi-timestep-prediction approaches.

DMD matches clean-image distributions; TDM applies distribution matching across timesteps/trajectories. Krea’s stated goal is flexible multistep distillation, not necessarily one-step generation.

### Prompt expansion

Krea’s prompt expander solves the training/inference prompt-distribution gap. It maps short, underspecified prompts into rich model-friendly captions.

Training:

- SFT an open-source LLM on synthetic user-prompt → long-caption pairs;
- synthesize thinking traces to preserve reasoning/intent reconstruction;
- oversample visually rich/artistic imagery and add a light photographic-medium bias;
- RL optimize expansions through resulting image quality and user-intent preservation using GDPO;
- use prompt-level verifiable rewards and safety/constraint gates;
- use realistic prompts plus mined hard cases;
- keep a DINOv3 embedding diversity reward active to avoid house-style collapse.

Key lesson: prompt expansion should be optimized for downstream generated images, not just caption imitation.

### Style reference system

Krea trains a style-reference module on top of the base model to allow style/mood injection with adjustable strength and weighted mixing. The main failure mode is content leakage from reference images. Krea says they use a novel self-supervised technique plus preference optimization, but does not disclose enough detail to reproduce it.

## Infrastructure and data systems

Krea’s training stack:

- PyTorch, DTensor, torch-native Torchtitan-related features;
- FSDP2 + Megatron-LM-style tensor parallelism;
- async tensor parallelism for TP > 2;
- autoencoder replicated; text encoder and MMDiT sharded;
- NVLinkSharp intra-node and InfiniBand inter-node;
- torch.compile as main optimization;
- cuDNN attention by default, FlexAttention or FlashAttention 3 as needed;
- selective activation checkpointing at low resolution, full checkpointing at high resolution.
- a deliberately slightly wider model / larger hidden dimension, because higher compute intensity helped hide FSDP2 prefetch latency, reduced all-gather/reduce-scatter count by using fewer layers, reduced NCCL-related errors, and amortized 8-bit quant/dequant overhead.

Cluster scheduling / inference coexistence:

- research and production inference shared one Kubernetes cluster;
- when training claimed the full local GPU pool, production inference migrated outside the cluster;
- Kueue provided gang scheduling plus borrowing/lending/reclamation semantics, though manual GPU-count configuration became an operational annoyance;
- Krea used a Virtual Kubelet-style external inference layer so Kubernetes could schedule pods onto provider-backed virtual nodes and let normal HPA/reconciliation semantics replace failed replicas.

Training launch operations:

- large runs often needed known-faulty nodes excluded before launch;
- the launch CLI evolved to select clean nodes, apply labels/affinity, and sometimes taint/drain them for maximum stability;
- the faulty-node list moved from a text file to node labels;
- “Packerman” packed dev machines onto faulty nodes, reserving healthy nodes for training.

Data loading:

- Parquet rows with image reference, crop/resize dimensions, captions, metadata;
- preshuffle and pack rows so each dataloader worker loads batches with same aspect ratio;
- sequential disk scan + global shuffling + reproducible replay;
- pre-crop/resize large images to target resolution for homogeneous CPU/I/O/GPU load.

Reliability:

- Krea optimized MTTR with fast/frequent checkpointing rather than perfect fault prevention;
- Weka replaced Ceph and allowed ~30 second checkpoints;
- tensor core utilization (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) was their preferred health metric;
- GPU temperature above roughly 75–78°C increased instability through throttling;
- `DCGM_FI_DEV_GPU_UTIL` was often misleading because it reports kernel-active time and usually showed symptoms rather than root cause;
- FB_USED helped catch allocation stalls where a few GPUs stuck near ~5 GiB while others initialized normally;
- XIDs, remapped-row metrics, and PCIe replay counters were useful when present, especially for memory or motherboard/GPU connection faults;
- custom NVLink metrics filled gaps in DCGM coverage;
- InfiniBand instability was the largest crash source, requiring custom metrics for CRC, symbol errors, link flaps, congestion, port waits/errors, and throughput disparities.

Scale reliability lesson: Krea reports that runs below 128 GPUs were often stable for days, but very large runs became much flakier and did not complete a run longer than 24 hours without crashing. Their practical response was not to eliminate every failure, but to make restart/checkpoint/debug loops fast enough.

Data infrastructure:

- “krablet” system: sharded PostgreSQL metadata store + queue semantics;
- all dataset metadata and object-storage keys live in Postgres shards;
- reported scale: 208 TB of metadata and tens of thousands of contended UPSERTs per second;
- workers claim work with `FOR UPDATE SKIP LOCKED` and update retry timestamps atomically;
- crashed workers do not lose rows;
- partial/incremental processing is natural;
- researchers get immediate visibility in dashboards;
- pluck API supports notebook-style global map over the same queue semantics.

The research lesson is that **continuous data curation needs a stateful, inspectable queue**, not a one-shot batch job.

## Referenced technique map

### Diffusion Transformer / DiT

[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) replaces U-Net backbones with transformers over latent patches and shows quality scaling with forward-pass complexity. Krea’s model is squarely in this lineage.

### Flow matching / rectified flow

[Flow Matching](https://arxiv.org/abs/2210.02747) trains vector fields for continuous normalizing flows without simulation, and [Rectified Flow](https://arxiv.org/abs/2209.03003) learns straighter transport paths from noise to data. Krea uses rectified-flow loss with v-parameterization.

### GQA

[GQA](https://arxiv.org/abs/2305.13245) generalizes multi-query attention by using fewer KV heads than query heads. Krea ports the efficiency idea into DiT attention.

### Gated sigmoid attention

[Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) studies sigmoid-gated SDPA variants and reports stability/performance benefits in LLMs. Krea adopts it primarily for stability.

### SwiGLU

[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) motivates gated feed-forward layers. Krea reports consistent performance gains over GeLU MLP.

### RoPE / axial RoPE

[RoFormer / RoPE](https://arxiv.org/abs/2104.09864) encodes position by rotating query/key features and induces relative-position behavior. Krea uses a 3D axial variant for frame/height/width.

### Autoencoders and representation alignment

[Qwen-Image](https://arxiv.org/abs/2508.02324) is an image foundation-model report whose abstract emphasizes text rendering and editing. Krea specifically uses the **Qwen Image VAE** as one of its strong autoencoder choices. [DC-AE](https://arxiv.org/abs/2410.10733) explores high-compression autoencoders; Krea found detail ceilings despite efficiency. [REPA-E](https://arxiv.org/abs/2504.10483) suggests representation alignment can unlock VAE+DiT end-to-end tuning.

### Qwen3-VL and UniFusion

[Qwen3-VL](https://arxiv.org/abs/2511.21631) provides multimodal long-context text/image/video understanding. [UniFusion](https://arxiv.org/abs/2510.12789) conditions image generation on frozen VLM features and motivates layerwise attention pooling instead of using only the final VLM layer.

### Data curation and embeddings

[Automatic Data Curation for Self-Supervised Learning](https://arxiv.org/abs/2405.15613) motivates hierarchical clustering for diverse/balanced pretraining sets. [SigLIP 2](https://arxiv.org/abs/2502.14786) provides stronger multilingual/localized vision-language embeddings. [Sparse Autoencoders](https://arxiv.org/abs/2309.08600) provide the feature-discovery idea Krea adapts for artifact tags.

### Muon

[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) argues matrix-orthogonalized updates can scale with weight decay and per-parameter update-scale handling. Krea’s practical finding is that diffusion transformers need careful parameter exclusions and Nesterov momentum.

### Preference and RL methods

[Diffusion-DPO](https://arxiv.org/abs/2311.12908) adapts DPO-style preference optimization to diffusion models. [GRPO](https://arxiv.org/abs/2402.03300) is originally from DeepSeekMath’s LLM RL setting; Krea says they use a GRPO-style multi-reward method and also cites flow/diffusion RL work such as [Flow-GRPO](https://arxiv.org/abs/2505.05470). [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) explains the conditional/unconditional guidance mechanism Krea keeps as inference-only in RL.

### Distillation

[DMD](https://arxiv.org/abs/2311.18828), [DMD2](https://arxiv.org/abs/2405.14867), and [TDM](https://arxiv.org/abs/2503.06674) are few-step diffusion distillation approaches. Krea selected TDM for flexible multistep distillation.

## Architecture/training ablation lessons extracted from Krea

These are the decision patterns most worth carrying into local ablation planning:

1. **Separate “slightly better” from “worth complexity.”** Krea rejected MLA and hybrid stream despite gains/possible gains because overhead/complexity did not justify adoption.
2. **Evaluate over horizon and resolution.** Partial RoPE looked good for zero-shot resolution transfer but lost after continued high-res training. Muon looked good initially but needed fixes for long-horizon stability.
3. **Measure representation ceilings.** DC-AE efficiency did not matter if reconstruction error capped fine detail. Any latent/projector choice needs a reconstruction ceiling diagnostic.
4. **Treat timestep conditioning as a scaling issue.** In-context time tokens worked at 256px but not high-res. Do not overgeneralize low-res wins.
5. **Use text-feature aggregation, not just encoder swapping.** VLM layer aggregation addresses the mismatch between next-token final layers and image-generation conditioning.
6. **Data curation can be an ablation axis.** Caption length mix, prompt style mix, dedup thresholds, rare-concept sampling, and artifact tag filters are controllable variables.
7. **Post-training can create artifacts.** Preference/RL gains need anti-divergence and artifact-specific rewards/eval.
8. **Diversity needs explicit protection.** Both data curation and prompt expansion include diversity-preserving mechanisms.
9. **Infrastructure affects research truth.** Fast checkpointing, exact data replay, and source-level metadata are what make spike/debug iterations reliable.

## Discussion and future work from Krea

Krea’s future-work section is useful because it says which parts of Krea 2 they consider conservative or unfinished:

- **Scaling:** Krea says the Krea 2 architecture/optimizer choices were relatively conservative for stability and iteration speed. Next-cycle directions include MoE, native 2K–4K resolution, sparse attention, NVFP4 pretraining, and scaling Muon.
- **Undertraining:** they state current models would benefit from longer training.
- **Multi-teacher on-policy distillation (MOPD):** Krea wants domain experts trained by separate RL teams, then distilled into one student without capability conflicts. They say they have internally validated OPD/MOPD for diffusion models but do not share details.
- **Architectural simplification:** Krea calls out the operational cost of multi-component image stacks: autoencoder, DiT, text encoder, prompt expander, style-reference model, upscaler, etc. They want future systems to unify more components under one model.
- **New capabilities:** editing, image reference, native high-resolution generation, and more native comprehension of non-natural-language prompt formats such as tags, JSON, bounding boxes, visual guidelines, and Markdown.

For AsymFlow, the important interpretation is not “copy Krea’s next stack”; it is that even a strong production model sees simplification, sparse high-res attention, Muon scaling, and richer prompt formats as unresolved research directions.

## Open questions / not disclosed by Krea

- Exact Krea 2 parameter counts, token counts, model widths/depths, and training FLOPs.
- Exact dataset size per stage except broad scale language such as billions at low-res.
- Exact STPO objective.
- Exact style-reference self-supervised method.
- Exact Qwen3-VL layer aggregation architecture beyond “shallow attention” and bidirectional token layers.
- Exact reward model datasets and reward weights.
- Whether any reported architecture result would transfer to native-pixel / non-latent AsymFlow settings without latent VAE assumptions.

## Author commentary: implications for AsymFlow planning

This section is a bridge from the source-grounded Krea report to the companion local plan, [[AsymFlow i1, Pyramid, and Fast Ablation Plan]]. It is not a claim that Krea tested AsymFlow directly. The Krea report should not cause a wholesale architecture pivot by itself. It mostly supports a **backlog of targeted ablation ideas** around:

- lightweight timestep-conditioning alternatives;
- QKNorm / zero-centered RMSNorm / gated sigmoid attention;
- text/VLM layer aggregation and bidirectional token mixing;
- Muon-family parameter-exclusion and Nesterov controls;
- shifted logit-normal / resolution-aware timestep sampling;
- PMA/checkpoint-merge alternatives to EMA;
- data/caption mix ablations: long/medium/short caption ratio, OCR/metadata-rich captions, dedup/hash policy, rare-concept/domain-balanced sampling, and artifact-tag filtering;
- reconstruction-ceiling diagnostics for any projector/latent-like representation.

These are best treated as **secondary ablation backlog**, not the immediate core path, because the current AsymFlow plan already has major architecture/objective/routing questions in flight.

## Source index

Primary:

- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)

Core generation/model papers cited or discussed:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis / FLUX](https://arxiv.org/abs/2403.03206)
- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)

Architecture/component papers:

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) — cited by Krea for QKNorm context.
- [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://arxiv.org/abs/2410.10733)
- [REPA-E](https://arxiv.org/abs/2504.10483)
- [UniFusion](https://arxiv.org/abs/2510.12789)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)

Data/curation papers:

- [Automatic Data Curation for Self-Supervised Learning](https://arxiv.org/abs/2405.15613)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)
- [DINOv3](https://arxiv.org/abs/2508.10104)

Training/post-training papers:

- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [DMD](https://arxiv.org/abs/2311.18828)
- [DMD2](https://arxiv.org/abs/2405.14867)
- [Trajectory Distribution Matching](https://arxiv.org/abs/2503.06674)
- [TREAD](https://arxiv.org/abs/2501.04765)
- [Model Merging in Pre-training of Large Language Models](https://arxiv.org/abs/2505.12082)
