---
name: multimodal-expansion
description: Add audio, speech, or vision modalities to an existing LLM. Covers three approaches — adapter-based (LLaVA-style), REAP + expert replacement, and EvoMoE-inspired specialization — with training pipelines, tool options, and GPU feasibility assessments.
---

# Multimodal Expansion: Adding Modalities to an Existing LLM

Trigger when: user wants to add audio, speech, vision, or video capabilities to a language model that doesn't have them. Especially relevant for MoE models where expert-level approaches are on the table.

## Quick Decision Tree

```
Want to add audio/speech to an LLM?
├─ Text-only model, no vision? → Full LLaVA-style: add both vision + audio encoders
├─ Already has vision (like Qwen3.6 35B A3B)? → Just add audio encoder + projector
├─ Want speech OUTPUT too? → Add Talker module OR use external TTS (simpler)
├─ Want native MoE integration (not bolt-on)? → See approach B or C below
└─ Limited compute (dual 3090 class)? → Approach A with LoRA is the only viable path

Building a multi-modal hub with multiple specialist models?
├─ Same backbone family? → Merge shared transformer layers (Darwin MRI-Trust)
├─ Different architectures (T5 + DiT + Qwen)? → Keep specialists routed, merge only the common backbone
├─ "One base model for everything"? → Anchor on a dense 3B Omni (text + vision + audio native), route all generation to specialists
└─ Need self-evolving architecture? → Darwin (weight-space) + GEPA (parameter-space) optimization loop
```

## Three Approaches

### 🅰️ Approach A: Adapter-Based (LLaVA Pattern) — MOST FEASIBLE

Add modality-specific encoders + projectors. The LLM stays mostly frozen.

**Architecture:**
```
[Audio] → [Audio Encoder] → [Projector/Adapter] → [LLM (frozen or LoRA)] → text
                                                                          → [Talker] → speech (optional)
```

**Training Pipeline (3 stages):**
1. **Encoder Alignment** — Freeze LLM. Train encoder + projector on paired data (e.g. ASR: speech→text). ~10K hours.
2. **Multimodal Fine-Tune** — LoRA on LLM. Diverse audio tasks: ASR, captioning, QA, music, sound events. ~1M examples.
3. **Speech Output** (optional) — Train Talker on Thinker hidden states, OR use external TTS (ElevenLabs, Edge TTS).

**Audio Encoder Options:**
- Whisper-large-v3 (easy, well-supported)
- Qwen-Audio encoder (better Chinese/multilingual)
- AuT encoder from Qwen3-Omni (best but 20M hrs training data not public)

**Tools:** [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) — open-source framework for speech/audio/music → LLM training. Has pretrained checkpoints and training recipes.

**GPU for dual 3090 (48GB):** Stage 1 fits on single 3090. Stage 2 with LoRA r=64 fits on dual 3090 for 3B active models. Full fine-tune needs cloud.

### 🅱️ Approach B: REAP + Expert Replacement — MOST AMBITIOUS

Prune least-used MoE experts via REAP, then replace them with modality-specialized experts.

**REAP Formula:** S_j = (1/|X_j|) * Σ g_j(x) · ‖f_j(x)‖₂
- g_j(x): router gate-value for expert j on input x
- f_j(x): output of expert j
- Prune experts with lowest S_j scores

**Pipeline (4 stages):**
1. **REAP Calibration** — Run data through model, score 256 experts, prune bottom 30-50%. Needs full FP16 model (~70GB) — cloud GPU.
2. **Expert Initialization** — Init new experts. Options: clone existing + noise, copy from Omni Thinker (probably incompatible — Omni uses standard transformers, Qwen3.6 uses DeltaNet+MoE), or random init.
3. **Router Retraining** — Freeze experts, train router to route audio tokens → new experts.
4. **Full Multimodal FT** — Unfreeze everything. Needs A100/H100 class.

**Key Risk:** Architecture compatibility. MoE experts from different model families (standard transformer vs DeltaNet) are almost certainly dimension-mismatched.

**No existing tools.** Would require building calibration, expert init, router training from scratch.

### 🅲 Approach C: EvoMoE-Inspired Specialization — MIDDLE GROUND

Based on [EvoMoE](https://arxiv.org/abs/2505.23830) (AAAI 2026). Let existing experts naturally specialize through staged training — no pruning or swapping.

**Pipeline (3 stages):**
1. **Warm-up** — Multimodal instruction data to familiarize model with audio tokens.
2. **Expert Diversification** — Contrastive learning pushes experts to specialize per modality.
3. **Router Refinement** — Fine-tune router for optimal expert selection per input type.

**Key Insight:** The router learns to route audio tokens to specific experts organically. No architectural changes needed (beyond adding the audio encoder + projector, same as Approach A).

**Paper only, no released code.** Feasibility uncertain.

## Model Architecture Reference

See `references/qwen-architecture-comparison.md` for detailed specs and the full Senter Omni architecture decision log.

## Senter Omni Architecture (As-Built)

### Components Confirmed Local
| Component | Path | Type | Key Specs |
|-----------|------|------|-----------|
| Omni thinker | `/home/sovthpaw/Models/hf/Qwen2.5-Omni-3B/` | Qwen2.5-3B + NaViT + Whisper | 36L, hidden=2048, heads=16, KV=2, vocab=151936 |
| Lance 3B | `/home/sovthpaw/Models/senter-omni/lance/weights/Lance_3B/` | Qwen2.5-VL-4B backbone + DiT head | 36L, hidden=2048, heads=16, KV=2, vocab=151936 |
| ACE-Step LM | `/home/sovthpaw/Models/senter-omni/ace-step-lora/weights/ace-step-lm-qwen3-4b/` | Qwen3-4B | 36L, hidden=2560, heads=32, KV=8, vocab=217204 |
| ACE-Step DiT+UMT5 | `/home/sovthpaw/Models/senter-omni/ace-step-lora/weights/` | Diffusion + UMT5 encoder | Music generation specialist only |

### Critical Discovery: Lance is pre-trained MoE
Every layer in Lance has `_moe_gen` twin weights (e.g. `self_attn.q_proj` + `self_attn.q_proj_moe_gen`). These are a second expert per layer trained for generation tasks. This means Lance already contains a native 36-layer MoE — we just need to formalize it with a learned router.

### Darwin→mergekit Two-Stage Pipeline
```
Stage 1 (Darwin):  Merge P1(Omni) + P2(Lance) text bodies → unified child checkpoint
                     Same-arch merge: hidden=2048, heads=16, 36L MATCH PERFECTLY
Stage 2 (mergekit): Build MoE + Dense variants from the unified child
                     MoE: 9 experts (per-layer), ~3B active
                     Dense: smaller for resource-constrained inference
```

### Key Insight: "One encoder" means ONE encoder
The user explicitly rejected swapping Lance's ViT in. The architecture uses:
- Omni's NaViT vision encoder (unchanged)
- Omni's Whisper-based audio encoder (unchanged)
- Omni+Lance merged text body (shared reasoning)
- Lance's pre-existing `_moe_gen` weights as the generation expert
- ACE-Step Qwen3-4B LM as a routed music specialist (projection adapter at boundary)

### Why ACE-Step can't be weight-merged into Qwen2.5
Different hidden sizes (2560 vs 2048), vocab sizes (217204 vs 151936), and attention head configs (32/8 vs 16/2). Darwin Architecture Mapper could theoretically bridge this but requires research-grade implementation. Keep ACE-Step as a routed expert with projection adapters at the dispatch boundary.

## Project State

See `references/senter-omni-project-state.md` for the live directory layout, downloaded weights, and next-step checklist for the Senter Omni build.

## REAP Technique Reference

See `references/reap-technique.md` for the full REAP methodology, performance benchmarks, and code links.

## Pitfalls

- **Always verify architecture compatibility before merging.** The ACE-Step 1.5 LM IS Qwen3-based (confirmed from model card), not UMT5-based. The UMT5 is just a lyric conditioning encoder. Don't assume LM = whole model — check the actual config.json and model card before declaring incompatibility.
- **Verify facts before claiming "impossible."** When user asked about Qwen3.6 4B or Qwen3-4B existence, these models DO exist. The correct response is to look them up, not assert they don't.
- **MoE experts are FFN replacements, not modality handlers.** Audio understanding happens at the input encoding level — the encoder + projector, not inside the FFN experts. "Swapping an expert for an Omni" doesn't make architectural sense unless you replace the entire Thinker.
- **Omni thinker has submodules.** The full Omni model includes thinker, talker, token2wav, and vision/audio configs. Only `thinker.model.layers.0-35` overlaps with a Qwen2.5-VL text body. Including Omni-only layers (token2wav, talker) in a merge will crash. Filter to thinker.text body only.
- **Architecture compatibility between model families is low.** Qwen3.6 uses Gated DeltaNet + MoE; Qwen3-VL uses standard transformers; ACE-Step uses UMT5-T5; Lance uses Qwen2.5-VL backbone. None of these can be weight-merged with each other. Only models sharing the exact same backbone family (Qwen2.5 with Qwen2.5) can be merged.
- **"One base model" means ONE LLM backbone + ONE encoder stack.** The backbone handles all reasoning; the encoder handles all perception. Generation heads (music DiT, video DiT) are specialists that stay routed, not merged. The user wants ONE encoder — keep it Omni's native encoder, do NOT swap in Lance's ViT.
- **Specialist models with unique heads (DiT, VAE) cannot be weight-merged into an LLM.** ACE-Step's music decoder, Lance's video DiT — these are purpose-built. Route to them via a dispatcher layer instead.
- **Darwin→mergekit two-stage pipeline.** Darwin produces a unified base model from cross-architecture parents. mergekit then builds MoE + Dense variants from that unified base. Run Darwin FIRST, get stable generations, THEN do mergekit variants.
- **Lance is pre-trained MoE.** Every layer has `_moe_gen` twin weights — a second expert per layer trained for generation. This is not imposed architecture, it's pre-existing. The merge job is to formalize it with a learned router.
- **Speech output (Talker) is the hardest part.** External TTS is the pragmatic choice for Phase 1. Building a Talker requires the full Omni training pipeline.
- **User preference: action over planning.** When the user says "go" or "stop talking, start building," stop explaining and execute. Do not generate lengthy markdown plans when code/config/shell output is what's needed.
- **User preference: extend existing skills rather than inventing new ones.** Always check installed skills (e.g., herm-tui-radio for music, llama-cpp for local inference, multimodal-expansion for this exact domain) before building parallel implementations.
- **User preference: GPU resource awareness.** Dual 3090s with 24GB each. Monitor VRAM usage when staging models. Prefer GGUF quantization for local inference.
- **Multi-profile Discord pattern:** Each bot profile needs its own `.env` with its own `DISCORD_BOT_TOKEN`. Do NOT symlink `.env` to root — Discord kicks duplicate tokens. Keep `auth.json` symlinked (OAuth tokens are not Discord-scoped).

## Recommended Phased Approach

1. **Phase 1:** Approach A with Whisper encoder + LoRA on Qwen3.6 35B A3B. Use SLAM-LLM. External TTS for speech output.
2. **Phase 2:** Try Approach C (EvoMoE) once audio tokens flow through the model.
3. **Phase 3:** Approach B only with cloud compute budget — this is publishable research.
