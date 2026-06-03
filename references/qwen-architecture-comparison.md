# Qwen Architecture Comparison

Key architecture differences between the models relevant to multimodal expansion decisions.

## Qwen3.6 35B A3B (May 2026)

- **Type:** Causal Language Model with Vision Encoder (multimodal: text + images)
- **Total Params:** 35B | **Active:** 3B
- **Hidden Dim:** 2048
- **Layers:** 40
- **Layer Pattern:** 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
- **Gated DeltaNet:** 32 V heads / 16 QK heads, head dim 128 (linear attention)
- **Gated Attention:** 16 Q heads / 2 KV heads, head dim 256, RoPE dim 64
- **MoE:** 256 experts, 8 routed + 1 shared, expert intermediate dim 512
- **Token Embedding:** 248,320 (padded)
- **Context:** 262K native, extensible to 1M
- **MTP:** Trained with multi-steps
- **License:** Apache 2.0
- **Already has:** Vision encoder (image input) ✅
- **Missing:** Audio input, speech output

Key: Gated DeltaNet is a linear attention variant — NOT standard transformer attention. This makes architecture compatibility with standard transformer models tricky.

## Qwen3-Omni-30B-A3B (Oct 2025)

- **Type:** Natively end-to-end omni-modal (text + images + audio + video → text + speech)
- **Architecture:** Thinker-Talker MoE
- **Thinker:** MoE LLM (Qwen3-based) + AuT audio encoder + vision encoder
- **Talker:** Dual-track autoregressive for streaming speech output
- **Audio Encoder:** AuT (Audio Transformer), trained on 20M hours of audio, 12.5Hz token rate
- **Speech Input:** 19 languages | **Speech Output:** 10 languages
- **Text:** 119 languages
- **Performance:** SOTA on 22/36 audio benchmarks, open-source SOTA on 32/36
- **Training:** 4-stage process for Talker
- **License:** Apache 2.0

Key: Uses standard Qwen3 MoE as Thinker, NOT DeltaNet architecture. AuT encoder is the secret sauce but not publicly available as standalone.

## Qwen3.5-Omni (Apr 2026)

- **Type:** Next-gen omni-modal, Hybrid Attention MoE for both Thinker AND Talker
- **Scale:** Hundreds of billions of parameters
- **Context:** 256K
- **Training Data:** 100M+ hours audio-visual content
- **Key Innovation:** ARIA — dynamically aligns text and speech units for stable streaming synthesis
- **Performance:** SOTA on 215 audio/visual benchmarks, surpasses Gemini 3.1 Pro on key audio tasks
- **New Capability:** Audio-Visual Vibe Coding (coding from audio-visual instructions)
- **Availability:** NOT open weights as of May 2026

## Qwen2.5-Omni-7B (Mar 2025)

- **Type:** End-to-end multimodal, Thinker-Talker architecture
- **Thinker:** Standard Qwen2.5 LLM
- **Talker:** Sliding-window DiT for streaming audio output
- **Position Encoding:** TMRoPE (Time-aligned Multimodal RoPE) for video-audio sync
- **Training Stages:**
  1. Encoder Alignment (freeze LLM, train encoders + adapters)
  2. Full Multimodal Fine-Tuning (unfreeze LLM)
  3. Long-Sequence Support
- **Available:** Open weights, Apache 2.0

## Qwen2.5-Omni-3B (Jan 2026, local)

- **Type:** End-to-end multimodal (text + images + audio + video → text + speech)
- **Architecture:** Standard Qwen2.5 transformer + Omni encoder (unified audio+image+video tokens)
- **Total Params:** 3B dense
- **Hidden Dim:** 2048
- **Layers:** 27 (standard Qwen2.5-3B backbone)
- **Audio:** Native streaming ASR + TTS, real-time interruption support
- **Speech Input:** Multilingual | **Speech Output:** Streaming
- **Vision:** Omni encoder handles image+video understanding
- **Status:** Local GGUF (Q4_K_M), actively used for Hermes agent integration
- **Key advantage:** Small enough for dual 3090, real-time speech, the "one base model" anchor

## Qwen3-VL-4B-Instruct (Apr 2026)

- **Type:** Vision-language model (text + images + video → text)
- **Architecture:** Qwen3-4B backbone + SigLIP2 encoder (300M params)
- **Total Params:** 4B dense
- **Layers:** 32
- **Vision:** SigLIP2-SO-400M (small) or SigLIP2-Large (2B/4B variants use Large)
- **Status:** Available on HuggingFace, GGUF community builds exist
- **Key advantage:** Newer vision encoder than Qwen2.5-VL, but text-only output (no audio, no native generation)

## ACE-Step 1.5 (Feb 2026, music generation specialist)

- **Type:** Hybrid LM-planner + diffusion-renderer for music generation
- **LM component:** Qwen3-based planners (confirmed from model card):
  - `acestep-5Hz-lm-0.6B` (Qwen3-0.6B)
  - `acestep-5Hz-lm-1.7B` (Qwen3-1.7B)
  - `acestep-5Hz-lm-4B` (Qwen3-4B) ← largest/best quality
- **UMT5 encoder:** T5-family encoder for lyric conditioning (NOT the primary LM — this is a secondary condition encoder)
- **DiT:** 2B (standard) / 4B (XL) diffusion transformer for audio rendering
- **VAE:** Music-specific 1D DC-AE + FSQ tokenizer, 5Hz frame rate
- **License:** MIT, commercial use permitted
- **Key insight:** The LM planners ARE Qwen3-based and can theoretically be routed to from a Qwen2.5 base via projection adapters. The UMT5 and DiT components are specialists that stay routed — they cannot be weight-merged with Qwen.
- **Hardware:** <4GB VRAM for inference, RTX 3090 generates full song in <10s
- **Architecture note:** The LM is the "omni-capable planner" that transforms queries into song blueprints. It has full CoT, audio understanding, and composition capability. This is the Qwen-family component.
- **Architecture note:** The LM is the "omni-capable planner" that transforms queries into song blueprints. It has full CoT, audio understanding, and composition capability. This is the Qwen-family component.

## Lance 3B (May 2026, ByteDance)

- **Type:** Native unified multimodal (text + images + video → understanding + generation + editing)
- **Architecture:** Qwen2.5-VL-derived, 36 layers, hidden_size 2048
- **Components:**
  - `Lance_3B/` — core LLM + ViT + text decoder (12GB safetensors)
  - `Lance_3B_Video/` — same + video decoder head (14GB safetensors)
  - `Qwen2.5-VL-ViT/` — separate Vision Transformer weights (1.3GB, 32 depth, 1280 hidden)
- **Vision encoder:** Qwen2.5-VL SigLIP2-style (depth 32, patch 14, temporal patch 2)
- **Generation:** Image-to-image, text-to-image, text-to-video, video editing
- **License:** Apache 2.0
- **Key insight:** Lance's LLM backbone shares Qwen2.5-VL-4B architecture (same hidden_size=2048, similar layer count). In theory, Darwin merge of the TRANSFORMER BACKBONE layers is possible. But Lance's generation heads (DiT decoders) are unique to Lance and cannot be weight-merged with Omni.

## Darwin Family (arXiv 2605.14386, May 2026)

- **Framework:** Training-free evolutionary merging via gradient-free weight recombination
- **Key mechanisms:**
  1. **14-dim adaptive merge genome** — per-block recombination ratios (attention vs MLP)
  2. **MRI-Trust Fusion** — balance diagnostic layer-importance (MRI) with evolutionary search, controlled by trust parameter
  3. **Architecture Mapper** — enables cross-architecture breeding (e.g., Transformer + Mamba components)
- **Results:** Darwin-27B-Opus = 86.9% GPQA Diamond (#6 among 1252 models)
- **Practical use for Omni project:** MRI-Trust can optimize merge weights between Omni-3B + Lance-3B transformer layers. Architecture Mapper handles cross-family merges (Qwen2.5 ↔ Qwen3 are compatible enough for mapper to learn the translation). GEPA can optimize the 14-dim genome via Pareto search.

## GEPA (Genetic-Pareto, GitHub gepa-ai/gepa)

- **Framework:** LLM-based reflection + Pareto-efficient evolutionary search for optimizing textual system components
- **Use case for Omni:** Optimize the Darwin merge genome (14 dims), modality routing thresholds, ACE-Step prompt templates, and agent architecture parameters
- **Integration:** GEPA evaluates candidate genomes → Darwin merge → benchmark on Hermes eval set → Pareto front → select best → feed back as next generation seeds
- **Analogy:** Darwin handles weight-space evolution; GEPA handles parameter-space (text/config) evolution. Together = full self-improving loop.

## Architectural Incompatibilities Summary

| Feature | Omni-3B | Lance-3B | ACE-Step 1.5 |
|---------|---------|----------|---------------|
| LLM backbone | Qwen2.5-3B | Qwen2.5-VL-4B-derived | UMT5-Base (T5) |
| Vision encoder | Omni native | Qwen2.5-VL SigLIP2 | N/A |
| Audio encoder | Omni native | N/A | SSL (MERT + M-HuBERT) |
| Music decoder | N/A | N/A | DiT diffusion + DCAE |
| Generation head | None (text-only) | Lance DiT | ACE-Step DiT |
| Merge-compatible with Omni | ✅ (same family) | ⚠️ backbone only | ❌ (T5 ≠ Qwen) |

**Practical conclusion:** Only the transformer backbone layers can be merged between Omni-3B and Lance-3B (same Qwen2.5 family). ACE-Step must stay as a completely separate routed specialist. Omni's native encoder handles all perception. Lance's DiT heads handle image/video generation. ACE-Step handles music generation. Omni routes between them.
