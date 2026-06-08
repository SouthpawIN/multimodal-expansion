# Multimodal Expansion

> **Add audio, speech, or vision to an LLM that doesn't have them.** Three concrete approaches, training pipelines, GPU feasibility, and the architectural pitfalls that derail cross-family merges.

This is the practical playbook for **adding a new modality to an existing language model** — whether you're bolting Whisper onto Qwen, replacing MoE experts via REAP, or letting an existing MoE organically specialize.

The hard truth up front: **most cross-family merges don't work.** Qwen3 with DeltaNet can't weight-merge with Qwen2.5 transformers. ACE-Step's UMT5 + DiT can't be folded into a Qwen LLM. Knowing *what you can't merge* is half the value of this skill.

---

## Quick decision tree

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

---

## The three approaches

| Approach | Effort | Compute | Native? | When to use |
|---|---|---|---|---|
| **A. LLaVA-style adapter** | Days | Dual 3090 OK with LoRA | Bolt-on | First-time modality add, limited GPU |
| **B. REAP + expert replacement** | Weeks | Cloud (A100/H100) | Native (MoE) | Have an MoE LLM, want zero-shot modality integration |
| **C. EvoMoE-style specialization** | Weeks | Dual 3090+ | Native (organic) | Have an MoE LLM, want it to learn modalities naturally |

### 🅰️ Approach A: Adapter-Based (LLaVA Pattern) — MOST FEASIBLE

Add modality-specific encoders + projectors. The LLM stays mostly frozen.

**Architecture:**
```
[Audio] → [Audio Encoder] → [Projector/Adapter] → [LLM (frozen or LoRA)] → text
                                                                      → [Talker] → speech (optional)
```

**Training pipeline (3 stages):**

1. **Encoder Alignment** — Freeze LLM. Train encoder + projector on paired data (e.g. ASR: speech→text). ~10K hours.
2. **Multimodal Fine-Tune** — LoRA on LLM. Diverse audio tasks: ASR, captioning, QA, music, sound events. ~1M examples.
3. **Speech Output** (optional) — Train Talker on Thinker hidden states, OR use external TTS (ElevenLabs, Edge TTS).

**Audio encoder options:**

| Encoder | Strengths | License | Notes |
|---|---|---|---|
| Whisper-large-v3 | Easy, well-supported, 99 langs | MIT | Default choice |
| Qwen-Audio encoder | Better Chinese/multilingual | Apache 2.0 | Pair with Qwen LLMs |
| AuT (Qwen3-Omni) | Best in class | Apache 2.0 (model) | 20M hr training data NOT public |
| Wav2Vec2-BERT | SSL pretraining | Apache 2.0 | Good for fine-tuning |

**Tools:**
- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM) — open-source framework for speech/audio/music → LLM training. Has pretrained checkpoints and training recipes.
- [Audio Flamingo](https://github.com/NVIDIA/audio-flamingo) — alternative with good zero-shot
- [SALMONN](https://github.com/bytedance/SALMONN) — speech/audio LLM

**GPU budget (dual 3090, 48GB total):**
- Stage 1 (encoder align): fits on a single 3090
- Stage 2 (LoRA r=64 multimodal FT): fits on dual 3090 for ≤3B active models
- Full fine-tune: needs cloud (A100/H100)

### 🅱️ Approach B: REAP + Expert Replacement — MOST AMBITIOUS

Prune least-used MoE experts via REAP, then replace them with modality-specialized experts.

**REAP saliency score:**
```
S_j = (1/|X_j|) * Σ g_j(x) · ‖f_j(x)‖₂
```
- `g_j(x)`: router gate-value for expert j on input x
- `f_j(x)`: output of expert j
- `X_j`: set of inputs where expert j is active (TopK)

Prune experts with lowest S_j scores. Then replace with modality-specialized ones.

**Pipeline (4 stages):**

1. **REAP Calibration** — Run data through model, score 256 experts, prune bottom 30-50%. Needs full FP16 model (~70GB) — cloud GPU.
2. **Expert Initialization** — Init new experts. Options: clone existing + noise, copy from Omni Thinker (probably incompatible), or random init.
3. **Router Retraining** — Freeze experts, train router to route audio tokens → new experts.
4. **Full Multimodal FT** — Unfreeze everything. Needs A100/H100 class.

**Performance (50% pruning baseline from REAP paper):**

| Model | Metric | REAP | Merging (HC-SMoE) |
|---|---|---|---|
| Qwen3-30B-A3B | Code Gen | **95.9%** | 65.2% |
| GLM-4.5-Air | LiveCodeBench | **94.1%** | 58.8% |
| Qwen3-480B-Coder | SWE-Bench | **96.7%** | N/A |

REAP > merging for generative tasks because the router keeps dynamic control of which expert fires. Merging replaces dynamic choice with a static average → irreducible error.

**Key risk:** Architecture compatibility. MoE experts from different model families (standard transformer vs DeltaNet) are almost certainly dimension-mismatched.

**No existing off-the-shelf tools for the "replace with modality expert" step.** Would require building calibration, expert init, router training from scratch on top of the [Cerebras REAP repo](https://github.com/CerebrasResearch/reap).

### 🅲 Approach C: EvoMoE-Inspired Specialization — MIDDLE GROUND

Based on [EvoMoE](https://arxiv.org/abs/2505.23830) (AAAI 2026). Let existing experts naturally specialize through staged training — no pruning or swapping.

**Pipeline (3 stages):**

1. **Warm-up** — Multimodal instruction data to familiarize model with audio tokens.
2. **Expert Diversification** — Contrastive learning pushes experts to specialize per modality.
3. **Router Refinement** — Fine-tune router for optimal expert selection per input type.

**Key insight:** The router learns to route audio tokens to specific experts organically. No architectural changes needed (beyond adding the audio encoder + projector, same as Approach A).

**Status:** Paper only, no released code. Feasibility uncertain — but if it works, it's the cleanest path because you keep the full MoE intact.

---

## Architecture compatibility (the "what can I actually merge?" matrix)

| Component A | Component B | Merge? | Notes |
|---|---|---|---|
| Qwen2.5 LLM | Qwen2.5-VL text body | ✅ Yes | Same backbone family, hidden=2048, 36L |
| Qwen2.5-Omni-3B thinker | Lance 3B (Qwen2.5-VL derived) | ✅ Yes | After prefix normalization, 432 common tensors |
| Qwen2.5 LLM | Qwen3 LLM (different family) | ⚠️ Partial | Architecture Mapper can bridge, but limited |
| Qwen2.5 LLM | ACE-Step DiT (diffusion) | ❌ No | Different modality entirely |
| Qwen2.5 LLM | ACE-Step LM (Qwen3-4B) | ⚠️ Skips mostly | Different hidden (2048 vs 2560), vocab mismatch |
| Qwen3.6 (DeltaNet) | Qwen2.5 (transformer) | ❌ No | Linear attention vs full attention are incompatible |
| ACE-Step LM | UMT5 (T5 family) | ❌ No | Different family, UMT5 is a condition encoder |
| Any LLM | Any DiT/VAE/vocoder | ❌ No | These are purpose-built specialists, route them |

**Practical rule:** Only models sharing the exact same backbone family (Qwen2.5 with Qwen2.5) can be weight-merged. Specialists (DiT, VAE, talker) stay routed, not merged.

---

## Model reference (Qwen family)

### Qwen2.5-Omni-3B (Jan 2026, the "one base model" anchor)
- **Type:** End-to-end multimodal (text + images + audio + video → text + speech)
- **Architecture:** Standard Qwen2.5 transformer + Omni encoder
- **Total:** 3B dense | **Hidden:** 2048 | **Layers:** 27
- **Audio:** Native streaming ASR + TTS, real-time interruption
- **Status:** Local GGUF (Q4_K_M), the default OmniSenter anchor

### Qwen2.5-Omni-7B (Mar 2025)
- Thinker-Talker architecture, sliding-window DiT for streaming audio
- TMRoPE for video-audio sync
- Open weights, Apache 2.0

### Qwen3-Omni-30B-A3B (Oct 2025)
- Thinker-Talker MoE, AuT audio encoder (20M hours, 12.5Hz)
- SOTA on 22/36 audio benchmarks
- Uses standard Qwen3 MoE (not DeltaNet) — best for our purposes

### Qwen3.6 35B A3B (May 2026)
- **Type:** Vision-language (text + images → text)
- 10 × (3 × Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE)
- 256 experts, 8 routed + 1 shared
- **Already has:** vision ✅ **Missing:** audio input, speech output
- **Caveat:** Gated DeltaNet is linear attention, NOT standard transformer — making architecture compatibility with Qwen2.5/Qwen3 tricky.

### Qwen3.5-Omni (Apr 2026, NOT open weights)
- Hybrid Attention MoE for both Thinker and Talker
- Hundreds of billions of parameters
- SOTA on 215 audio/visual benchmarks
- Use as research reference, can't deploy locally

### ACE-Step 1.5 (Feb 2026, music specialist)
- **LM component:** Qwen3-based planners (`acestep-5Hz-lm-0.6B/1.7B/4B`)
- **UMT5 encoder:** T5-family, secondary condition encoder only
- **DiT:** 2B/4B diffusion transformer for audio
- **VAE:** Music-specific 1D DC-AE + FSQ tokenizer, 5Hz
- **Key insight:** The LM planners ARE Qwen3-based → theoretically routable from a Qwen2.5 base via projection adapters. The UMT5 and DiT components are specialists that stay routed.

### Lance 3B (May 2026, ByteDance)
- **Type:** Unified multimodal (text + images + video → understanding + generation + editing)
- **Architecture:** Qwen2.5-VL-derived, 36 layers, hidden=2048
- **Unique:** pre-trained MoE per layer (`_moe_gen` twin weights) — every layer has a generation expert
- **Insight:** Darwin merge of the TRANSFORMER BACKBONE layers is possible. Generation heads (DiT) stay routed.

---

## Senter Omni architecture (as-built)

| Component | Path | Type | Key specs |
|---|---|---|---|
| Omni thinker | `Models/hf/Qwen2.5-Omni-3B/` | Qwen2.5-3B + NaViT + Whisper | 36L, hidden=2048, heads=16, KV=2, vocab=151936 |
| Lance 3B | `Models/senter-omni/lance/weights/Lance_3B/` | Qwen2.5-VL-4B + DiT head | 36L, hidden=2048, heads=16, KV=2, vocab=151936 |
| ACE-Step LM | `Models/senter-omni/ace-step-lora/weights/ace-step-lm-qwen3-4b/` | Qwen3-4B | 36L, hidden=2560, heads=32, KV=8, vocab=217204 |
| ACE-Step DiT+UMT5 | `Models/senter-omni/ace-step-lora/weights/` | Diffusion + UMT5 encoder | Music generation specialist only |

**The "one encoder" rule:** ONE LLM backbone + ONE encoder stack. The backbone handles all reasoning; the encoder handles all perception. Generation heads (music DiT, video DiT) are specialists that stay routed.

### Darwin → mergekit two-stage pipeline

```
Stage 1 (Darwin):  Merge P1(Omni) + P2(Lance) text bodies → unified child checkpoint
                   Same-arch merge: hidden=2048, heads=16, 36L MATCH PERFECTLY
Stage 2 (mergekit): Build MoE + Dense variants from the unified child
                   MoE: 9 experts (per-layer), ~3B active
                   Dense: smaller for resource-constrained inference
```

Run Darwin FIRST, get stable generations, THEN do mergekit variants.

### Why ACE-Step can't be weight-merged into Qwen2.5

Different hidden sizes (2560 vs 2048), vocab sizes (217204 vs 151936), and attention head configs (32/8 vs 16/2). Darwin Architecture Mapper could theoretically bridge this but requires research-grade implementation. Keep ACE-Step as a routed expert with projection adapters at the dispatch boundary.

---

## Recommended phased approach

1. **Phase 1:** Approach A with Whisper encoder + LoRA on Qwen3.6 35B A3B. Use SLAM-LLM. External TTS for speech output. (Realistic on dual 3090.)
2. **Phase 2:** Try Approach C (EvoMoE) once audio tokens flow through the model.
3. **Phase 3:** Approach B only with cloud compute budget — this is publishable research.

---

## Pitfalls (must-read)

1. **Always verify architecture compatibility before merging.** The ACE-Step 1.5 LM IS Qwen3-based (confirmed from model card), not UMT5-based. The UMT5 is just a lyric conditioning encoder. Don't assume LM = whole model — check `config.json` and the model card.
2. **Verify facts before claiming "impossible."** When asked about Qwen3.6 4B or Qwen3-4B, these models DO exist. Look them up, don't assert they don't.
3. **MoE experts are FFN replacements, not modality handlers.** Audio understanding happens at the input encoding level — the encoder + projector, not inside the FFN experts. "Swapping an expert for an Omni" doesn't make architectural sense unless you replace the entire Thinker.
4. **Omni thinker has submodules.** Full Omni = thinker + talker + token2wav + vision/audio configs. Only `thinker.model.layers.0-35` overlaps with a Qwen2.5-VL text body. Including Omni-only layers (token2wav, talker) in a merge will crash. Filter to thinker.text body only.
5. **Architecture compatibility between model families is low.** Qwen3.6 uses Gated DeltaNet + MoE; Qwen3-VL uses standard transformers; ACE-Step uses UMT5-T5; Lance uses Qwen2.5-VL backbone. None of these can be weight-merged with each other.
6. **"One base model" means ONE LLM backbone + ONE encoder stack.** Keep it Omni's native encoder. Do NOT swap in Lance's ViT.
7. **Specialist models with unique heads (DiT, VAE) cannot be weight-merged into an LLM.** Route to them via a dispatcher layer.
8. **Darwin → mergekit two-stage pipeline.** Darwin first, then mergekit variants.
9. **Lance is pre-trained MoE.** Every layer has `_moe_gen` twin weights — a second expert per layer trained for generation. The merge job is to formalize it with a learned router.
10. **Speech output (Talker) is the hardest part.** External TTS is the pragmatic choice for Phase 1. Building a Talker requires the full Omni training pipeline.
11. **Action over planning.** When the user says "go" or "stop talking, start building," stop explaining and execute.
12. **Extend existing skills rather than inventing new ones.** Always check installed skills before building parallel implementations.
13. **GPU resource awareness.** Dual 3090s with 24GB each. Monitor VRAM. Prefer GGUF quantization for local inference.

---

## Skill contents

```
multimodal-expansion/
├── SKILL.md                                # the master procedure
├── README.md                               # this file
├── LICENSE                                 # MIT
└── references/
    ├── qwen-architecture-comparison.md     # Qwen family + ACE-Step + Lance specs
    ├── reap-technique.md                   # REAP saliency, performance, GPU needs
    └── senter-omni-project-state.md        # live directory layout + next-step checklist
```

---

## Related skills

- [`evolutionary-model-merging`](../evolutionary-model-merging) — Darwin weight-space merging (used to combine Omni + Lance text bodies).
- [`evolutionary-radio`](../evolutionary-radio) — the music-modality plugin of OmniSenter.
- `gepa-prompt-evolution` (wiki entity) — orthogonal parameter-space evolution.
- `herm-tui-radio` — the TUI music bar that surfaces radio state.

---

## References

- **REAP** (Cerebras Research, Oct 2025): [arxiv.org/abs/2510.13999](https://arxiv.org/abs/2510.13999) — [github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- **EvoMoE** (AAAI 2026): [arxiv.org/abs/2505.23830](https://arxiv.org/abs/2505.23830)
- **SLAM-LLM** (X-LANCE): [github.com/X-LANCE/SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)
- **Qwen2.5-Omni**: [Qwen2.5-Omni-3B on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- **Qwen3-Omni**: [Qwen3-Omni-30B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B)
- **ACE-Step**: [ACE-Step/ACE-Step-v1-3.5B on HuggingFace](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)
- **Lance**: [bytedance-research/Lance on HuggingFace](https://huggingface.co/bytedance-research/Lance)
- **Darwin Family paper**: [arxiv.org/abs/2605.14386](https://arxiv.org/abs/2605.14386)

---

## Author

Chris (SouthpawIN) — Senter Dev Discord, Nous Research
See also: [`evolutionary-model-merging`](../evolutionary-model-merging), [`evolutionary-radio`](../evolutionary-radio)


---

## Part of the Omni Family

This repo is one of the 6 GitHub repos in the [OmniSenter / Omni Family](https://github.com/SouthpawIN/evolutionary-training/blob/master/blog/the-omni-family.md) project.

**Naming (read first):** [the-omni-family.md](https://github.com/SouthpawIN/evolutionary-training/blob/master/blog/the-omni-family.md) — defines **Omni** (multimodal), **Senter** (agentic core), **Ohm** (self-evolving engine), **Senter Ohm** (the flagship ~32A8B MoE). Every model in the family has a name that composes these suffixes.

**Full blog catalog:** [CATALOG.md](https://github.com/SouthpawIN/evolutionary-training/blob/master/blog/CATALOG.md) — 13 posts covering the architecture, the math, the pipeline, the concepts (Synthesia, Ohm), the integration with Hermes, the notebook schema, and the research direction.

**HuggingFace (transitional v1):** [`sovthpaw/omnistep-12a3b`](https://huggingface.co/sovthpaw/omnistep-12a3b) (12B total / 3B active, multimodal), [`sovthpaw/Omni-Senter-3B`](https://huggingface.co/sovthpaw/Omni-Senter-3B) (3B), [`sovthpaw/OmniSenter-Base-16B`](https://huggingface.co/sovthpaw/OmniSenter-Base-16B) (16B base). These are the v1 lineage — the new architecture (Senter Ohm 32A8B, OmniSenter 12B, OmniSenterStep) will replace them as it ships.

**Sibling repos:**
- [`SouthpawIN/evolutionary-training`](https://github.com/SouthpawIN/evolutionary-training) — main repo, this blog, training scripts, Ohm runtime
- [`SouthpawIN/evolutionary-model-merging`](https://github.com/SouthpawIN/evolutionary-model-merging) — Darwin Family (CMA-ES + paper-exact merge)
- [`SouthpawIN/multimodal-expansion`](https://github.com/SouthpawIN/multimodal-expansion) — REAP + EvoMoE + `sparse_upcycle.py`
- [`SouthpawIN/omnistep-fusion`](https://github.com/SouthpawIN/omnistep-fusion) — Cosmos × ACE-Step multimodal merge
- [`SouthpawIN/evolutionary-radio`](https://github.com/SouthpawIN/evolutionary-radio) — OmniStep-brained music radio
- [`SouthpawIN/hermes-agent`](https://github.com/SouthpawIN/hermes-agent) — the smart agent Senter is auxiliary to
