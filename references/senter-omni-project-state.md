---
title: Senter Omni Project State (2026-05-31)
---

## Directory Layout
```
/home/sovthpaw/Models/senter-omni/
├── lance/
│   └── weights/
│       ├── Lance_3B/               # 12GB — LLM with _moe_gen dual weights
│       ├── Lance_3B_Video/         # 14GB — video generation head
│       └── Qwen2.5-VL-ViT/         # 1.3GB — SigLIP2 encoder (NOT used)
├── ace-step-lora/
│   └── weights/
│       ├── ACE-Step-v1-3.5B/       # UMT5 + DiT + VAE (downloaded, not for merge)
│       ├── ace-step-lm-qwen3-4b/   # 7.9GB — Qwen3-4B LM (correct weights!)
│       └── umt5-base/              # UMT5 lyric encoder (DON'T merge this)
├── omni-senter-9a3b/               # Main build directory
│   ├── configs/
│   │   ├── architecture.yaml       # MoE architecture spec
│   │   └── variants.yaml           # Dense + MoE variant plans
│   ├── scripts/
│   │   ├── arcturus_merge.py       # THE merge script (runs Omni+Lance Darwin merge)
│   │   ├── check_compat.py         # Tensor compatibility checker
│   │   └── darwin_moe_build.py     # Original scaffold (superseded by arcturus)
│   └── checkpoints/
│       └── gen_0/                  # ← Output goes here (in progress)
└── lances-omni-0/                  # Smaller variant project
    └── configs/
        └── architecture.yaml       # Simplified single-model plan
```

## Tensor Compatibility (Verified)

### Omni-3B thinker ↔ Lance VL text body: MATCH
- 432 common layer tensors (after prefix normalization)
- All shapes match: hidden=2048, heads=16, KV=2, intermediate=11008
- Omni prefix: `thinker.model.layers.N.*`
- Lance prefix: `language_model.model.layers.N.*`
- Merge is straightforward after prefix normalization

### Omni-3B thinker ↔ ACE-Step LM: INCOMPATIBLE
- Different hidden: 2048 (Omni) vs 2560 (ACE)
- Different vocab: 151936 vs 217204
- Different heads: 16/2 vs 32/8
- Solution: routed expert with projection adapters at dispatch boundary

## Darwin Genome Configuration
```python
GENOME = [0.6, 0.55, 0.5, 0.5, 0.45, 0.45, 0.4, 0.4, 0.45]
# 9 blocks, early layers Omni-heavy (0.6), balanced middle, Lance-leaning late
trust_parameter = 0.5  # MRI-Trust balance
```

## Merge Pipeline (Arcturus)

1. Load Omni thinker text body (434 tensors from 3 shards)
2. Load Lance text body (1011 tensors including `_moe_gen` from single file)
3. Normalize prefixes → 432 common tensors
4. Apply Darwin genome per layer:
   - MRI-Trust fusion: L2 norm as importance proxy
   - Weighted blend: omega = trust * mri_omni + (1-trust) * genome_weight
5. Preserve all `_moe_gen` weights from Lance (Expert B — generation specialist)
6. Add learned router: 1-layer MLP, 2048 → 3 experts
7. Save sharded safetensors + config

## Critical Discoveries

### Lance is already pre-trained MoE
Every layer has `_moe_gen` twin weights (e.g. `self_attn.q_proj` + `self_attn.q_proj_moe_gen`). These are a second expert per layer trained for generation tasks. We're not imposing MoE — we're unlocking what ByteDance already built and adding a learned router on top.

### ACE-Step 1.5 architecture
- LM: Qwen3-4B (Qwen3ForCausalLM, 36L, hidden=2560)
- Lyric encoder: UMT5-Base (T5-family, NOT Qwen)
- DiT: custom diffusion transformer (2B params)
- Only the LM is compatible with our merging strategy. DiT and UMT5 are specialists.

## Next Steps
1. ✅ Downloads complete (Lance + ACE-Step LM)
2. 🔄 Merge script running (arcturus_merge.py in background)
3. ⬜ Evaluate gen_0 on text + vision tasks
4. ⬜ GEPA optimize genome for better routing
5. ⬜ Wire ACE-Step LM as routed expert with projection adapters
6. ⬜ Convert to GGUF for llama.cpp deployment
7. ⬜ Build variant configs (Dense Small, Dense Large, MoE Small 9A3B)
