# REAP: Router-weighted Expert Activation Pruning

Source: Cerebras Research, Oct 2025. Paper: [arxiv.org/abs/2510.13999](https://arxiv.org/abs/2510.13999)
Code: [github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
Models: [huggingface.co/cerebras](https://huggingface.co/collections/cerebras/cerebras-reap)

## Core Idea

One-shot pruning for SMoE (Sparse Mixture of Experts) models. Removes low-impact experts while preserving the router's dynamic control over surviving experts. **Pruning > Merging** for generative tasks.

## Saliency Score

```
S_j = (1/|X_j|) * Σ g_j(x) · ‖f_j(x)‖₂
```

- f_j(x): Output of expert j
- g_j(x): Router's gate-value for expert j on input x
- X_j: Set of inputs where expert j is active (TopK)

Higher score = more important. Prune lowest-scoring experts.

## Why Merging Fails

Merging experts introduces irreducible error because it replaces the router's dynamic choice with a static average. Error is proportional to:
- Router policy variability (how much it varies mixing strategy)
- Expert gap (how different the two experts are)
- Router scale (magnitude of gate values)

Late layers suffer most — experts are highly specialized, merging causes ~100x reduction in functional diversity.

## Performance (50% pruning)

| Model | Metric | REAP | Merging (HC-SMoE) |
|---|---|---|---|
| Qwen3-30B-A3B | Code Gen | **95.9%** | 65.2% |
| GLM-4.5-Air | LiveCodeBench | **94.1%** | 58.8% |
| Qwen3-480B-Coder | SWE-Bench | **96.7%** | N/A |

## Pipeline

1. **Calibration:** Run calibration dataset through model, collect expert activations and outputs
2. **Scoring:** Compute saliency score S_j for each expert
3. **Pruning:** Remove bottom N% of experts (typically 30-50%)
4. **Router Renorm:** Renormalize router logits for remaining experts
5. **Export:** Save pruned model with adjusted router

## GPU Requirements

For a 35B model in FP16: ~70GB VRAM minimum for full model in memory. The Cerebras repo has layerwise calibration observer to cap peak memory by loading/unloading blocks.

## Key Insight for Multimodal Expansion

REAP frees up expert "slots" in MoE layers. The pruned experts could theoretically be replaced with modality-specialized ones. However, the new experts would need to be trained from scratch (weights from different architectures like Omni's Thinker are NOT compatible with Qwen3.6's DeltaNet-based MoE). The router would also need retraining to learn to route audio tokens to the new experts.

## Useful Commands (from Cerebras repo)

The repo at `github.com/CerebrasResearch/reap` provides:
- `scripts/pruning-cli.sh` — main entry point
- `config/` — YAML configs for different models
- `src/reap/` — core pruning logic
- Layerwise observer for memory-constrained environments
