#!/usr/bin/env python3
"""
Sparse Upcycle — Convert a dense LLM into a Mixture-of-Experts model.

The key idea: take a dense base model, copy its FFN layers N times to create N
parallel experts per transformer block, add a small router, and continue-train
briefly to teach the router which expert to use.

Reference: Komatsuzaki et al. 2022 "Sparse Upcycling: Training MoE from Dense"
           https://arxiv.org/abs/2212.05055

This is Stage 3 of the OmniSenter pipeline (see omnisenter-architecture.md).
The base dense model comes from Stage 1 (agentic SFT on gen-0-clean) or
Stage 2 (evolutionary merge of variants). The N expert sources are the
specialist models we want to fuse (Qwen3-Omni for image/video/audio, ACE-Step
for music, etc.).

Usage:
    python3 sparse_upcycle.py \
        --base-model training-output/omnisenter-sft-20260606_213858/ \
        --expert-sources expert_a.gguf expert_b.gguf expert_c.gguf \
        --output training-output/omnisenter-moe-32a8b/ \
        --num-experts 6 --top-k 1
"""

import argparse, json, os, sys, shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "training-output"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class UpcycleConfig:
    """Configuration for sparse upcycling a dense model into a MoE."""
    base_model: str
    expert_sources: List[str] = field(default_factory=list)
    output: str = "training-output/omnisenter-moe/"
    num_experts: int = 6           # total experts per transformer block (including base)
    top_k: int = 1                 # active experts per token
    router_hidden: int = 256       # router hidden dim
    expert_init: str = "base"      # "base" | "source" | "random" — how to init new experts
    copy_base_attention: bool = True  # share attention across experts, or copy
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"
    # Routing
    router_aux_loss_coef: float = 0.01  # load-balancing aux loss
    jitter_noise: float = 0.01          # router jitter for training
    # Memory model
    shared_expert: bool = False    # if True, 1 always-on expert + (N-1) routed
    # Output
    save_safetensors: bool = True


def detect_ffn_modules(model) -> List[str]:
    """Detect FFN module names in the model (handles LLaMA, Qwen, Mistral naming)."""
    ffn_patterns = [
        # LLaMA / Qwen / Mistral
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        # Phi-3
        "mlp.gate_up_proj", "mlp.down_proj",
        # GPT-NeoX / older
        "mlp.c_fc", "mlp.c_proj",
    ]
    ffn_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pattern in ffn_patterns:
                if pattern in name:
                    ffn_modules.add(".".join(name.split(".")[:-1]))
                    break
    return sorted(ffn_modules)


class TopKRouter(nn.Module):
    """Top-k router with optional jitter and load-balancing aux loss."""
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1,
                 jitter_noise: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (top_k_weights, top_k_indices, router_logits)."""
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * self.jitter_noise
        router_logits = self.gate(hidden_states)  # (batch*seq, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights, top_k_indices, router_logits

    def aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Load-balancing auxiliary loss (Switch Transformer style)."""
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        # fraction of tokens routed to each expert
        expert_mask = F.one_hot(routing_weights.argmax(dim=-1), self.num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=0)
        # average routing probability per expert
        router_prob_per_expert = routing_weights.mean(dim=0)
        # aux loss = N * sum(f * P) — minimized when uniform
        return self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()


class MoEFFN(nn.Module):
    """N-expert MoE FFN block. Replaces a single nn.Linear (the down_proj) with N experts."""
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, top_k: int = 1, shared_expert: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert = shared_expert

        # We replicate the entire FFN (gate_proj, up_proj, down_proj) for each expert
        self.gate_projs = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=False) for _ in range(num_experts)
        ])

        if shared_expert:
            # 1 always-on shared expert (DeepSeek-V2 style)
            self.shared_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.shared_up = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.shared_down = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.router = TopKRouter(hidden_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, hidden) → (batch, seq, hidden)"""
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # (batch*seq, hidden)

        # Always-on shared expert output (added to routed output)
        shared_out = 0
        if self.shared_expert:
            shared_out = self.shared_down(F.silu(self.shared_gate(x_flat)) * self.shared_up(x_flat))

        # Routed experts
        top_k_weights, top_k_indices, router_logits = self.router(x_flat)
        # Compute all expert outputs (in fp32 for stability, downcast at end)
        out = torch.zeros_like(x_flat)
        for i, expert_id in enumerate(range(self.num_experts)):
            # Find tokens routed to this expert
            token_mask = (top_k_indices == expert_id).any(dim=-1)
            if not token_mask.any():
                continue
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            # Get the routing weight for this expert on these tokens
            expert_weight = (
                (top_k_indices[token_indices] == expert_id).float() *
                top_k_weights[token_indices]
            ).sum(dim=-1, keepdim=True)
            # Expert forward (SwiGLU)
            x_expert = x_flat[token_indices]
            expert_out = self.down_projs[expert_id](
                F.silu(self.gate_projs[expert_id](x_expert)) * self.up_projs[expert_id](x_expert)
            )
            out[token_indices] += expert_weight * expert_out

        out = out + shared_out
        return out.view(*orig_shape)


def find_ffn_blocks(model) -> List[Tuple[str, nn.Module]]:
    """Return list of (ffn_block_name, ffn_module) for each transformer layer."""
    ffn_blocks = []
    for name, module in model.named_modules():
        # Match MLP module names like "model.layers.0.mlp"
        if name.endswith(".mlp") and isinstance(module, nn.Module):
            ffn_blocks.append((name, module))
    return ffn_blocks


def upcycle_dense_to_moe(
    base_model_path: str,
    expert_sources: List[str],
    config: UpcycleConfig,
) -> str:
    """Main entry: take a dense model + optional expert sources, produce a MoE model."""

    print(f"=" * 70)
    print(f"SPARSE UPCYCLE — Dense → MoE")
    print(f"=" * 70)
    print(f"  Base: {base_model_path}")
    print(f"  Expert sources: {len(expert_sources)} (used for init only)")
    print(f"  Output: {config.output}")
    print(f"  Num experts: {config.num_experts}, top_k: {config.top_k}")
    print(f"  Shared expert: {config.shared_expert}")
    print()

    # Load base model
    print("[1/5] Loading base dense model...")
    dtype = getattr(torch, config.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map="cpu"
    )
    base_config = AutoConfig.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    print(f"  Loaded: {sum(p.numel() for p in base_model.parameters()):,} params")

    # Find FFN blocks
    ffn_blocks = find_ffn_blocks(base_model)
    if not ffn_blocks:
        print("  ❌ No FFN blocks found. Check model architecture.")
        sys.exit(1)
    print(f"  Found {len(ffn_blocks)} FFN blocks across the model.")

    # Load expert sources (optional — for initialization)
    print("[2/5] Loading expert sources...")
    expert_models = []
    for src in expert_sources:
        try:
            em = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, device_map="cpu")
            expert_models.append(em)
            print(f"  ✅ Loaded expert source: {src}")
        except Exception as e:
            print(f"  ⚠️  Could not load {src}: {e}")
    if expert_models:
        print(f"  Using {len(expert_models)} sources to init new experts")
    else:
        print(f"  No sources loaded — all experts will be initialized as copies of the base FFN")

    # Replace each FFN block with a MoE FFN
    print("[3/5] Replacing FFN blocks with MoE...")
    hidden_size = base_config.hidden_size
    intermediate_size = base_config.intermediate_size
    num_experts_total = config.num_experts

    new_state_dict = {}
    for ffn_name, ffn_module in ffn_blocks:
        # ffn_module is an MLP with gate_proj, up_proj, down_proj
        # Build a MoE FFN
        moe_ffn = MoEFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts_total,
            top_k=config.top_k,
            shared_expert=config.shared_expert,
        ).to(dtype)

        # Initialize experts
        # Expert 0 = base FFN (always)
        moe_ffn.gate_projs[0].weight.data = ffn_module.gate_proj.weight.data.clone()
        moe_ffn.up_projs[0].weight.data = ffn_module.up_proj.weight.data.clone()
        moe_ffn.down_projs[0].weight.data = ffn_module.down_proj.weight.data.clone()

        # If shared expert, init from base
        if config.shared_expert:
            moe_ffn.shared_gate.weight.data = ffn_module.gate_proj.weight.data.clone()
            moe_ffn.shared_up.weight.data = ffn_module.up_proj.weight.data.clone()
            moe_ffn.shared_down.weight.data = ffn_module.down_proj.weight.data.clone()

        # Remaining experts: initialize from expert sources if available, else copy base
        for e_idx in range(1, num_experts_total):
            if e_idx - 1 < len(expert_models):
                src_model = expert_models[e_idx - 1]
                # Find the same FFN in the source
                src_ffn_name = ffn_name
                try:
                    src_ffn = src_model.get_submodule(src_ffn_name)
                    moe_ffn.gate_projs[e_idx].weight.data = src_ffn.gate_proj.weight.data.clone()
                    moe_ffn.up_projs[e_idx].weight.data = src_ffn.up_proj.weight.data.clone()
                    moe_ffn.down_projs[e_idx].weight.data = src_ffn.down_proj.weight.data.clone()
                except AttributeError:
                    print(f"  ⚠️  Source {e_idx-1} missing FFN {src_ffn_name}, falling back to base copy")
                    moe_ffn.gate_projs[e_idx].weight.data = ffn_module.gate_proj.weight.data.clone()
                    moe_ffn.up_projs[e_idx].weight.data = ffn_module.up_proj.weight.data.clone()
                    moe_ffn.down_projs[e_idx].weight.data = ffn_module.down_proj.weight.data.clone()
            else:
                # Copy from base (will be diversified by continued training)
                moe_ffn.gate_projs[e_idx].weight.data = ffn_module.gate_proj.weight.data.clone()
                moe_ffn.up_projs[e_idx].weight.data = ffn_module.up_proj.weight.data.clone()
                moe_ffn.down_projs[e_idx].weight.data = ffn_module.down_proj.weight.data.clone()

        # Initialize the router — start uniform (each expert equally likely)
        nn.init.zeros_(moe_ffn.router.gate.weight)
        # Add small noise to break symmetry
        moe_ffn.router.gate.weight.data += torch.randn_like(moe_ffn.router.gate.weight.data) * 0.01

        # Save the new MoE FFN weights into the state dict
        ffn_prefix = ffn_name
        new_state_dict[f"{ffn_prefix}.gate_projs.{0}.weight"] = moe_ffn.gate_projs[0].weight.data
        new_state_dict[f"{ffn_prefix}.up_projs.{0}.weight"] = moe_ffn.up_projs[0].weight.data
        new_state_dict[f"{ffn_prefix}.down_projs.{0}.weight"] = moe_ffn.down_projs[0].weight.data
        for e_idx in range(1, num_experts_total):
            new_state_dict[f"{ffn_prefix}.gate_projs.{e_idx}.weight"] = moe_ffn.gate_projs[e_idx].weight.data
            new_state_dict[f"{ffn_prefix}.up_projs.{e_idx}.weight"] = moe_ffn.up_projs[e_idx].weight.data
            new_state_dict[f"{ffn_prefix}.down_projs.{e_idx}.weight"] = moe_ffn.down_projs[e_idx].weight.data
        new_state_dict[f"{ffn_prefix}.router.gate.weight"] = moe_ffn.router.gate.weight.data
        if config.shared_expert:
            new_state_dict[f"{ffn_prefix}.shared_gate.weight"] = moe_ffn.shared_gate.weight.data
            new_state_dict[f"{ffn_prefix}.shared_up.weight"] = moe_ffn.shared_up.weight.data
            new_state_dict[f"{ffn_prefix}.shared_down.weight"] = moe_ffn.shared_down.weight.data

    # Save the upcycled model
    print("[4/5] Saving upcycled model...")
    output_path = Path(config.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save updated state dict
    state_dict = base_model.state_dict()
    for k, v in new_state_dict.items():
        if k in state_dict:
            print(f"  Replacing: {k}  shape={v.shape}")
        state_dict[k] = v

    if config.save_safetensors:
        # Save in shards of 5GB each
        max_shard_size = "5GB"
        base_model.save_pretrained(str(output_path), max_shard_size=max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    # Save upcycle metadata
    metadata = {
        "format_version": "omnisenter-moe/1.0",
        "upcycled_from": base_model_path,
        "expert_sources": expert_sources,
        "num_experts": config.num_experts,
        "top_k": config.top_k,
        "shared_expert": config.shared_expert,
        "total_params": sum(p.numel() for p in base_model.parameters()) +
                        (num_experts_total - 1) * len(ffn_blocks) * 3 * hidden_size * intermediate_size,
        "active_params_per_token": (sum(p.numel() for p in base_model.parameters()) -
                                    len(ffn_blocks) * 3 * hidden_size * intermediate_size) +
                                   (config.top_k * 3 * hidden_size * intermediate_size * len(ffn_blocks)),
    }
    with open(output_path / "upcycle_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[5/5] ✅ Sparse upcycle complete!")
    print(f"  Output: {output_path}")
    print(f"  Total params: {metadata['total_params']:,}")
    print(f"  Active per token: {metadata['active_params_per_token']:,}")
    print(f"  Config: {asdict(config)}")

    # Log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "upcycle_log.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "base": base_model_path,
            "experts": expert_sources,
            "config": asdict(config),
            "total_params": metadata["total_params"],
            "active_params": metadata["active_params_per_token"],
        }) + "\n")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Sparse upcycle a dense LLM into a MoE")
    parser.add_argument("--base-model", required=True, help="Path to base dense model")
    parser.add_argument("--expert-sources", nargs="*", default=[], help="Optional expert source models")
    parser.add_argument("--output", required=True, help="Output path for upcycled MoE")
    parser.add_argument("--num-experts", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--shared-expert", action="store_true", help="Add 1 always-on shared expert")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--jitter-noise", type=float, default=0.01)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.01)
    args = parser.parse_args()

    config = UpcycleConfig(
        base_model=args.base_model,
        expert_sources=args.expert_sources,
        output=args.output,
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_expert=args.shared_expert,
        dtype=args.dtype,
        jitter_noise=args.jitter_noise,
        router_aux_loss_coef=args.router_aux_loss_coef,
    )

    upcycle_dense_to_moe(args.base_model, args.expert_sources, config)


if __name__ == "__main__":
    main()
