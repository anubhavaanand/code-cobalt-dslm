"""The model.

A decoder-only transformer using the modern small-model stack. These are all
settled choices, not innovations — copy them:

  RMSNorm (pre-norm)   cheaper and more stable than LayerNorm
  RoPE                 rotary position embeddings; no learned position table
  SwiGLU               better than GELU at equal parameter count
  no biases            everywhere; saves params and is strictly better
  weight tying          input embedding == output projection
  SDPA                  fused attention kernel (flash on supported hardware)

Deliberately *not* included, because they are optimisations rather than
architecture, and they complicate a codebase meant to be read:
  GQA, QK-norm, z-loss, sliding window, ALiBi
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dslm.config import ModelConfig


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, then cast back.
        dtype = x.dtype
        x32 = x.float()
        norm = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * norm).to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos, sin):
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        # is_causal=True gives us the causal mask for free and lets PyTorch
        # pick the flash-attention kernel when the hardware supports it.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.mlp_ratio)
        # Round to a multiple of 64 so tensor cores stay busy.
        hidden = max(64, (hidden // 64) * 64)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=RMSNorm(cfg.n_embd),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the embedding and the output projection share a matrix.
        # Saves ~30% of parameters at this scale and usually helps.
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # -- plumbing ---------------------------------------------------------- #
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.lm_head.weight.numel()
        return n

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ):
        """2D parameters get weight decay; 1D (norms, biases) do not.

        This is the standard nanoGPT-style split and it matters — decaying
        norm weights hurts.
        """
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    # -- forward ----------------------------------------------------------- #
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # -- sampling ---------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx