"""TinyLM — a minimal decoder-only transformer for function-calling.

Architecture overview
---------------------
```
Input tokens → Embedding (vocab × d_model, tied with output head)
             → N × TransformerBlock
                  ├─ RMSNorm → fused QKV projection → RoPE → causal attention → output projection
                  └─ RMSNorm → GELU feed-forward (up → gelu → down)
             → RMSNorm → logits (via tied embedding transpose)
```

Key design choices (educational notes)
---------------------------------------
* **RMSNorm** instead of LayerNorm: ~same quality but fewer ops (no mean
  subtraction), which matters at scale and simplifies the code.
* **RoPE** (Rotary Position Embeddings): injects position information by
  rotating pairs of dimensions in Q and K.  Unlike learned positional
  embeddings, RoPE naturally encodes *relative* distances, which helps
  generalisation.  We pre-compute the cos/sin tables once and slice per
  sequence length.
* **Fused QKV**: a single `Linear(d, 3d)` for Q, K, V saves one kernel launch
  on GPU vs three separate projections.  llama.cpp can't convert this layout
  (it expects separate Q/K/V), which is why `train_llama.py` exists.
* **Tied embeddings**: input embedding weights are reused as the output
  projection (`logits = norm(x) @ emb.weight.T`), halving the parameter
  count of the vocabulary layer.
* **GELU FFN**: a simple up→gelu→down feed-forward.  The Llama variant in
  `train_llama.py` uses gated SiLU (SwiGLU) instead, adding a third weight
  matrix — that is the standard Llama MLP.

Limitations (honest about what this is NOT)
-------------------------------------------
* ~0.47M params is orders of magnitude smaller than useful general-purpose LMs.
  This model can only solve its narrow trained task (web_search routing).
* No KV cache: every forward pass recomputes attention from scratch, so
  generation is O(T²) per token.  Fine for MAX_LEN=80, prohibitive for long
  sequences.
* No dropout — with 80k synthetic training samples and heavy data
  augmentation the model converges without it, but this means it will overfit
  fast on small *real* datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Normalises input by its RMS (no mean-centering, no bias), then applies a
    learned element-wise scale ``w``.  Cheaper than LayerNorm and empirically
    equivalent for transformers.

    Args:
        d:   feature dimension to normalise over (last axis).
        eps: small constant to prevent division by zero.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(mean(x²) + eps) is numerically stable and fuses well on GPU
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_cache(T: int, hd: int, base: float = 10000.0):
    """Pre-compute RoPE cosine and sine tables.

    Returns two tensors of shape ``(T, hd//2)`` — one for cosines, one for
    sines — indexed by position.  These are registered as non-persistent
    buffers in the model so they move with ``.to(device)`` but are not saved
    in the state dict.

    Args:
        T:    maximum sequence length.
        hd:   head dimension (must be even — RoPE pairs adjacent elements).
        base: wavelength base; 10000 is the standard value from Vaswani et al.
    """
    # Inverse frequency for each dimension pair: θ_i = 1 / base^(2i/d)
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    # Outer product gives (position × dimension_pair) frequency matrix
    f = torch.outer(torch.arange(T).float(), inv)  # shape: (T, hd//2)
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to queries or keys.

    Takes the even- and odd-indexed elements of the head dimension, rotates
    them by the position-dependent angle, and interleaves them back.

    Args:
        x:   tensor of shape ``(B, H, T, hd)`` — queries or keys.
        cos: cosine table ``(T, hd//2)`` from :func:`rope_cache`.
        sin: sine  table ``(T, hd//2)`` from :func:`rope_cache`.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]       # even / odd elements
    o1 = x1 * cos - x2 * sin                   # standard 2D rotation
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), -1).flatten(-2)  # interleave back


class Block(nn.Module):
    """A single transformer block: attention + feed-forward, both pre-normed.

    Pre-norm (norm → sublayer → residual) is more stable than post-norm for
    deep networks and is what modern LLMs (Llama, GPT-NeoX) use.

    Args:
        d:        model dimension.
        h:        number of attention heads.
        ffn_mult: feed-forward inner dimension multiplier (inner_dim = ffn_mult × d).
    """

    def __init__(self, d: int, h: int, ffn_mult: int):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        # Fused QKV saves a kernel launch vs three separate Linear layers
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        # Simple two-matrix FFN (up → GELU → down).  Llama uses a three-matrix
        # gated SiLU variant instead — see train_llama.py for that version.
        self.up = nn.Linear(d, ffn_mult * d, bias=False)
        self.down = nn.Linear(ffn_mult * d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # --- multi-head self-attention (pre-norm, causal, with RoPE) ---
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        # --- feed-forward (pre-norm, GELU activation) ---
        return x + self.down(F.gelu(self.up(self.n2(x))))


class TinyLM(nn.Module):
    """A minimal decoder-only transformer for function-call routing.

    This is the *custom* architecture used by the main pipeline (train.py /
    infer.py / serve.py).  It is NOT compatible with llama.cpp — for that, see
    ``train_llama.py`` which trains an equivalent model using HF's
    ``LlamaForCausalLM``.

    Args:
        vocab:    vocabulary size (number of BPE tokens + special tokens).
        d:        model / embedding dimension.
        n_layers: number of transformer blocks.
        n_heads:  attention heads per block (d must be divisible by n_heads,
                  and head_dim = d / n_heads must be even for RoPE).
        ffn_mult: FFN inner dimension = ffn_mult × d.
        max_len:  maximum sequence length (for pre-computing RoPE tables).
    """

    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int,
                 ffn_mult: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, ffn_mult) for _ in range(n_layers))
        self.norm = RMSNorm(d)

        # Pre-compute RoPE tables (not saved in state_dict — recomputed on load)
        cos, sin = rope_cache(max_len, d // n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Small-normal init for all weights (good default for small transformers)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        """Initialise Linear and Embedding weights from N(0, 0.02).

        This follows GPT-2's initialisation scheme.  For a model this small it
        doesn't matter much, but it's good practice and makes training curves
        more reproducible.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None):
        """Run a forward pass and optionally compute cross-entropy loss.

        Args:
            idx:    input token ids, shape ``(B, T)``.
            labels: target token ids for the loss, shape ``(B, T)``.  Positions
                    set to -100 are ignored (standard PyTorch convention).
                    Pass ``None`` at inference time to skip loss computation.

        Returns:
            ``(logits, loss)`` — logits has shape ``(B, T, vocab)``; loss is a
            scalar tensor or ``None``.
        """
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        # Tied output head: project through embedding transpose → logits
        logits = self.norm(x) @ self.emb.weight.T
        loss = None
        if labels is not None:
            # Shift: predict token t+1 from position t
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def n_params(self) -> int:
        """Total number of trainable parameters (for logging)."""
        return sum(p.numel() for p in self.parameters())
