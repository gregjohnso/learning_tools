from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 50304
N_LAYER = 12
N_HEAD = 6
N_EMBD = 768


class Rotary(torch.nn.Module):
    """
    Implements rotary positional embeddings.

    This class generates and caches rotary positional embeddings for use in
    transformer models. It uses a base frequency and dimension to compute
    inverse frequencies, which are then used to generate sine and cosine
    embeddings.

    Args:
        dim (int): The dimension of the embeddings.
        base (float, optional): The base for the frequency calculation. Defaults to 10000.

    Attributes:
        inv_freq (torch.Tensor): Inverse frequencies used for embedding calculation.
        seq_len_cached (Optional[int]): Cached sequence length for optimization.
        cos_cached (Optional[torch.Tensor]): Cached cosine embeddings.
        sin_cached (Optional[torch.Tensor]): Cached sine embeddings.
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings for the input tensor.

        This method generates or retrieves cached sine and cosine embeddings
        based on the sequence length of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, ...).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine
            embeddings, each of shape (1, seq_len, 1, dim//2).
        """
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head: int = N_HEAD, n_embd: int = N_EMBD):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = (
            F.rms_norm(q, (q.size(-1),)),
            F.rms_norm(k, (k.size(-1),)),
        )  # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view_as(x)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int = N_EMBD):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_head: int = N_HEAD, n_embd: int = N_EMBD):
        super().__init__()
        self.attn = CausalSelfAttention(n_head, n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# -----------------------------------------------------------------------------
# The main GPT-2 model


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        n_layer: int = N_LAYER,
        n_head: int = N_HEAD,
        n_embd: int = N_EMBD,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                h=nn.ModuleList([Block(n_head, n_embd) for _ in range(n_layer)]),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(
        self,
        idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = logits.float()

        return logits
