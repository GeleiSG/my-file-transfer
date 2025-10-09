from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.window_size = window_size
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        seqlen = x.size(1)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        is_causal, mask = True, None
        if self.window_size is not None:
            is_causal = False
            mask = torch.ones(size=(seqlen, seqlen), device=x.device, dtype=x.dtype)
            mask.tril_(diagonal=0).triu_(diagonal=1 - self.window_size)
            mask.log_()
        # with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(x.size(0), seqlen, -1)
        output = self.wo(output)
        return output


def compute_loss(preds: Tensor, targets: Tensor) -> Tensor:
    logits, targets = preds.flatten(0, 1), targets.flatten(0, 1)
    return F.cross_entropy(logits, targets)


if __name__ == "__main__":
    dtype, device = torch.bfloat16, torch.device("cuda")
    layer = Attention(128, window_size=8)
    proj = nn.Linear(128, 1024, bias=False)
    model = nn.Sequential(layer, proj).to(device=device, dtype=dtype)

    x = torch.randn(256, 64, 128, device=device, dtype=dtype)
    out = model(x)
    targets = torch.randint(low=0, high=1024, size=(256, 64), device=device)
    loss = compute_loss(out, targets)
    loss.backward()