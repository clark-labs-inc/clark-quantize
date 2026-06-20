"""Tiny DiT (Diffusion Transformer) for 32x32 image generation.

A minimal DiT that mirrors the architecture of Sana/FLUX but at ~10M params:
  - Patchify to tokens
  - Time embedding (island)
  - N transformer blocks: self-attention + MLP (SwiGLU)
  - Adaptive norm modulation (island)
  - Final norm + projection (island)

Designed to train in minutes on Apple Silicon MPS.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLU(nn.Module):
    """SwiGLU FFN: hidden = x * silu(gate), out = linear(hidden)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, 4 * dim),  # outputs 6*dim for modulation, but we'll chunk
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """DiT block with adaptive norm modulation, self-attention, and SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)

        # Custom attention (avoids nn.MultiheadAttention which accesses .weight directly)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = SwiGLU(dim, int(dim * mlp_ratio))

        # Modulation: 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(self, x, temb):
        B, N, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        # Self-attention with modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa

        q = self.to_q(h).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(h).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(1, 2).reshape(B, N, D)
        h = self.to_out(h)

        x = x + gate_msa * h

        # FFN with modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ff(h)

        return x


class TinyDiT(nn.Module):
    """Tiny Diffusion Transformer for 32x32 images.

    Architecture:
      - Patch embed: 2x2 patches -> tokens of dim `d`
      - Time embedder (island)
      - N DiT blocks with self-attention + SwiGLU FFN
      - Final norm + projection (island)

    ~10M params with dim=256, depth=6, heads=4.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Island: patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim, bias=True)

        # Island: positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Island: time embedder
        self.time_embed = TimestepEmbedder(dim)
        # time_embed.mlp outputs 4*dim, but we need 6*dim for modulation
        self.tempo_to_mod = nn.Linear(4 * dim, 6 * dim, bias=True)

        # Trunk: transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # Island: final norm + projection
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale_shift_table_out = nn.Parameter(torch.randn(2, dim) / dim ** 0.5)
        self.proj_out = nn.Linear(dim, patch_dim, bias=True)

        self._is_initialized = False
        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[TinyDiT] {total/1e6:.2f}M params  (dim={self.dim}, depth={self.depth}, "
              f"patches={self.num_patches}, heads={self.blocks[0].num_heads})")

    def forward(self, x, t):
        """
        x: [B, C, H, W] input image
        t: [B] timestep (0 to 1)
        Returns: [B, C, H, W] predicted velocity
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Patchify: [B, C, H, W] -> [B, num_patches, patch_dim]
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, C * p * p)

        # Embed
        x = self.patch_embed(x) + self.pos_embed

        # Time embedding -> modulation
        temb = self.time_embed(t)
        temb = self.tempo_to_mod(temb)  # [B, 6*dim]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, temb)

        # Final norm with modulation
        shift, scale = (
            self.scale_shift_table_out[None] + temb[:, :2 * self.dim].reshape(B, 2, -1)
        ).chunk(2, dim=1)
        x = self.norm_out(x)
        x = x * (1 + scale) + shift

        # Project to patches
        x = self.proj_out(x)

        # Unpatchify: [B, num_patches, patch_dim] -> [B, C, H, W]
        x = x.reshape(B, H // p, W // p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return x


# ---------------------------------------------------------------------------
# Flow matching scheduler (simple rectified flow)
# ---------------------------------------------------------------------------

class FlowMatchScheduler:
    """Rectified flow / flow matching scheduler.

    x_t = (1-t)*noise + t*x_0
    v = x_0 - noise  (velocity target)
    x_{t-1} = x_t + v * dt
    """

    def __init__(self, num_steps: int = 20, device: str = "cpu"):
        self.num_steps = num_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    def get_timesteps(self):
        return self.sigmas[:-1]

    def add_noise(self, x0, noise, t):
        """x_t = (1-t)*noise + t*x0, t is [B]."""
        return (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x0

    def velocity_target(self, x0, noise):
        """v = x0 - noise."""
        return x0 - noise

    def step(self, x_t, v_pred, t, dt):
        """Euler step: x_{t+dt} = x_t + v * dt."""
        return x_t + v_pred * dt[:, None, None, None]

    @torch.no_grad()
    def sample(self, model, shape, device, num_steps=None):
        """Generate samples by integrating the flow."""
        num_steps = num_steps or self.num_steps
        dt_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        x = torch.randn(shape, device=device)

        for i in range(num_steps):
            t = torch.full((shape[0],), dt_vals[i], device=device)
            dt = torch.full((shape[0],), dt_vals[i] - dt_vals[i + 1], device=device)
            v = model(x, t)
            x = x + v * dt[0]  # uniform dt

        return x
