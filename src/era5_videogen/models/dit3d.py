import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint

from era5_videogen.models.abstract import AbstractNetwork, Shape2D, Shape3D


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """RMS normalization over the last dimension (used for QK-norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("1D sin-cos position embedding dimension must be even.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = 1.0 / 10000 ** (omega / (embed_dim / 2))
    out = torch.einsum("m,d->md", pos.reshape(-1).to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=1).float()


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int, int],
) -> torch.Tensor:
    if embed_dim < 6 or embed_dim % 2 != 0:
        raise ValueError("3D sin-cos position embedding dimension must be an even value >= 6.")

    frames, height, width = grid_size
    grid_t, grid_h, grid_w = torch.meshgrid(
        torch.arange(frames, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    axis_dim = (embed_dim // 6) * 2
    emb_t = get_1d_sincos_pos_embed(axis_dim, grid_t.reshape(-1))
    emb_h = get_1d_sincos_pos_embed(axis_dim, grid_h.reshape(-1))
    emb_w = get_1d_sincos_pos_embed(embed_dim - 2 * axis_dim, grid_w.reshape(-1))
    return torch.cat([emb_t, emb_h, emb_w], dim=1)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10_000,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        # The TrigFlow loss conditions on the angle t=atan(sigma) in [0, pi/2),
        # whose tiny numeric range leaves the sinusoidal embedding nearly
        # constant across noise levels (adjacent levels become indistinguishable).
        # input_scale > 1 rescales t into a range the frequencies can resolve.
        self.input_scale = float(input_scale)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(
            t * self.input_scale, self.frequency_embedding_size, self.max_period
        )
        return self.mlp(t_freq)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: tuple[int, int, int],
        hidden_size: int,
        grid_size: tuple[int, int, int],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)
        self.proj = nn.Linear(in_channels * math.prod(patch_size), hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        return self.proj(
            rearrange(
                x,
                "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
                pt=pt,
                ph=ph,
                pw=pw,
            )
        )


class ConditionEmbed2D(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        patch_size: tuple[int, int, int],
        grid_size: tuple[int, int, int],
        hidden_size: int,
    ):
        super().__init__()
        _, ph, pw = patch_size
        self.patch_size = (ph, pw)
        self.grid_size = grid_size
        self.proj = nn.Linear(condition_channels * ph * pw, hidden_size, bias=True)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        frames, height, width = self.grid_size
        ph, pw = self.patch_size
        tokens = self.proj(
            rearrange(
                condition,
                "b c (h ph) (w pw) -> b (h w) (ph pw c)",
                ph=ph,
                pw=pw,
            )
        )
        tokens = tokens.view(condition.shape[0], height, width, -1)
        tokens = tokens.unsqueeze(1).expand(-1, frames, -1, -1, -1)
        return tokens.reshape(condition.shape[0], frames * height * width, -1)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: tuple[int, int, int],
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT3D(AbstractNetwork):
    """3D diffusion transformer for tensors with shape (B, C, T, H, W)."""

    def __init__(
        self,
        img_resolution: Union[int, tuple[int, int], list[int], ListConfig],
        in_channels: int,
        out_channels: int,
        num_frames: int,
        patch_size: Union[int, tuple[int, int, int], list[int], ListConfig],
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        condition_channels: int = 0,
        auxilary_dim: int = 0,
        timestep_frequency_embedding_size: int = 256,
        timestep_max_period: int = 10_000,
        timestep_input_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
        logvar: bool = False,
    ):
        super().__init__(Shape2D(img_resolution).shape, in_channels, out_channels)
        image_height, image_width = self.img_resolution
        patch_size = Shape3D(patch_size).shape
        patch_time, patch_height, patch_width = patch_size

        if num_frames % patch_time != 0:
            raise ValueError("num_frames must be divisible by patch_size[0].")
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("img_resolution must be divisible by the spatial patch size.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.num_frames = num_frames
        self.patch_size = patch_size
        self.grid_size = (
            num_frames // patch_time,
            image_height // patch_height,
            image_width // patch_width,
        )
        self.condition_channels = condition_channels
        self.auxilary_dim = auxilary_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.x_embedder = PatchEmbed3D(in_channels, patch_size, hidden_size, self.grid_size)
        self.t_embedder = TimestepEmbedder(
            hidden_size,
            frequency_embedding_size=timestep_frequency_embedding_size,
            max_period=timestep_max_period,
            input_scale=timestep_input_scale,
        )
        self.condition_embedder = (
            ConditionEmbed2D(condition_channels, patch_size, self.grid_size, hidden_size)
            if condition_channels > 0
            else None
        )
        self.auxiliary_embedder = nn.Linear(auxilary_dim, hidden_size) if auxilary_dim else None
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.logvar_embed = nn.Linear(hidden_size, 1) if logvar else None

        self.register_buffer(
            "pos_embed",
            get_3d_sincos_pos_embed(hidden_size, self.grid_size).unsqueeze(0),
            persistent=True,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        gt, gh, gw = self.grid_size
        pt, ph, pw = self.patch_size
        return rearrange(
            x,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=gt,
            h=gh,
            w=gw,
            pt=pt,
            ph=ph,
            pw=pw,
            c=self.out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        auxiliary: Optional[torch.Tensor] = None,
        auxilary: Optional[torch.Tensor] = None,
        jvp: bool = False,
        return_logvar: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if auxilary is not None and auxiliary is None:
            auxiliary = auxilary
        if x.ndim != 5:
            raise ValueError("DiT3D expects input with shape (B, C, T, H, W).")

        x = self.x_embedder(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)

        if self.condition_embedder is not None:
            if condition is None:
                raise ValueError("condition must be provided when condition_channels > 0.")
            x = x + self.condition_embedder(condition).to(dtype=x.dtype)

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(x.size(0))
        t = t.reshape(x.size(0))
        c = self.t_embedder(t)
        if self.auxiliary_embedder is not None and auxiliary is not None:
            c = c + self.auxiliary_embedder(auxiliary * math.sqrt(self.auxilary_dim))

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training and not jvp:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.logvar_embed is not None and return_logvar:
            logvar = self.logvar_embed(c).squeeze(-1)
            return x, logvar
        return x
