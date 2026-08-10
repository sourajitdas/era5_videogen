from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6, affine=True)


def _as_tuple(value, *, length: int | None = None, name: str = "value") -> tuple[int, ...]:
    if isinstance(value, int):
        if length is None:
            return (value,)
        return tuple(value for _ in range(length))
    result = tuple(int(item) for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have {length} entries, got {len(result)}")
    return result


def _resolve_channels(
    *,
    ch: int | None,
    ch_mult: tuple[int, ...] | list[int] | None,
    hid_channels: tuple[int, ...] | list[int] | None,
) -> tuple[int, tuple[int, ...]]:
    if hid_channels is not None:
        channels = _as_tuple(hid_channels, name="hid_channels")
        if not channels:
            raise ValueError("hid_channels must contain at least one level")
        base = int(ch or channels[0])
        if base <= 0:
            raise ValueError("base channel count must be positive")
        if any(channel % base != 0 for channel in channels):
            raise ValueError("Every hid_channels entry must be divisible by ch/base channels")
        return base, tuple(channel // base for channel in channels)

    base = int(ch or 128)
    mult = tuple(ch_mult or (1, 2, 4, 8))
    return base, tuple(int(item) for item in mult)


def softclip2(x: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("softclip2 scale must be positive")
    return x * torch.rsqrt(1.0 + (x / scale).pow(2))


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, scale_factor: tuple[int, int, int], with_conv: bool):
        super().__init__()
        self.scale_factor = tuple(scale_factor)
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, stride: tuple[int, int, int], with_conv: bool):
        super().__init__()
        self.stride = tuple(stride)
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(x)
        return F.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = make_group_norm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = make_group_norm(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                self.nin_shortcut = None
            else:
                self.conv_shortcut = None
                self.nin_shortcut = nn.Conv3d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Encoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int | tuple[int, ...],
        dropout: float,
        in_channels: int,
        z_channels: int,
        resamp_with_conv: bool = True,
        downsample_strides: tuple[tuple[int, int, int], ...] | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = _as_tuple(
            num_res_blocks,
            length=self.num_resolutions,
            name="num_res_blocks",
        )
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.downsample_strides = downsample_strides or tuple(
            (2, 2, 2) for _ in range(self.num_resolutions - 1)
        )

        if len(self.downsample_strides) != self.num_resolutions - 1:
            raise ValueError("downsample_strides must have len(ch_mult) - 1 entries")

        self.conv_in = nn.Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = self.ch

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D(
                    block_in,
                    stride=self.downsample_strides[i_level],
                    with_conv=resamp_with_conv,
                )
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        self.norm_out = make_group_norm(block_in)
        self.conv_out = nn.Conv3d(
            block_in,
            z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not self.gradient_checkpointing:
            return block(x)
        return checkpoint(
            lambda hidden: block(hidden),
            x,
            use_reentrant=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                h = self._run_block(block, h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self._run_block(self.mid.block_1, h)
        h = self._run_block(self.mid.block_2, h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int | tuple[int, ...],
        dropout: float,
        z_channels: int,
        resamp_with_conv: bool = True,
        downsample_strides: tuple[tuple[int, int, int], ...] | None = None,
        decoder_extra_blocks: int = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = _as_tuple(
            num_res_blocks,
            length=self.num_resolutions,
            name="num_res_blocks",
        )
        self.decoder_extra_blocks = int(decoder_extra_blocks)
        self.gradient_checkpointing = gradient_checkpointing
        self.downsample_strides = downsample_strides or tuple(
            (2, 2, 2) for _ in range(self.num_resolutions - 1)
        )

        if len(self.downsample_strides) != self.num_resolutions - 1:
            raise ValueError("downsample_strides must have len(ch_mult) - 1 entries")

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks[i_level] + self.decoder_extra_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                stride = self.downsample_strides[i_level - 1]
                up.upsample = Upsample3D(
                    block_in,
                    scale_factor=stride,
                    with_conv=resamp_with_conv,
                )
            self.up.insert(0, up)

        self.norm_out = make_group_norm(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not self.gradient_checkpointing:
            return block(x)
        return checkpoint(
            lambda hidden: block(hidden),
            x,
            use_reentrant=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self._run_block(self.mid.block_1, h)
        h = self._run_block(self.mid.block_2, h)

        for i_level in reversed(range(self.num_resolutions)):
            for block in self.up[i_level].block:
                h = self._run_block(block, h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class Autoencoder3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_ch: int,
        ch: int | None = None,
        ch_mult: tuple[int, ...] | list[int] | None = None,
        hid_channels: tuple[int, ...] | list[int] | None = None,
        hid_blocks: int | tuple[int, ...] | list[int] | None = None,
        num_res_blocks: int | tuple[int, ...] | list[int] | None = None,
        z_channels: int = 128,
        dropout: float = 0.0,
        downsample_strides: tuple[tuple[int, int, int], ...] | list[list[int]] = (
            (1, 2, 2),
            (2, 2, 2),
            (2, 1, 1),
        ),
        resamp_with_conv: bool = True,
        decoder_extra_blocks: int = 1,
        gradient_checkpointing: bool = False,
        saturation: str | None = "softclip2",
        saturation_scale: float = 5.0,
    ):
        super().__init__()
        base_channels, channel_mult = _resolve_channels(
            ch=ch,
            ch_mult=ch_mult,
            hid_channels=hid_channels,
        )
        block_counts = num_res_blocks if num_res_blocks is not None else hid_blocks
        if block_counts is None:
            block_counts = 2

        self.in_channels = in_channels
        self.out_channels = out_ch
        self.z_channels = z_channels
        self.saturation = "none" if saturation is None else str(saturation).lower()
        self.saturation_scale = float(saturation_scale)

        downsample_strides = tuple(tuple(int(item) for item in stride) for stride in downsample_strides)
        self.encoder = Encoder3D(
            ch=base_channels,
            ch_mult=channel_mult,
            num_res_blocks=block_counts,
            dropout=dropout,
            in_channels=in_channels,
            z_channels=z_channels,
            resamp_with_conv=resamp_with_conv,
            downsample_strides=downsample_strides,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder = Decoder3D(
            ch=base_channels,
            out_ch=out_ch,
            ch_mult=channel_mult,
            num_res_blocks=block_counts,
            dropout=dropout,
            z_channels=z_channels,
            resamp_with_conv=resamp_with_conv,
            downsample_strides=downsample_strides,
            decoder_extra_blocks=decoder_extra_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )

    def apply_latent_saturation(self, z: torch.Tensor) -> torch.Tensor:
        if self.saturation in {"", "none", "identity", "false"}:
            return z
        if self.saturation == "softclip2":
            return softclip2(z, self.saturation_scale)
        if self.saturation == "tanh":
            return torch.tanh(z / self.saturation_scale) * self.saturation_scale
        raise ValueError(f"Unsupported latent saturation: {self.saturation}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.apply_latent_saturation(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        latent_noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        decode_z = z
        if self.training and latent_noise_std > 0.0:
            decode_z = z + torch.randn_like(z) * float(latent_noise_std)
        recon = self.decode(decode_z)
        return recon, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x, latent_noise_std=0.0)
        return recon
