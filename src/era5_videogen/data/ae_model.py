"""Faithful (vendored) copy of the ERA5 ``Autoencoder3D`` used to produce the
``era5_ae`` latents, plus a small loader for the decoder.

The latents are written by the deterministic ``model.Autoencoder3D`` of the
autoencoder training code in ``src/ae``. To decode sampled latents back to ERA5
fields from inside this package, we vendor the exact architecture here so the
saved ``decoder_state_dict`` (and ``model_state_dict``) load without importing
the training code.

Key detail: ``softclip2`` saturation is applied to the *encoder output* (the
latent) inside :meth:`Autoencoder3D.encode`. The decoder has **no** output
saturation, so decoding sampled latents is just ``decoder(z)``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

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

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Checkpoint whenever activations would otherwise be kept for backward
        # (training, or eval-mode guidance gradients in posterior sampling).
        if not self.gradient_checkpointing or not torch.is_grad_enabled() or not x.requires_grad:
            return block(x)
        return checkpoint(lambda hidden: block(hidden), x, use_reentrant=False)

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

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.gradient_checkpointing or not torch.is_grad_enabled() or not x.requires_grad:
            return block(x)
        return checkpoint(lambda hidden: block(hidden), x, use_reentrant=False)

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

    def forward(self, x: torch.Tensor, latent_noise_std: float = 0.0):
        z = self.encode(x)
        decode_z = z
        if self.training and latent_noise_std > 0.0:
            decode_z = z + torch.randn_like(z) * float(latent_noise_std)
        recon = self.decode(decode_z)
        return recon, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x, latent_noise_std=0.0)
        return recon


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed and normalized:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in normalized):
                normalized = {key[len(prefix):]: value for key, value in normalized.items()}
                changed = True
    return normalized


def _filter_model_config(model_config: dict, model_cls=None) -> dict:
    """Keep only kwargs that ``model_cls.__init__`` accepts."""
    model_cls = model_cls or Autoencoder3D
    accepted = set(inspect.signature(model_cls.__init__).parameters)
    accepted.discard("self")
    return {k: v for k, v in dict(model_config).items() if k in accepted}


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ----------------------------------------------------------------------------
# Surface-conditioned variant (vendored from ``ae_new/model1.py``). The decoder
# is additionally conditioned on static 2D surface fields (e.g. land_sea_mask,
# geopotential_at_surface) injected at every resolution level. The encoder and
# latent geometry are identical to the unconditional Autoencoder3D.
# ----------------------------------------------------------------------------


def load_static_cond_fields(
    source_root: str | Path,
    cond_variables: list[str],
    *,
    split: str = "train",
    normalize: bool = True,
) -> torch.Tensor:
    """Load the static decoder-conditioning fields as a ``(1, n_cond, H, W)`` tensor.

    The fields are read from the first ``*.h5`` file of ``split`` of the source
    ERA5 dataset (they are time-invariant, so any file works) and, when
    ``normalize`` is set, z-scored with the same ``normalize_mean.npz`` /
    ``normalize_std.npz`` statistics the weather inputs use.
    """
    import os
    from glob import glob

    import h5py
    import numpy as np

    root = str(source_root)
    files = sorted(glob(os.path.join(root, split, "*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No .h5 files found in {os.path.join(root, split)} to load "
            f"static conditioning fields {cond_variables}."
        )

    arrays = []
    with h5py.File(files[0], "r") as handle:
        group = handle["input"] if "input" in handle else handle
        for name in cond_variables:
            if name not in group:
                raise KeyError(
                    f"Conditioning variable '{name}' not found in {files[0]} (input group)."
                )
            field = np.squeeze(np.asarray(group[name][...], dtype=np.float32))
            if field.ndim != 2:
                raise ValueError(
                    f"Expected static field '{name}' to be 2D (H, W), got shape {field.shape}."
                )
            arrays.append(field)
    fields = np.stack(arrays, axis=0)  # (n_cond, H, W)

    if normalize:
        means = np.load(os.path.join(root, "normalize_mean.npz"))
        stds = np.load(os.path.join(root, "normalize_std.npz"))
        mean = np.array(
            [float(np.asarray(means[name]).reshape(-1)[0]) for name in cond_variables],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
        std = np.array(
            [float(np.asarray(stds[name]).reshape(-1)[0]) for name in cond_variables],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
        fields = (fields - mean) / std

    return torch.from_numpy(np.ascontiguousarray(fields)).unsqueeze(0)


class SurfaceConditionedDecoder3D(Decoder3D):
    """:class:`Decoder3D` that injects static surface fields at every level."""

    def __init__(
        self,
        *,
        n_cond: int,
        cond_kernel: int = 3,
        cond_zero_init: bool = True,
        cond_interp_mode: str = "area",
        **decoder_kwargs,
    ):
        super().__init__(**decoder_kwargs)
        self.n_cond = int(n_cond)
        self.cond_interp_mode = str(cond_interp_mode)

        padding = int(cond_kernel) // 2
        self.cond_proj = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            level_in_channels = self.up[i_level].block[0].in_channels
            conv = nn.Conv3d(
                self.n_cond,
                level_in_channels,
                kernel_size=int(cond_kernel),
                stride=1,
                padding=padding,
            )
            if cond_zero_init:
                nn.init.zeros_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
            self.cond_proj.append(conv)

    def _expand_cond(self, cond2d: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Reshape the ``(1, n_cond, H0, W0)`` field to match ``h`` (B, C, T, H, W)."""
        batch, _, frames, height, width = h.shape
        field = cond2d
        if field.shape[-2:] != (height, width):
            field = F.interpolate(field.float(), size=(height, width), mode=self.cond_interp_mode)
        field = field.to(dtype=h.dtype)
        field = field.unsqueeze(2)  # (1, n_cond, 1, H, W)
        return field.expand(batch, field.shape[1], frames, height, width)

    def forward(self, z: torch.Tensor, cond2d: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self._run_block(self.mid.block_1, h)
        h = self._run_block(self.mid.block_2, h)

        for i_level in reversed(range(self.num_resolutions)):
            cond = self._expand_cond(cond2d, h)
            h = h + self.cond_proj[i_level](cond)
            for block in self.up[i_level].block:
                h = self._run_block(block, h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class SurfaceConditionedAutoencoder3D(nn.Module):
    """Autoencoder whose decoder is conditioned on static surface fields.

    Mirrors :class:`Autoencoder3D` (encoder, latent saturation, channel/block
    configuration) and only swaps in :class:`SurfaceConditionedDecoder3D` plus
    the static-field buffer. The static fields are NOT stored in checkpoints
    (non-persistent buffer in the original training repo), so they must be
    reloaded from the source ERA5 dataset via :func:`load_static_cond_fields`
    and attached with :meth:`set_cond_fields`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_ch: int,
        cond_variables: list[str],
        cond_interp_mode: str = "area",
        cond_kernel: int = 3,
        cond_zero_init: bool = True,
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

        self.cond_variables = list(cond_variables)
        if not self.cond_variables:
            raise ValueError("cond_variables must list at least one static field.")

        self.in_channels = in_channels
        self.out_channels = out_ch
        self.z_channels = z_channels
        self.saturation = "none" if saturation is None else str(saturation).lower()
        self.saturation_scale = float(saturation_scale)

        downsample_strides = tuple(
            tuple(int(item) for item in stride) for stride in downsample_strides
        )
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
        self.decoder = SurfaceConditionedDecoder3D(
            n_cond=len(self.cond_variables),
            cond_kernel=cond_kernel,
            cond_zero_init=cond_zero_init,
            cond_interp_mode=cond_interp_mode,
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
        self.register_buffer("cond_fields", torch.empty(0), persistent=False)

    @property
    def has_cond_fields(self) -> bool:
        return self.cond_fields.numel() > 0

    def set_cond_fields(self, cond_fields: torch.Tensor) -> None:
        if cond_fields.ndim == 3:
            cond_fields = cond_fields.unsqueeze(0)
        if cond_fields.ndim != 4 or cond_fields.shape[1] != len(self.cond_variables):
            raise ValueError(
                f"Expected cond_fields with shape (1, {len(self.cond_variables)}, H, W), "
                f"got {tuple(cond_fields.shape)}"
            )
        self.cond_fields = cond_fields.detach().to(torch.float32)

    def _require_cond_fields(self) -> torch.Tensor:
        if not self.has_cond_fields:
            raise RuntimeError(
                "Surface-conditioned AE has no static cond fields attached. Call "
                "set_cond_fields(load_static_cond_fields(source_root, cond_variables))."
            )
        return self.cond_fields

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
        return self.decoder(z, self._require_cond_fields())

    def forward(self, x: torch.Tensor, latent_noise_std: float = 0.0):
        z = self.encode(x)
        decode_z = z
        if self.training and latent_noise_std > 0.0:
            decode_z = z + torch.randn_like(z) * float(latent_noise_std)
        recon = self.decode(decode_z)
        return recon, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x, latent_noise_std=0.0)
        return recon


def _is_surface_cond_config(model_config: dict) -> bool:
    impl = str(model_config.get("impl", "default")).lower()
    return impl in {"surface_cond", "surface", "model1", "decoder_cond"} or (
        "cond_variables" in model_config and model_config.get("cond_variables")
    )


class AELatentDecoder(nn.Module):
    """Thin wrapper exposing ``decoder(z) -> fields`` for sampled latents.

    For surface-conditioned decoders, the static fields are stored as a buffer
    and passed automatically.
    """

    def __init__(self, decoder: Decoder3D, cond_fields: torch.Tensor | None = None):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("cond_fields", torch.empty(0), persistent=False)
        if cond_fields is not None:
            self.cond_fields = cond_fields.detach().to(torch.float32)

    @property
    def downsample_strides(self):
        return self.decoder.downsample_strides

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if isinstance(self.decoder, SurfaceConditionedDecoder3D):
            if self.cond_fields.numel() == 0:
                raise RuntimeError(
                    "Surface-conditioned decoder requires static cond fields. "
                    "Load it via load_ae_decoder(..., source_root=...)."
                )
            return self.decoder(z, self.cond_fields)
        return self.decoder(z)


def load_ae_model(
    bundle_path: str | Path,
    device: torch.device,
    source_root: str | Path | None = None,
    cond_split: str = "train",
) -> nn.Module:
    """Load the full autoencoder (encoder + decoder) from an ``ae_model.pt`` bundle.

    Dispatches on the bundle's ``model_config['impl']`` between the plain
    :class:`Autoencoder3D` and the :class:`SurfaceConditionedAutoencoder3D`.
    For the surface-conditioned variant the static fields are reloaded from
    ``source_root`` (they are not stored in the bundle).
    """
    bundle = _torch_load(bundle_path, device=torch.device("cpu"))
    model_config = bundle.get("model_config", bundle.get("hyper_parameters"))
    if model_config is None:
        raise KeyError("AE bundle must contain 'model_config' or 'hyper_parameters'.")
    model_config = dict(model_config)

    if _is_surface_cond_config(model_config):
        model = SurfaceConditionedAutoencoder3D(
            **_filter_model_config(model_config, SurfaceConditionedAutoencoder3D)
        )
        if source_root is None:
            raise ValueError(
                "This AE bundle is surface-conditioned; pass source_root so the "
                "static conditioning fields can be reloaded."
            )
        model.set_cond_fields(
            load_static_cond_fields(
                source_root,
                model.cond_variables,
                split=cond_split,
                normalize=bool(model_config.get("cond_normalize", True)),
            )
        )
    else:
        model = Autoencoder3D(**_filter_model_config(model_config, Autoencoder3D))

    model_state = bundle.get("model_state_dict")
    if model_state is not None:
        model.load_state_dict(_normalize_state_dict_keys(model_state))
    else:
        encoder_state = bundle.get("encoder_state_dict")
        decoder_state = bundle.get("decoder_state_dict")
        if encoder_state is None or decoder_state is None:
            raise KeyError(
                "AE bundle must contain 'model_state_dict' or both "
                "'encoder_state_dict' and 'decoder_state_dict'."
            )
        model.encoder.load_state_dict(_normalize_state_dict_keys(encoder_state))
        model.decoder.load_state_dict(_normalize_state_dict_keys(decoder_state))

    return model.to(device).eval()


def load_ae_decoder(
    bundle_path: str | Path,
    device: torch.device,
    source_root: str | Path | None = None,
    cond_split: str = "train",
) -> AELatentDecoder:
    """Load just the decoder from a saved ``ae_model.pt`` bundle.

    The bundle is produced by ``ae_new/save_latents.py`` and contains
    ``model_config`` plus ``decoder_state_dict`` (and a full ``model_state_dict``).
    We build the full autoencoder so the channel/stride geometry is guaranteed
    to match, load the decoder weights, and return only the decoder. For
    surface-conditioned bundles the static fields are reloaded from
    ``source_root`` and baked into the returned wrapper.
    """
    bundle = _torch_load(bundle_path, device=torch.device("cpu"))
    model_config = bundle.get("model_config", bundle.get("hyper_parameters"))
    if model_config is None:
        raise KeyError("AE bundle must contain 'model_config' or 'hyper_parameters'.")
    model_config = dict(model_config)

    if _is_surface_cond_config(model_config):
        model = load_ae_model(
            bundle_path, device=torch.device("cpu"), source_root=source_root,
            cond_split=cond_split,
        )
        decoder = AELatentDecoder(model.decoder, cond_fields=model.cond_fields)
        return decoder.to(device).eval()

    model = Autoencoder3D(**_filter_model_config(model_config, Autoencoder3D))

    decoder_state = bundle.get("decoder_state_dict")
    if decoder_state is not None:
        model.decoder.load_state_dict(_normalize_state_dict_keys(decoder_state))
    else:
        model_state = bundle.get("model_state_dict")
        if model_state is None:
            raise KeyError(
                "AE bundle must contain 'decoder_state_dict' or 'model_state_dict'."
            )
        model.load_state_dict(_normalize_state_dict_keys(model_state))

    decoder = AELatentDecoder(model.decoder)
    return decoder.to(device).eval()
