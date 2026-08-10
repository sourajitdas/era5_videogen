"""Surface-conditioned 3D autoencoder.

Identical model size / input variables / training recipe as :mod:`model`, with
one difference: the decoder is additionally conditioned on two *static* surface
fields -- ``land_sea_mask`` and ``geopotential_at_surface`` -- which are fed in
as decoder-only inputs (they are never reconstructed). The encoder is unchanged
and still takes only the weather variables, so the latent geometry is identical.

Because the land-sea mask and surface geopotential are time-invariant boundary
conditions (the same field in every ERA5 window), they are loaded once from the
dataset at construction time and stored as a non-persistent buffer. At every
decoder resolution level the field is reshaped (spatially resized to the level's
grid and broadcast over the temporal axis) and injected into the feature map via
a small per-level (zero-initialized) projection conv. This keeps the training
loop untouched: ``model(target)`` still returns ``(reconstruction, latent)``.
"""
from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import (
        Autoencoder3D,
        Decoder3D,
        Encoder3D,
        _resolve_channels,
        nonlinearity,
        softclip2,
    )
    from .data import _resolve_path, _resolve_special_values
except ImportError:
    if __package__:
        raise
    from model import (
        Autoencoder3D,
        Decoder3D,
        Encoder3D,
        _resolve_channels,
        nonlinearity,
        softclip2,
    )
    from data import _resolve_path, _resolve_special_values


def _resolve_data_root(data_root: str | Path) -> Path:
    """Resolve a (possibly ``${oc.env:...}`` / relative) dataset root to a path."""
    resolved = _resolve_special_values(str(data_root))
    candidate = Path(resolved)
    if not candidate.is_absolute():
        candidate = _resolve_path(resolved)
    return Path(candidate)


def load_static_cond_fields(
    data_root: str | Path,
    cond_variables: list[str],
    *,
    split: str = "train",
    normalize: bool = True,
) -> torch.Tensor:
    """Load the static decoder-conditioning fields as a ``(1, n_cond, H, W)`` tensor.

    The fields are read from the first ``*.h5`` file of ``split`` (they are
    time-invariant, so any file works) and, when ``normalize`` is set, z-scored
    with the same ``normalize_mean.npz`` / ``normalize_std.npz`` statistics the
    weather inputs use.
    """
    root = _resolve_data_root(data_root)
    files = sorted(glob(os.path.join(str(root), split, "*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No .h5 files found in {os.path.join(str(root), split)} to load "
            f"static conditioning fields {cond_variables}."
        )

    arrays = []
    with h5py.File(files[0], "r") as handle:
        group = handle["input"]
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
        means = np.load(os.path.join(str(root), "normalize_mean.npz"))
        stds = np.load(os.path.join(str(root), "normalize_std.npz"))
        mean = np.array(
            [float(np.asarray(means[name]).reshape(-1)[0]) for name in cond_variables],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
        std = np.array(
            [float(np.asarray(stds[name]).reshape(-1)[0]) for name in cond_variables],
            dtype=np.float32,
        ).reshape(-1, 1, 1)
        fields = (fields - mean) / std

    return torch.from_numpy(np.ascontiguousarray(fields)).unsqueeze(0)  # (1, n_cond, H, W)


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

    Mirrors :class:`model.Autoencoder3D` exactly (encoder, latent saturation,
    channel/block configuration) and only swaps in
    :class:`SurfaceConditionedDecoder3D` plus the static-field buffer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_ch: int,
        cond_variables: list[str],
        data_root: str | Path,
        cond_split: str = "train",
        cond_normalize: bool = True,
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

        cond_fields = load_static_cond_fields(
            data_root,
            self.cond_variables,
            split=cond_split,
            normalize=cond_normalize,
        )
        # Static boundary condition: not learned, recomputed from the dataset at
        # construction, so it is kept out of the checkpoint (persistent=False).
        self.register_buffer("cond_fields", cond_fields, persistent=False)

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
        return self.decoder(z, self.cond_fields)

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


__all__ = [
    "SurfaceConditionedAutoencoder3D",
    "SurfaceConditionedDecoder3D",
    "load_static_cond_fields",
]
