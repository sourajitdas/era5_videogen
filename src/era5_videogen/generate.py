import argparse
import json
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from era5_videogen.generating.diffusion import DiffusionSampler

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


def _model_target(model_cfg: DictConfig) -> str:
    return str(model_cfg.get("_target_", ""))


def _is_video3d_model(model_cfg: DictConfig) -> bool:
    target = _model_target(model_cfg)
    return target.endswith(
        (
            "models.swin3d.Swin3D",
            "models.swin3d.SwinV2",
            "models.dit3d.DiT3D",
            "models.songunet3d.SongUNet3D",
            "models.songunet3d.SongUNet",
        )
    )


def _loss_target(loss_cfg: DictConfig) -> str:
    return str(loss_cfg.get("_target_", ""))


def _is_trigflow_loss(cfg: DictConfig) -> bool:
    return _loss_target(cfg.loss).endswith("training.loss.TrigFlowLoss")


def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def _make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6, affine=True)


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


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float,
        temb_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _make_group_norm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = (
            nn.Linear(temb_channels, self.out_channels)
            if temb_channels > 0
            else None
        )
        self.norm2 = _make_group_norm(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            nn.Conv3d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            if self.in_channels != self.out_channels
            else None
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h = _nonlinearity(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(_nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
        h = _nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h


class ConditionProjector3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(out_channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class Decoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...],
        num_res_blocks: int,
        dropout: float,
        in_channels: int,
        z_channels: int,
        cond_channels: int = 0,
        resamp_with_conv: bool = True,
        downsample_strides: tuple[tuple[int, int, int], ...] | None = None,
        give_pre_end: bool = False,
        tanh_out: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.cond_channels = cond_channels
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
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid_condition = (
            ConditionProjector3D(cond_channels, block_in) if cond_channels > 0 else None
        )

        self.up = nn.ModuleList()
        condition_levels: list[nn.Module | None] = []
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            condition_levels.insert(
                0,
                ConditionProjector3D(cond_channels, block_in) if cond_channels > 0 else None,
            )
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
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

        self.level_conditions = nn.ModuleList(
            [module for module in condition_levels if module is not None]
        )
        self.use_condition = cond_channels > 0
        self.out_condition = (
            ConditionProjector3D(cond_channels, block_in) if cond_channels > 0 else None
        )
        self.norm_out = _make_group_norm(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def _fuse_condition(
        self,
        h: torch.Tensor,
        condition: torch.Tensor | None,
        projector: nn.Module | None,
    ) -> torch.Tensor:
        if projector is None or condition is None:
            return h
        cond = F.interpolate(condition, size=h.shape[-3:], mode="nearest")
        return h + projector(cond)

    def forward(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        temb = None
        h = self.conv_in(z)
        h = self._fuse_condition(h, condition, self.mid_condition)
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            projector = self.level_conditions[i_level] if self.use_condition else None
            h = self._fuse_condition(h, condition, projector)
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self._fuse_condition(h, condition, self.out_condition)
        h = self.norm_out(h)
        h = _nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class ERA5VAEDecoder(nn.Module):
    def __init__(
        self,
        model_config: dict,
        decoder_state_dict: dict[str, torch.Tensor],
        fixed_condition: torch.Tensor | None = None,
    ):
        super().__init__()
        model_config = dict(model_config)
        model_config.pop("double_z", None)
        model_config.pop("gradient_checkpointing", None)
        self.cond_channels = int(model_config.get("cond_channels", 0))
        self.decoder = Decoder3D(**model_config)
        self.decoder.load_state_dict(decoder_state_dict)
        self.register_buffer("fixed_condition", torch.empty(0), persistent=False)
        if fixed_condition is not None:
            self.set_condition(fixed_condition)

    @property
    def has_fixed_condition(self) -> bool:
        return self.fixed_condition.numel() > 0

    def set_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim == 4:
            if condition.shape[0] != 1:
                raise ValueError("Fixed condition with 4 dims must have batch size 1")
            condition = condition[0]
        if condition.ndim != 3:
            raise ValueError("Expected fixed condition with shape (C_cond, H, W)")
        if self.cond_channels > 0 and condition.shape[0] != self.cond_channels:
            raise ValueError(
                f"Expected {self.cond_channels} fixed condition channels, got {condition.shape[0]}"
            )
        self.fixed_condition = condition.detach().clone().to(torch.float32)

    def _prepare_condition(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if condition is None and self.has_fixed_condition:
            condition = self.fixed_condition
        if condition is None:
            return None

        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        if condition.ndim == 4:
            condition = condition.unsqueeze(2)
        if condition.ndim != 5:
            raise ValueError(
                "Condition must have shape (C, H, W), (B, C, H, W), or (B, C, T, H, W)"
            )

        if condition.shape[0] == 1 and batch_size != 1:
            condition = condition.expand(batch_size, -1, -1, -1, -1)
        elif condition.shape[0] != batch_size:
            raise ValueError(
                f"Condition batch size {condition.shape[0]} does not match input batch size {batch_size}"
            )

        if condition.shape[2] == 1:
            condition = condition.expand(-1, -1, num_frames, -1, -1)
        elif condition.shape[2] != num_frames:
            condition = F.interpolate(
                condition,
                size=(num_frames, condition.shape[-2], condition.shape[-1]),
                mode="nearest",
            )

        return condition.to(device=device, dtype=dtype)

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_num_frames = z.shape[2]
        for stride in self.decoder.downsample_strides:
            output_num_frames *= int(stride[0])
        prepared_condition = self._prepare_condition(
            batch_size=z.shape[0],
            num_frames=output_num_frames,
            device=z.device,
            dtype=z.dtype,
            condition=condition,
        )
        return self.decoder(z, prepared_condition)


def _torch_load(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed and normalized:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in normalized):
                normalized = {
                    key[len(prefix):]: value
                    for key, value in normalized.items()
                }
                changed = True
    return normalized


def _decoder_state_dict(bundle: dict) -> dict[str, torch.Tensor]:
    if "decoder_state_dict" in bundle:
        return _normalize_state_dict_keys(bundle["decoder_state_dict"])

    model_state = bundle.get("model_state_dict")
    if model_state is None:
        raise KeyError("VAE bundle must contain decoder_state_dict or model_state_dict.")

    decoder_state = {}
    for key, value in _normalize_state_dict_keys(model_state).items():
        if key.startswith("decoder."):
            decoder_state[key[len("decoder."):]] = value
    if not decoder_state:
        raise KeyError("Could not find decoder weights in VAE bundle.")
    return decoder_state


def _load_vae_decoder(
    vae_model_path: str | Path,
    device: torch.device,
) -> ERA5VAEDecoder:
    bundle = _torch_load(vae_model_path, device=torch.device("cpu"))
    model_config = bundle.get("model_config", bundle.get("hyper_parameters"))
    if model_config is None:
        raise KeyError("VAE bundle must contain model_config or hyper_parameters.")
    decoder = ERA5VAEDecoder(
        model_config=model_config,
        decoder_state_dict=_decoder_state_dict(bundle),
        fixed_condition=bundle.get("fixed_condition"),
    )
    return decoder.to(device).eval()


def _vae_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate unconditional ERA5 video samples."
    )
    parser.add_argument(
        "--diffusion_config",
        type=str,
        required=True,
        help="Path to the training config (.yaml), e.g. results/.../.hydra/config.yaml",
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        required=True,
        help="Path to the diffusion checkpoint (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory where generated samples will be written.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional subdirectory name under output_dir.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used to load metadata and fixed conditioning variables.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of video clips to sample.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sampling batch size. Defaults to num_samples.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.005,
        help="Minimum sigma for diffusion sampling.",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum sigma for diffusion sampling.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Rho parameter for the EDM sampler.",
    )
    parser.add_argument(
        "--S_churn",
        type=float,
        default=0.0,
        help="EDM churn parameter.",
    )
    parser.add_argument(
        "--S_min",
        type=float,
        default=0.0,
        help="Minimum sigma for churn.",
    )
    parser.add_argument(
        "--S_max",
        type=float,
        default=float("inf"),
        help="Maximum sigma for churn.",
    )
    parser.add_argument(
        "--S_noise",
        type=float,
        default=1.0,
        help="Noise inflation factor during churn.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu).",
    )
    parser.add_argument(
        "--compile_model",
        "--compile_models",
        action="store_true",
        dest="compile_model",
        help="Compile the diffusion model for faster inference.",
    )
    parser.set_defaults(decode_vae=None)
    parser.add_argument(
        "--decode_vae",
        "--decode",
        action="store_true",
        dest="decode_vae",
        help="Decode sampled latents (VAE or AE) to ERA5 fields before saving.",
    )
    parser.add_argument(
        "--no_decode_vae",
        "--no_decode",
        action="store_false",
        dest="decode_vae",
        help="Save sampled latent tensors without decoding.",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default=None,
        help="Path to a saved VAE bundle. Defaults to <latent_root>/vae_model.pt.",
    )
    parser.add_argument(
        "--ae_model_path",
        type=str,
        default=None,
        help="Path to a saved AE bundle (ae_model.pt). Defaults to <latent_root>/ae_model.pt.",
    )
    parser.add_argument(
        "--vae_decode_batch_size",
        type=int,
        default=None,
        help="Batch size for VAE decoding. Defaults to the sampling batch size.",
    )
    parser.add_argument(
        "--vae_decode_precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Autocast precision for VAE decoding.",
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Also save sampled VAE latents when decoding.",
    )

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1")
    if args.num_steps < 2:
        parser.error("--num_steps must be at least 2")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.vae_decode_batch_size is not None and args.vae_decode_batch_size < 1:
        parser.error("--vae_decode_batch_size must be at least 1")
    return args


def _load_dataset(cfg: DictConfig, split: str):
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.dataset, resolve=True))
    dataset_cfg.split = split
    return instantiate(dataset_cfg, _convert_="object")


def _build_network(cfg: DictConfig, dataset, device: torch.device) -> torch.nn.Module:
    condition_channels = getattr(
        dataset, "condition_channels", getattr(dataset, "cond_channels", 0)
    )
    out_channels = getattr(dataset, "out_channels", dataset.n_channels)
    num_frames = getattr(dataset, "num_frames", None)
    is_video3d = _is_video3d_model(cfg.model)

    if cfg.get("precond") is not None:
        net: torch.nn.Module = instantiate(
            cfg.precond,
            model=cfg.model,
            img_resolution=dataset.img_resolution,
            img_channels=dataset.n_channels,
            condition_channels=condition_channels,
            num_frames=num_frames,
            _recursive_=False,
            _convert_="object",
        )
    else:
        model_kwargs = dict(
            img_resolution=dataset.img_resolution,
            in_channels=dataset.n_channels,
            out_channels=out_channels,
            _convert_="object",
        )
        if is_video3d:
            if num_frames is None:
                raise ValueError("Dataset must define num_frames for 3D video models.")
            model_kwargs["num_frames"] = num_frames
            model_kwargs["condition_channels"] = condition_channels
        net = instantiate(cfg.model, **model_kwargs)

    return net.to(device).eval()


def _checkpoint_state_dict(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        if "ema" in state:
            return state["ema"]
        if "net" in state:
            return state["net"]
    return state


def _condition_batch(dataset, batch_size: int, device: torch.device) -> torch.Tensor | None:
    condition = dataset.get_condition() if hasattr(dataset, "get_condition") else None
    if condition is None:
        return None
    # Static/spatial conditions are (C, H, W) -> broadcast to (B, C, H, W);
    # forecast conditions are (C, T, H, W) -> broadcast to (B, C, T, H, W).
    repeat_arg = (batch_size,) + (1,) * condition.ndim
    return condition.unsqueeze(0).repeat(*repeat_arg).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def _sample_shape(dataset) -> tuple[int, int, int, int]:
    return (
        dataset.n_channels,
        dataset.num_frames,
        *dataset.img_resolution,
    )


def _sample_videos(
    args,
    cfg: DictConfig,
    net: torch.nn.Module,
    dataset,
    device: torch.device,
) -> torch.Tensor:
    sampler = DiffusionSampler(net)
    sample_shape = _sample_shape(dataset)
    batch_size = args.batch_size or args.num_samples
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batches = []
    use_trigflow = _is_trigflow_loss(cfg)
    sigma_data = float(cfg.loss.get("sigma_data", 1.0))

    for start in tqdm(range(0, args.num_samples, batch_size), desc="sampling"):
        cur_batch_size = min(batch_size, args.num_samples - start)
        condition = _condition_batch(dataset, cur_batch_size, device)
        latents = torch.randn(
            (cur_batch_size, *sample_shape),
            generator=generator,
            device=device,
        )
        if use_trigflow:
            samples = sampler.trigflow_sampler(
                latents=latents,
                condition=condition,
                num_steps=args.num_steps,
                sigma_data=sigma_data,
                sigma_min=float(cfg.loss.get("sigma_min", args.sigma_min)),
                sigma_max=float(cfg.loss.get("sigma_max", args.sigma_max)),
                rho=args.rho,
                use_pp=True,
            )
        else:
            samples = sampler.edm_sampler(
                latents=latents,
                condition=condition,
                num_steps=args.num_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                S_churn=args.S_churn,
                S_min=args.S_min,
                S_max=args.S_max,
                S_noise=args.S_noise,
            )
        batches.append(samples.cpu())

    return torch.cat(batches, dim=0)


def _should_decode_vae(args, dataset) -> bool:
    if args.decode_vae is not None:
        return bool(args.decode_vae)
    return bool(getattr(dataset, "is_vae_latent_dataset", False))


def _resolve_vae_model_path(args, dataset) -> str:
    vae_model_path = args.vae_model_path or getattr(dataset, "vae_model_path", None)
    if vae_model_path is None:
        raise FileNotFoundError(
            "VAE decoding requested, but no VAE model path was provided and "
            "the dataset did not expose vae_model_path."
        )
    if not Path(vae_model_path).exists():
        raise FileNotFoundError(f"VAE model bundle not found: {vae_model_path}")
    return str(vae_model_path)


def _latents_for_vae_decode(dataset, samples: torch.Tensor) -> torch.Tensor:
    latents = samples.to(torch.float32)
    if hasattr(dataset, "unnormalize_latents"):
        latents = dataset.unnormalize_latents(latents)
    return latents.to(torch.float32)


def _decode_vae_samples(
    latents: torch.Tensor,
    vae_model_path: str,
    device: torch.device,
    batch_size: int,
    precision: str,
) -> torch.Tensor:
    decoder = _load_vae_decoder(vae_model_path, device=device)
    return _decode_with_module(decoder, latents, device, batch_size, precision)


def _decode_with_module(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
    precision: str,
) -> torch.Tensor:
    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, latents.shape[0], batch_size), desc="decoding"):
            batch = latents[start:start + batch_size].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _vae_autocast(device, precision):
                decoded = decoder(batch)
            batches.append(decoded.float().cpu())

    return torch.cat(batches, dim=0)


def _is_ae_latent_dataset(dataset) -> bool:
    return bool(getattr(dataset, "is_ae_latent_dataset", False))


def clamp_generated_humidity(fields, variables):
    """Clamp specific-humidity channels of DECODED, DENORMALIZED generated
    samples at 0. The AE decoder can emit slightly negative specific humidity
    where q is near zero, which is unphysical. Applies to generated model
    output only, in physical units -- never to ERA5 truth, and never in
    standardized space (where sub-mean values are legitimately negative).

    ``fields`` is a numpy array shaped (C, T, H, W) or (B, C, T, H, W); the
    non-humidity channels are untouched and the array is modified in place.
    """
    qi = [i for i, v in enumerate(variables) if v.startswith("specific_humidity")]
    if not qi:
        return fields
    axis = 0 if fields.ndim == 4 else 1
    sl = [slice(None)] * fields.ndim
    sl[axis] = qi
    fields[tuple(sl)] = np.maximum(fields[tuple(sl)], 0.0)
    return fields


def _resolve_ae_model_path(args, dataset) -> str:
    ae_model_path = (
        args.ae_model_path
        or args.vae_model_path
        or getattr(dataset, "ae_model_path", None)
    )
    if ae_model_path is None:
        raise FileNotFoundError(
            "AE decoding requested, but no AE model path was provided and the "
            "dataset did not expose ae_model_path."
        )
    if not Path(ae_model_path).exists():
        raise FileNotFoundError(f"AE model bundle not found: {ae_model_path}")
    return str(ae_model_path)


def _write_metadata(
    output_dir: Path,
    args,
    cfg: DictConfig,
    dataset,
    samples: np.ndarray,
    decoded_vae: bool = False,
    vae_model_path: str | None = None,
    latent_shape: tuple[int, ...] | None = None,
    ) -> None:
    sampler_name = "trigflow" if _is_trigflow_loss(cfg) else "edm"
    sampler_variant = "dpm++" if sampler_name == "trigflow" else "edm-heun"
    metadata_sigma_min = (
        float(cfg.loss.get("sigma_min", args.sigma_min))
        if sampler_name == "trigflow"
        else float(args.sigma_min)
    )
    metadata_sigma_max = (
        float(cfg.loss.get("sigma_max", args.sigma_max))
        if sampler_name == "trigflow"
        else float(args.sigma_max)
    )
    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "variables": list(getattr(dataset, "variables", [])),
        "condition_variables": list(getattr(dataset, "condition_variables", [])),
        "num_samples": int(samples.shape[0]),
        "channels": int(samples.shape[1]),
        "frames": int(samples.shape[2]),
        "height": int(samples.shape[3]),
        "width": int(samples.shape[4]),
        "split": args.split,
        "sampler": sampler_name,
        "sampler_variant": sampler_variant,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "sigma_data": cfg.loss.get("sigma_data", None),
        "sigma_min": metadata_sigma_min,
        "sigma_max": metadata_sigma_max,
        "rho": args.rho,
        "latent_scale_factor": getattr(dataset, "latent_scale_factor", 1.0),
        "checkpoint": args.diffusion_checkpoint,
        "config": args.diffusion_config,
        "decoded_vae": bool(decoded_vae),
        "vae_model_path": vae_model_path,
        "latent_shape": list(latent_shape) if latent_shape is not None else None,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cfg = OmegaConf.load(args.diffusion_config)
    run_name = args.name or cfg.get("experiment_name", Path(args.diffusion_config).stem)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata from split={args.split}")
    dataset = _load_dataset(cfg, split=args.split)

    print("Building network...")
    net = _build_network(cfg, dataset, device=device)
    net.load_state_dict(_checkpoint_state_dict(args.diffusion_checkpoint, device))

    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this environment.")
        print("Compiling diffusion model...")
        net = torch.compile(net, mode="reduce-overhead")

    print("Sampling videos...")
    samples = _sample_videos(args, cfg, net, dataset, device=device)

    decoded_vae = _should_decode_vae(args, dataset)
    vae_model_path = None
    latent_shape = tuple(samples.shape)

    if decoded_vae:
        # Map sampled latents from the (per-channel standardized / globally
        # scaled) diffusion space back to the raw VAE/AE latent space.
        latents = _latents_for_vae_decode(dataset, samples)
        if args.save_latents:
            np.save(output_dir / "latents.npy", latents.detach().cpu().numpy())

        decode_batch_size = args.vae_decode_batch_size or args.batch_size or args.num_samples
        if _is_ae_latent_dataset(dataset):
            vae_model_path = _resolve_ae_model_path(args, dataset)
            print("Decoding AE latents...")
            decoder = dataset.build_decoder(device, ae_model_path=vae_model_path)
            decoded = _decode_with_module(
                decoder,
                latents,
                device=device,
                batch_size=decode_batch_size,
                precision=args.vae_decode_precision,
            )
        else:
            vae_model_path = _resolve_vae_model_path(args, dataset)
            print("Decoding VAE latents...")
            decoded = _decode_vae_samples(
                latents=latents,
                vae_model_path=vae_model_path,
                device=device,
                batch_size=decode_batch_size,
                precision=args.vae_decode_precision,
            )
        # Map decoded fields from standardized ERA5 space to physical units,
        # then clamp generated specific humidity at 0 (decoder output only).
        samples_np = clamp_generated_humidity(
            dataset.denormalize(decoded), list(dataset.variables))
    else:
        # No decoding: undo the latent normalization so the saved latents are in
        # the raw AE/VAE latent space (denormalize-before-saving).
        samples_np = dataset.denormalize(samples)

    output_path = output_dir / "samples.npy"
    np.save(output_path, samples_np)
    _write_metadata(
        output_dir,
        args,
        cfg,
        dataset,
        samples_np,
        decoded_vae=decoded_vae,
        vae_model_path=vae_model_path,
        latent_shape=latent_shape,
    )

    print(f"Saved samples to {output_path}")
    print(f"Sample array shape: {samples_np.shape}")


if __name__ == "__main__":
    main()
