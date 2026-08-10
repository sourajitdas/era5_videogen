from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

try:
    from .data import ERA5VideoDataModule
    from .model import Autoencoder3D
except ImportError:
    if __package__:
        raise
    from data import ERA5VideoDataModule
    from model import Autoencoder3D


CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if not is_distributed():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def init_distributed_mode(device: str = "auto") -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    if not distributed:
        return {
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": resolve_device(device),
        }

    requested_device = resolve_device(device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA DDP requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        resolved_device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        resolved_device = torch.device("cpu")
        backend = "gloo"

    if backend == "nccl":
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=resolved_device,
            )
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        dist.init_process_group(backend=backend, init_method="env://")

    return {
        "distributed": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": resolved_device,
    }


def configure_cuda_runtime(train_cfg: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", False))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name] = tensor.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        self.backup = {}
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.backup[name] = tensor.detach().clone()
                tensor.copy_(self.shadow[name])
        try:
            yield
        finally:
            state_dict = model.state_dict()
            for name, tensor in state_dict.items():
                if name in self.backup:
                    tensor.copy_(self.backup[name])
            self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.shadow = normalize_state_dict_keys(state_dict["shadow"])


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
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


def torch_load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
):
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def checkpoint_path_for_epoch(checkpoint_dir: str | Path, epoch: int) -> Path:
    return Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    latest_epoch = None
    latest_path = None
    for path in checkpoint_dir.glob("epoch_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if latest_epoch is None or epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    global_step: int,
    best_val_rmse: float,
    best_epoch: int | None,
    config: dict[str, Any],
    ema: ModelEMA | None = None,
    val_metrics: dict[str, Any] | None = None,
) -> None:
    if not is_main_process():
        return
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "weights": normalize_state_dict_keys(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "ema": ema.state_dict() if ema is not None else None,
    }
    if val_metrics is not None:
        checkpoint["val_metrics"] = val_metrics

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ema: ModelEMA | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=map_location)
    weights = checkpoint.get("weights", checkpoint.get("model"))
    if weights is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model weights.")
    model.load_state_dict(normalize_state_dict_keys(weights))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    return checkpoint


def write_metrics(log_path: str | Path, metrics: dict[str, Any]) -> None:
    if not is_main_process():
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def broadcast_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if is_distributed():
        dist.broadcast(tensor, src=0)
    return float(tensor.item())


def reduce_metric_sums(
    metric_sums: dict[str, float | torch.Tensor],
    count: int,
    device: torch.device,
) -> dict[str, float]:
    keys = list(metric_sums.keys())
    values = []
    for key in keys:
        value = metric_sums[key]
        if isinstance(value, torch.Tensor):
            values.append(value.detach().to(device=device, dtype=torch.float64))
        else:
            values.append(torch.tensor(float(value), device=device, dtype=torch.float64))
    values.append(torch.tensor(float(count), device=device, dtype=torch.float64))
    tensor = torch.stack(values)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    total_count = max(int(round(tensor[-1].item())), 1)
    return {
        key: float(tensor[idx].item() / total_count)
        for idx, key in enumerate(keys)
    }


def count_model_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_params, trainable_params


def prepare_model_config(config: dict[str, Any], datamodule: ERA5VideoDataModule) -> dict[str, Any]:
    model_cfg = dict(config["model"])

    model_cfg.setdefault("in_channels", datamodule.n_channels)
    model_cfg.setdefault("out_ch", datamodule.n_channels)
    model_cfg.setdefault("gradient_checkpointing", True)

    if model_cfg["in_channels"] != datamodule.n_channels:
        raise ValueError(
            f"Model in_channels={model_cfg['in_channels']} but dataset has {datamodule.n_channels}"
        )
    if model_cfg["out_ch"] != datamodule.n_channels:
        raise ValueError(
            f"Model out_ch={model_cfg['out_ch']} but dataset has {datamodule.n_channels}"
        )
    return model_cfg


# Aliases for the selectable model implementations. "default" is the plain
# Autoencoder3D from model.py; "surface_cond" is the decoder-conditioned variant
# from model1.py (selected via the optional model.impl config key).
_DEFAULT_IMPLS = {"default", "base", "model", "autoencoder3d", "none", ""}
_SURFACE_COND_IMPLS = {"surface_cond", "surface", "model1", "decoder_cond"}


def build_autoencoder(
    model_cfg: dict[str, Any],
    datamodule: ERA5VideoDataModule,
) -> torch.nn.Module:
    """Instantiate the autoencoder selected by the optional ``model.impl`` key.

    ``model_cfg`` is consumed in place (the ``impl`` key is popped). When
    ``impl`` selects the surface-conditioned model, the resolved dataset root is
    injected so the static conditioning fields are loaded from the same data.
    """
    model_cfg = dict(model_cfg)
    impl = str(model_cfg.pop("impl", "default")).lower()
    if impl in _DEFAULT_IMPLS:
        return Autoencoder3D(**model_cfg)
    if impl in _SURFACE_COND_IMPLS:
        try:
            from .model1 import SurfaceConditionedAutoencoder3D
        except ImportError:
            if __package__:
                raise
            from model1 import SurfaceConditionedAutoencoder3D
        model_cfg.setdefault("data_root", datamodule.dataset_kwargs["root"])
        return SurfaceConditionedAutoencoder3D(**model_cfg)
    raise ValueError(
        f"Unknown model.impl='{impl}'. Use one of {sorted(_DEFAULT_IMPLS | _SURFACE_COND_IMPLS)}."
    )


def _downsample_extent(length: int, stride: int, *, with_conv: bool) -> int:
    if stride < 1:
        raise ValueError(f"Downsample stride must be positive, got {stride}")
    if with_conv:
        return math.ceil(length / stride)
    return length // stride


def infer_latent_shape(
    model_cfg: dict[str, Any],
    datamodule: ERA5VideoDataModule,
) -> tuple[int, int, int, int]:
    frames = int(datamodule.num_frames)
    height, width = (int(item) for item in datamodule.img_resolution)
    with_conv = bool(model_cfg.get("resamp_with_conv", True))
    for stride in model_cfg.get("downsample_strides", []):
        if len(stride) != 3:
            raise ValueError(f"Each downsample stride must have 3 entries, got {stride}")
        frames = _downsample_extent(frames, int(stride[0]), with_conv=with_conv)
        height = _downsample_extent(height, int(stride[1]), with_conv=with_conv)
        width = _downsample_extent(width, int(stride[2]), with_conv=with_conv)
    return int(model_cfg["z_channels"]), frames, height, width


def validate_latent_geometry(
    model_cfg: dict[str, Any],
    datamodule: ERA5VideoDataModule,
) -> tuple[int, int, int, int]:
    latent_shape = infer_latent_shape(model_cfg, datamodule)
    expected_shape = model_cfg.get("expected_latent_shape")
    if expected_shape is not None and tuple(int(item) for item in expected_shape) != latent_shape:
        raise ValueError(
            "Configured expected_latent_shape does not match the model/data geometry: "
            f"expected {tuple(expected_shape)}, inferred {latent_shape}"
        )
    return latent_shape


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    optim_cfg: dict[str, Any],
    *,
    steps_per_epoch: int,
    num_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        betas=(optim_cfg["beta_1"], optim_cfg["beta_2"]),
        weight_decay=optim_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs * steps_per_epoch),
        eta_min=optim_cfg["eta_min"],
    )
    return optimizer, scheduler


def build_latitude_weights(datamodule: ERA5VideoDataModule) -> torch.Tensor:
    lat, _ = datamodule.get_lat_lon()
    lat_tensor = torch.as_tensor(lat, dtype=torch.float32)
    max_abs_lat = float(lat_tensor.abs().max())
    lat_radians = lat_tensor if max_abs_lat <= math.pi + 1.0e-4 else torch.deg2rad(lat_tensor)
    weights = torch.cos(lat_radians).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1.0e-6)
    if weights.ndim == 1:
        return weights.view(1, 1, 1, -1, 1)
    if weights.ndim == 2:
        return weights.view(1, 1, 1, *weights.shape)
    raise ValueError(f"Expected latitude array with 1 or 2 dimensions, got {tuple(weights.shape)}")


def parse_variable_weights(
    spec: str | dict[str, float] | None,
    variables: list[str],
) -> torch.Tensor | None:
    if not spec:
        return None
    if isinstance(spec, dict):
        mapping = spec
    else:
        spec = spec.strip()
        if spec.startswith("{"):
            mapping = json.loads(spec)
        else:
            maybe_path = Path(spec)
            if maybe_path.exists():
                mapping = json.loads(maybe_path.read_text(encoding="utf-8"))
            else:
                mapping = {}
                for item in spec.split(","):
                    if not item.strip():
                        continue
                    name, value = item.split("=", maxsplit=1)
                    mapping[name.strip()] = float(value)

    weights = torch.tensor(
        [float(mapping.get(variable_name, 1.0)) for variable_name in variables],
        dtype=torch.float32,
    )
    return weights / weights.mean().clamp_min(1.0e-6)


def combine_loss_weights(
    *,
    latitude_weights: torch.Tensor | None,
    variable_weights: torch.Tensor | None,
    target: torch.Tensor,
) -> torch.Tensor | None:
    weights = None
    if variable_weights is not None:
        weights = variable_weights.to(device=target.device, dtype=target.dtype).view(
            1,
            target.shape[1],
            1,
            1,
            1,
        )
    if latitude_weights is not None:
        lat = latitude_weights.to(device=target.device, dtype=target.dtype)
        weights = lat if weights is None else weights * lat
    return weights


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    expanded_weights = weights.expand_as(values)
    return (values * expanded_weights).sum() / expanded_weights.sum().clamp_min(1.0)


def spatial_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    if prediction.shape[-2] > 1:
        dy_error = prediction[..., 1:, :] - prediction[..., :-1, :]
        dy_error.sub_(target[..., 1:, :]).add_(target[..., :-1, :])
        losses.append(dy_error.pow(2).mean())
    if prediction.shape[-1] > 1:
        dx_error = prediction[..., 1:] - prediction[..., :-1]
        dx_error.sub_(target[..., 1:]).add_(target[..., :-1])
        losses.append(dx_error.pow(2).mean())
    if not losses:
        return prediction.new_zeros(())
    return sum(losses) / len(losses)


def temporal_difference_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[2] <= 1:
        return prediction.new_zeros(())
    dt_error = prediction[:, :, 1:] - prediction[:, :, :-1]
    dt_error.sub_(target[:, :, 1:]).add_(target[:, :, :-1])
    return dt_error.pow(2).mean()


def laplacian_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize differences in the spatial Laplacian (a high-frequency / sharpness term)."""
    if prediction.shape[-2] < 3 or prediction.shape[-1] < 3:
        return prediction.new_zeros(())

    def _laplacian(field: torch.Tensor) -> torch.Tensor:
        return (
            field[..., 2:, 1:-1]
            + field[..., :-2, 1:-1]
            + field[..., 1:-1, 2:]
            + field[..., 1:-1, :-2]
            - 4.0 * field[..., 1:-1, 1:-1]
        )

    lap_error = _laplacian(prediction) - _laplacian(target)
    return lap_error.pow(2).mean()


def autoencoder_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    squared_error = (prediction - target).pow(2)
    recon_loss = squared_error.mean()
    objective_error = squared_error
    if bool(train_cfg.get("variance_normalized_mse", True)):
        variance_dims = (0, 2, 3, 4)
        variance = target.float().var(dim=variance_dims, unbiased=False).to(dtype=target.dtype)
        variance = variance.view(1, target.shape[1], 1, 1, 1)
        variance_epsilon = float(train_cfg.get("variance_epsilon", 1.0e-2))
        objective_error = objective_error / variance.add(variance_epsilon)
    weighted_recon_loss = weighted_mean(objective_error, weights)
    spatial_weight = float(train_cfg.get("spatial_gradient_loss_weight", 0.0) or 0.0)
    temporal_weight = float(train_cfg.get("temporal_difference_loss_weight", 0.0) or 0.0)
    laplacian_weight = float(train_cfg.get("laplacian_loss_weight", 0.0) or 0.0)
    grad_loss = spatial_gradient_loss(prediction, target) if spatial_weight else prediction.new_zeros(())
    temporal_loss = temporal_difference_loss(prediction, target) if temporal_weight else prediction.new_zeros(())
    lap_loss = laplacian_loss(prediction, target) if laplacian_weight else prediction.new_zeros(())
    total_loss = (
        weighted_recon_loss
        + spatial_weight * grad_loss
        + temporal_weight * temporal_loss
        + laplacian_weight * lap_loss
    )
    metrics = {
        "recon_loss": recon_loss.detach(),
        "weighted_recon_loss": weighted_recon_loss.detach(),
        "spatial_grad_loss": grad_loss.detach(),
        "temporal_diff_loss": temporal_loss.detach(),
        "laplacian_loss": lap_loss.detach(),
    }
    return total_loss, metrics


def get_denormalization_stds(datamodule: Any, device: torch.device) -> torch.Tensor | None:
    data_train = getattr(datamodule, "data_train", None)
    stds = getattr(data_train, "stds", None)
    if stds is None:
        return None
    return torch.as_tensor(stds, device=device, dtype=torch.float32)


VALIDATION_PHYSICAL_RMSE_VARIABLES = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
}


def compute_normalized_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    diff = prediction.float() - target.float()
    return float(torch.sqrt(diff.square().mean()).detach())


def compute_physical_rmse_by_variable(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    variables: list[str],
    denormalize_std: torch.Tensor | np.ndarray | None = None,
    denormalize_fn=None,
) -> dict[str, float]:
    results = {}
    variable_to_index = {name: idx for idx, name in enumerate(variables)}

    if denormalize_std is not None:
        diff = prediction.float() - target.float()
        std = torch.as_tensor(denormalize_std, device=diff.device, dtype=diff.dtype)
        for short_name, variable_name in VALIDATION_PHYSICAL_RMSE_VARIABLES.items():
            variable_idx = variable_to_index.get(variable_name)
            if variable_idx is None:
                continue
            channel_diff = diff[:, variable_idx:variable_idx + 1]
            channel_std = std[variable_idx].view(1, 1, 1, *std.shape[1:])
            physical_diff = channel_diff * channel_std
            results[f"physical_rmse_{short_name}"] = float(
                torch.sqrt(physical_diff.square().mean()).detach()
            )
        return results

    if denormalize_fn is None:
        return results

    prediction_np = denormalize_fn(prediction.float())
    target_np = denormalize_fn(target.float())
    diff_np = prediction_np - target_np
    for short_name, variable_name in VALIDATION_PHYSICAL_RMSE_VARIABLES.items():
        variable_idx = variable_to_index.get(variable_name)
        if variable_idx is None:
            continue
        channel_diff = diff_np[:, variable_idx]
        results[f"physical_rmse_{short_name}"] = float(
            np.sqrt(np.mean(channel_diff ** 2))
        )
    return results


def format_validation_rmse(metrics: dict[str, float]) -> str:
    parts = [f"norm_rmse={metrics['normalized_rmse']:.6f}"]
    for short_name in VALIDATION_PHYSICAL_RMSE_VARIABLES:
        key = f"physical_rmse_{short_name}"
        if key in metrics:
            parts.append(f"{short_name}_rmse={metrics[key]:.6f}")
    return " ".join(parts)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    ema_source_model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    train_cfg: dict[str, Any],
    scaler,
    ema: ModelEMA | None,
    latitude_weights: torch.Tensor | None,
    variable_weights: torch.Tensor | None,
    epoch: int,
    global_step: int,
    log_progress: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    accumulation_steps = max(1, train_cfg.get("gradient_accumulation_steps", 1))
    running = {
        "loss": torch.zeros((), device=device, dtype=torch.float64),
        "recon_loss": torch.zeros((), device=device, dtype=torch.float64),
        "weighted_recon_loss": torch.zeros((), device=device, dtype=torch.float64),
        "latent_noise_std": torch.zeros((), device=device, dtype=torch.float64),
        "spatial_grad_loss": torch.zeros((), device=device, dtype=torch.float64),
        "temporal_diff_loss": torch.zeros((), device=device, dtype=torch.float64),
        "laplacian_loss": torch.zeros((), device=device, dtype=torch.float64),
    }
    steps = 0
    accumulation_counter = 0
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(dataloader)
    if train_cfg["limit_train_batches"] is not None:
        total_batches = min(total_batches, train_cfg["limit_train_batches"])

    for batch_idx, batch in enumerate(dataloader):
        if train_cfg["max_steps"] is not None and global_step >= train_cfg["max_steps"]:
            break

        target = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        weights = combine_loss_weights(
            latitude_weights=latitude_weights,
            variable_weights=variable_weights,
            target=target,
        )

        accumulation_counter += 1
        should_step = accumulation_counter == accumulation_steps or batch_idx + 1 >= total_batches
        sync_context = model.no_sync() if hasattr(model, "no_sync") and not should_step else nullcontext()

        with sync_context:
            with autocast_context(device, train_cfg["precision"]):
                latent_noise_std = float(train_cfg.get("latent_noise_std", 0.0) or 0.0)
                reconstruction, _ = model(
                    target,
                    latent_noise_std=latent_noise_std,
                )
                loss, loss_metrics = autoencoder_loss(
                    reconstruction,
                    target,
                    weights=weights,
                    train_cfg=train_cfg,
                )
                loss_metrics["latent_noise_std"] = target.new_tensor(latent_noise_std)
                scaled_loss = loss / accumulation_steps

            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()

        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if accumulation_counter < accumulation_steps:
                grad_scale = accumulation_steps / accumulation_counter
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(grad_scale)
            if train_cfg["grad_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])

            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if ema is not None:
                ema.update(ema_source_model)
            global_step += 1
            accumulation_counter = 0

        steps += 1
        running["loss"] += loss.detach().to(dtype=torch.float64)
        for key, value in loss_metrics.items():
            running[key] += value.to(dtype=torch.float64)

        if (
            should_step
            and log_progress
            and train_cfg["log_every_steps"]
            and global_step > 0
            and global_step % train_cfg["log_every_steps"] == 0
        ):
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[ae] epoch={epoch} step={global_step} "
                f"loss={loss.detach().item():.6f} "
                f"recon={loss_metrics['recon_loss'].item():.6f} "
                f"weighted_recon={loss_metrics['weighted_recon_loss'].item():.6f} "
                f"latent_noise={loss_metrics['latent_noise_std'].item():.3e} "
                f"lr={current_lr:.3e}",
            )

        if train_cfg["limit_train_batches"] is not None and batch_idx + 1 >= train_cfg["limit_train_batches"]:
            break

    metrics = reduce_metric_sums(running, steps, device)
    metrics["global_step"] = global_step
    return metrics, global_step


@torch.no_grad()
def evaluate(
    *,
    model: Autoencoder3D,
    dataloader,
    device: torch.device,
    datamodule: ERA5VideoDataModule,
    train_cfg: dict[str, Any],
    latitude_weights: torch.Tensor | None,
    variable_weights: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    metrics = {
        "loss": torch.zeros((), device=device, dtype=torch.float64),
        "recon_loss": torch.zeros((), device=device, dtype=torch.float64),
        "weighted_recon_loss": torch.zeros((), device=device, dtype=torch.float64),
        "spatial_grad_loss": torch.zeros((), device=device, dtype=torch.float64),
        "temporal_diff_loss": torch.zeros((), device=device, dtype=torch.float64),
        "laplacian_loss": torch.zeros((), device=device, dtype=torch.float64),
        "normalized_rmse": 0.0,
        "rmse": 0.0,
    }
    for short_name, variable_name in VALIDATION_PHYSICAL_RMSE_VARIABLES.items():
        if variable_name in datamodule.variables:
            metrics[f"physical_rmse_{short_name}"] = 0.0
    steps = 0
    denormalize_std = get_denormalization_stds(datamodule, device)

    for batch_idx, batch in enumerate(dataloader):
        target = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        weights = combine_loss_weights(
            latitude_weights=latitude_weights,
            variable_weights=variable_weights,
            target=target,
        )

        with autocast_context(device, train_cfg["precision"]):
            reconstruction, _ = model(target)
            loss, loss_metrics = autoencoder_loss(
                reconstruction,
                target,
                weights=weights,
                train_cfg=train_cfg,
            )

        metrics["loss"] += loss.detach().to(dtype=torch.float64)
        for key, value in loss_metrics.items():
            metrics[key] += value.to(dtype=torch.float64)
        normalized_rmse = compute_normalized_rmse(reconstruction, target)
        metrics["normalized_rmse"] += normalized_rmse
        metrics["rmse"] += normalized_rmse
        physical_rmses = compute_physical_rmse_by_variable(
            reconstruction,
            target,
            variables=datamodule.variables,
            denormalize_fn=datamodule.denormalize if denormalize_std is None else None,
            denormalize_std=denormalize_std,
        )
        for key, value in physical_rmses.items():
            metrics[key] += value
        steps += 1

        if train_cfg["limit_val_batches"] is not None and batch_idx + 1 >= train_cfg["limit_val_batches"]:
            break

    return reduce_metric_sums(metrics, steps, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ERA5 3D autoencoder.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to autoencoder config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    distributed_state = init_distributed_mode(device=config["train"]["device"])
    device = distributed_state["device"]
    rank = distributed_state["rank"]
    world_size = distributed_state["world_size"]
    distributed = distributed_state["distributed"]
    configure_cuda_runtime(config["train"], device)

    try:
        set_seed(config.get("seed", 42) + rank)

        output_dir = Path(config["train"]["output_dir"])
        checkpoint_dir = output_dir / "checkpoints"
        latest_checkpoint = None
        if config["train"].get("resume", True):
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, output_dir / "requested_config.yaml")
            with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            if distributed:
                print(f"Running DDP autoencoder training with world_size={world_size} on device={device}")
        barrier()

        datamodule = ERA5VideoDataModule(**config["data"])
        datamodule.setup()

        model_cfg = prepare_model_config(config, datamodule)
        latent_shape = validate_latent_geometry(model_cfg, datamodule)
        model_cfg.pop("expected_latent_shape", None)
        raw_model = build_autoencoder(model_cfg, datamodule).to(device)

        if is_main_process():
            total_params, trainable_params = count_model_parameters(raw_model)
            print(f"Model parameters: total={total_params:,} trainable={trainable_params:,}")

        # Finetune init: when starting a fresh run (no checkpoint to resume), optionally
        # load weights only from another checkpoint so the optimizer/LR schedule restart.
        finetune_from = config["train"].get("finetune_from")
        if latest_checkpoint is None and finetune_from:
            finetune_path = Path(finetune_from)
            if not finetune_path.exists():
                raise FileNotFoundError(f"finetune_from checkpoint not found: {finetune_path}")
            finetune_ckpt = torch_load_checkpoint(finetune_path, map_location=device)
            finetune_weights = finetune_ckpt.get("weights", finetune_ckpt.get("model"))
            if finetune_weights is None:
                raise KeyError(f"finetune_from checkpoint {finetune_path} has no model weights.")
            raw_model.load_state_dict(normalize_state_dict_keys(finetune_weights))
            if is_main_process():
                print(f"Initialized weights from finetune_from={finetune_path} (fresh optimizer + LR schedule)")

        model = raw_model
        if distributed:
            ddp_kwargs = {"broadcast_buffers": True}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [distributed_state["local_rank"]]
                ddp_kwargs["output_device"] = distributed_state["local_rank"]
            model = DDP(raw_model, **ddp_kwargs)

        train_loader = datamodule.train_dataloader(
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        val_num_workers = None
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
        if distributed and slurm_cpus and slurm_cpus <= 1:
            val_num_workers = 0
            if is_main_process():
                print("Validation: forcing num_workers=0 because SLURM_CPUS_PER_TASK<=1")

        val_loader = datamodule.val_dataloader(
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            num_workers=val_num_workers,
        )

        steps_per_epoch = len(train_loader)
        if config["train"]["limit_train_batches"] is not None:
            steps_per_epoch = min(steps_per_epoch, config["train"]["limit_train_batches"])
        steps_per_epoch = max(
            1,
            math.ceil(
                steps_per_epoch
                / max(1, config["train"].get("gradient_accumulation_steps", 1))
            ),
        )
        optimizer, scheduler = build_optimizer_and_scheduler(
            raw_model,
            config["optim"],
            steps_per_epoch=steps_per_epoch,
            num_epochs=config["train"]["epochs"],
        )
        ema = None
        if config["optim"].get("ema_decay"):
            ema = ModelEMA(raw_model, decay=config["optim"]["ema_decay"])

        start_epoch = 1
        global_step = 0
        best_val_rmse = float("inf")
        best_epoch = None
        if latest_checkpoint is not None:
            checkpoint = load_checkpoint(
                latest_checkpoint,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                map_location=device,
            )
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            best_val_rmse = checkpoint.get("best_val_rmse", float("inf"))
            best_epoch = checkpoint.get("best_epoch")
            if is_main_process():
                print(
                    f"Resumed autoencoder from {latest_checkpoint} at epoch {checkpoint['epoch']} "
                    f"step {global_step}"
                )

        scaler = None
        if device.type == "cuda" and config["train"]["precision"] == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        loss_cfg = config.get("loss", {})
        config["train"]["spatial_gradient_loss_weight"] = loss_cfg.get(
            "spatial_gradient_loss_weight",
            config["train"].get("spatial_gradient_loss_weight", 0.0),
        )
        config["train"]["temporal_difference_loss_weight"] = loss_cfg.get(
            "temporal_difference_loss_weight",
            config["train"].get("temporal_difference_loss_weight", 0.0),
        )
        config["train"]["laplacian_loss_weight"] = loss_cfg.get(
            "laplacian_loss_weight",
            config["train"].get("laplacian_loss_weight", 0.0),
        )
        config["train"]["variance_normalized_mse"] = loss_cfg.get(
            "variance_normalized_mse",
            config["train"].get("variance_normalized_mse", True),
        )
        config["train"]["variance_epsilon"] = loss_cfg.get(
            "variance_epsilon",
            config["train"].get("variance_epsilon", 1.0e-2),
        )
        latitude_weights = build_latitude_weights(datamodule) if loss_cfg.get("latitude_weight", True) else None
        variable_weights = parse_variable_weights(loss_cfg.get("variable_weights"), datamodule.variables)

        if is_main_process():
            print("Autoencoder training:")
            print(f"  output dir: {output_dir}")
            print(f"  z channels: {model_cfg['z_channels']}")
            print(f"  latent shape: {latent_shape}")
            print(
                "  latent decoder noise: "
                f"{float(config['train'].get('latent_noise_std', 0.0) or 0.0):.3e}"
            )
            print(f"  optimizer steps per epoch: {steps_per_epoch}")
            print(f"  lr schedule: cosine without warmup")
            print(f"  latitude weighting: {loss_cfg.get('latitude_weight', True)}")
            print(f"  variance-normalized MSE: {config['train']['variance_normalized_mse']}")
            print(f"  spatial gradient weight: {config['train']['spatial_gradient_loss_weight']}")
            print(f"  laplacian weight: {config['train']['laplacian_loss_weight']}")
            print(f"  temporal difference weight: {config['train']['temporal_difference_loss_weight']}")

        if latest_checkpoint is None and val_loader is not None:
            barrier()
            context = ema.average_parameters(raw_model) if ema is not None else torch.no_grad()
            with context:
                initial_val_metrics = evaluate(
                    model=raw_model,
                    dataloader=val_loader,
                    device=device,
                    datamodule=datamodule,
                    train_cfg=config["train"],
                    latitude_weights=latitude_weights,
                    variable_weights=variable_weights,
                )
            if is_main_process():
                write_metrics(
                    output_dir / "metrics.jsonl",
                    {"epoch": 0, "split": "initial_val", **initial_val_metrics},
                )
                print(
                    f"[initial val] loss={initial_val_metrics['loss']:.6f} "
                    f"recon={initial_val_metrics['recon_loss']:.6f} "
                    f"weighted_recon={initial_val_metrics['weighted_recon_loss']:.6f} "
                    f"{format_validation_rmse(initial_val_metrics)}"
                )
            barrier()

        for epoch in range(start_epoch, config["train"]["epochs"] + 1):
            train_sampler = getattr(train_loader, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            train_metrics, global_step = train_one_epoch(
                model=model,
                ema_source_model=raw_model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                train_cfg=config["train"],
                scaler=scaler,
                ema=ema,
                latitude_weights=latitude_weights,
                variable_weights=variable_weights,
                epoch=epoch,
                global_step=global_step,
                log_progress=is_main_process(),
            )

            if is_main_process():
                write_metrics(
                    output_dir / "metrics.jsonl",
                    {"epoch": epoch, "split": "train", **train_metrics},
                )
                print(
                    f"[ae] epoch={epoch} "
                    f"loss={train_metrics['loss']:.6f} "
                    f"recon={train_metrics['recon_loss']:.6f} "
                    f"weighted_recon={train_metrics['weighted_recon_loss']:.6f} "
                    f"latent_noise={train_metrics['latent_noise_std']:.3e} "
                    f"grad={train_metrics['spatial_grad_loss']:.6f} "
                    f"lap={train_metrics['laplacian_loss']:.6f} "
                    f"dt={train_metrics['temporal_diff_loss']:.6f}"
                )

            barrier()
            val_metrics = None
            should_validate = (
                val_loader is not None
                and epoch % config["train"]["validate_every_epochs"] == 0
            )
            if should_validate:
                context = ema.average_parameters(raw_model) if ema is not None else torch.no_grad()
                with context:
                    val_metrics = evaluate(
                        model=raw_model,
                        dataloader=val_loader,
                        device=device,
                        datamodule=datamodule,
                        train_cfg=config["train"],
                        latitude_weights=latitude_weights,
                        variable_weights=variable_weights,
                    )

                if is_main_process():
                    write_metrics(
                        output_dir / "metrics.jsonl",
                        {"epoch": epoch, "split": "val", **val_metrics},
                    )
                    print(
                        f"[val] epoch={epoch} "
                        f"loss={val_metrics['loss']:.6f} "
                        f"recon={val_metrics['recon_loss']:.6f} "
                        f"weighted_recon={val_metrics['weighted_recon_loss']:.6f} "
                        f"{format_validation_rmse(val_metrics)}"
                    )
                    if val_metrics["rmse"] < best_val_rmse:
                        best_val_rmse = val_metrics["rmse"]
                        best_epoch = epoch

            best_val_rmse = broadcast_scalar(best_val_rmse, device)
            barrier()

            if is_main_process():
                save_checkpoint(
                    checkpoint_path_for_epoch(checkpoint_dir, epoch),
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_rmse=best_val_rmse,
                    best_epoch=best_epoch,
                    config=config,
                    ema=ema,
                    val_metrics=val_metrics,
                )
            barrier()

            if config["train"]["max_steps"] is not None and global_step >= config["train"]["max_steps"]:
                break
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
