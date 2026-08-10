import os
import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary

from era5_videogen.data.samplers import InfiniteSampler
from era5_videogen.training.trainer import Trainer
from era5_videogen.utils import io, stats

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)
# torch.utils.checkpoint (gradient checkpointing) still enters the deprecated
# torch.cpu.amp.autocast during recompute on torch<=2.4, emitting a FutureWarning
# from library code we do not control. Our own AMP uses torch.autocast(device_type=...)
# / torch.GradScaler, so silence only this specific internal message.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cpu\.amp\.autocast.*",
    category=FutureWarning,
)


def _init_distributed() -> None:
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.distributed.is_initialized() and world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _build_dataloader(dataset: Dataset, batch_size: int, **kwargs) -> DataLoader:
    data_workers = kwargs.pop("num_workers")
    persistent_workers = kwargs.pop("persistent_workers", data_workers > 0)
    prefetch_factor = kwargs.pop("prefetch_factor", 2)
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=data_workers,
        persistent_workers=(persistent_workers if data_workers > 0 else False),
        **kwargs,
    )
    if data_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["worker_init_fn"] = _seed_worker
    return DataLoader(**dataloader_kwargs)


def _seed_worker(worker_id: int) -> None:
    del worker_id  # Worker-specific seed comes from torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_KNOWN_DATA_KEYS = {
    "dataset",
    "batch_size",
    "val_batch_size",
    "data_workers",
    "persistent_workers",
    "prefetch_factor",
}


def _validate_data_cfg(cfg: DictConfig) -> None:
    """Fail loudly on `data.*` keys that would silently do nothing.

    The dataset is built from ``cfg.data.dataset`` (line below), so dataset
    arguments such as ``latent_mode`` / ``latent_scale_factor`` must be nested
    there. Keys placed directly under ``cfg.data`` are otherwise ignored without
    any warning -- a mis-nesting that previously caused a run to train on the
    default ``latent_mode=sample`` latents instead of the intended ``mean``.
    """
    unknown = [k for k in cfg.data.keys() if k not in _KNOWN_DATA_KEYS]
    if not unknown:
        return
    dataset_keys = set(cfg.data.get("dataset", {}).keys())
    misnested = [k for k in unknown if k in dataset_keys]
    msg = (
        f"Unrecognised key(s) directly under `data`: {sorted(unknown)}. "
        f"Recognised top-level `data` keys are {sorted(_KNOWN_DATA_KEYS)}."
    )
    if misnested:
        msg += (
            f" {misnested} look like dataset arguments -- nest them under "
            "`data.dataset`, not `data`."
        )
    raise ValueError(msg)


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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    _init_distributed()
    stats.init_multiprocessing(
        rank=io.get_rank(),
        sync_device=torch.device("cuda") if io.get_world_size() > 1 else None,
    )
    io.log0(f"Rank {io.get_rank()}, World Size {io.get_world_size()}")
    if io.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed(
        (
            cfg.seed
            * (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            + (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        )
        % (1 << 31)
    )
    torch.manual_seed(np.random.randint(1 << 31))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.system.torch.benchmark
        torch.backends.cudnn.allow_tf32 = cfg.system.torch.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.system.torch.allow_tf32
        torch.set_float32_matmul_precision(
            cfg.system.torch.set_float32_matmul_precision
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            cfg.system.torch.amp_type == "float16"
        )

    _validate_data_cfg(cfg)
    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    effective_global_batch = cfg.data.batch_size * io.get_world_size() * accumulation_steps
    io.log0(f"Using local batch size: {cfg.data.batch_size} (per GPU)")
    io.log0(f"Gradient accumulation steps per GPU: {accumulation_steps}")
    io.log0(f"Effective global batch size: {effective_global_batch}")
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset,
        rank=io.get_rank(),
        num_replicas=io.get_world_size(),
        shuffle=True,
        seed=cfg.seed,
    )
    dataloader = iter(
        _build_dataloader(
            dataset,
            cfg.data.batch_size,
            sampler=dataset_sampler,
            num_workers=cfg.data.data_workers,
            persistent_workers=cfg.data.get("persistent_workers", True),
            prefetch_factor=cfg.data.get("prefetch_factor", 2),
        )
    )

    val_dataloader = None
    if cfg.trainer.get("validation_ticks") is not None and io.get_rank() == 0:
        io.log0("Loading validation dataset...")
        val_dataset_config = OmegaConf.create(
            OmegaConf.to_container(cfg.data.dataset, resolve=True)
        )
        val_dataset_config.split = "val"
        val_dataset: Dataset = instantiate(val_dataset_config, _convert_="object")
        io.log0(f"Validation dataset size: {len(val_dataset)}")

        val_batch_size = cfg.data.get("val_batch_size", cfg.data.batch_size)
        val_dataloader = _build_dataloader(
            val_dataset,
            val_batch_size,
            shuffle=False,
            num_workers=cfg.data.data_workers,
            persistent_workers=cfg.data.get("persistent_workers", True),
            prefetch_factor=cfg.data.get("prefetch_factor", 2),
        )

    io.log0("Constructing network...")
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
        net = instantiate(
            cfg.model,
            **model_kwargs,
        )

    net.train().requires_grad_(True).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if io.get_rank() == 0:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        io.log0(
            "Model parameters: "
            f"total={total_params:,} ({total_params / 1e6:.2f}M), "
            f"trainable={trainable_params:,} ({trainable_params / 1e6:.2f}M), "
            f"frozen={frozen_params:,} ({frozen_params / 1e6:.2f}M)"
        )
        summary(net, depth=3)

    io.log0("Constructing optimizer...")
    optimizer = instantiate(cfg.optimizer, net.parameters(), _convert_="object")
    io.log0("Constructing loss function...")
    loss_fn = instantiate(cfg.loss, _convert_="object")
    trainer_conditional = bool(cfg.trainer.get("conditional", False) or condition_channels > 0)
    trainer: Trainer = instantiate(
        cfg.trainer,
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_type=cfg.system.torch.amp_type,
        conditional=trainer_conditional,
    )

    io.log0("Training...")
    trainer.train(dataloader, val_dataloader=val_dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    """
    local (1 gpu 1 node): python -m era5_videogen.train
    """
    main()
