from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from .data import ERA5VideoDataModule
    from .model import Autoencoder3D
    from . import train as train_lib
except ImportError:
    if __package__:
        raise
    from data import ERA5VideoDataModule
    from model import Autoencoder3D
    import train as train_lib


# Mirror era5_vae/save_latents.py but for the *deterministic* Autoencoder3D:
# each window encodes to a single latent tensor (no posterior mean/logvar). The
# default output base sits next to the VAE latents, in an ``era5_ae`` folder.
DEFAULT_OUTPUT_BASE_DIR = Path(
    os.environ.get("ERA5_LATENT_ROOT", Path.cwd() / "latents")
)
LATENT_FORMAT_VERSION = "era5_ae_latents_v1"


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indexes: Iterable[int]):
        self.dataset = dataset
        self.indexes = [int(index) for index in indexes]

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int):
        index = self.indexes[item]
        return index, self.dataset[index]


class SplitTimeCache:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache: dict[int, str] = {}

    def get(self, index: int) -> str:
        if index not in self.cache:
            self.cache[index] = str(self.dataset.get_time(index))
        return self.cache[index]

    def window(self, start_index: int) -> list[str]:
        return [
            self.get(start_index + offset)
            for offset in range(self.dataset.timesteps)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode ERA5 windows with a trained Autoencoder3D and save the "
            "deterministic latents in split HDF5 folders."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config YAML used for data/model settings and latest checkpoint lookup.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt to load (default: latest epoch_*.pt under train.output_dir/checkpoints).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        choices=("auto", "file", "checkpoint"),
        default="auto",
        help="Which model config to build from (default: auto -> checkpoint if available).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Exact folder for latent files. Overrides --base-dir and --name.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE_DIR),
        help=(
            "Base directory for latent caches when --output-root is omitted. "
            f"Defaults to {DEFAULT_OUTPUT_BASE_DIR}."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Directory name inside --base-dir. Defaults to the config filename stem.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Dataset splits to encode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example 'cpu', 'cuda', or 'auto'. Supports torchrun.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("config", "fp32", "bf16", "fp16"),
        default="config",
        help="Autocast precision for encoding. 'config' uses train.precision.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        choices=("float32", "float16"),
        default="float32",
        help="Datatype used for the saved latent array.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoding batch size. Defaults to data.batch_size from the config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Defaults to data.num_workers from the config.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor when workers are enabled.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Pin DataLoader memory for faster host-to-GPU transfer.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned DataLoader memory.",
    )
    parser.set_defaults(persistent_workers=None)
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between batches.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First sliding-window index to encode within each split.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end sliding-window index within each split.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples per split after start/end slicing.",
    )
    parser.set_defaults(use_ema=False)
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action="store_true",
        help="Use EMA weights when the checkpoint has them (default: raw weights).",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Use raw checkpoint weights (default).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=("gzip", "lzf", "none"),
        default="none",
        help="HDF5 compression for latent datasets.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level when --compression=gzip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate latent files that already exist.",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=1,
        help="Background HDF5 writer threads used to overlap encoding and disk writes.",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=None,
        help="Maximum queued background file writes. Defaults to 4 * --write-workers.",
    )
    parser.set_defaults(save_times=False)
    parser.add_argument(
        "--save-times",
        dest="save_times",
        action="store_true",
        help="Save source ERA5 timestamps for each latent window.",
    )
    parser.add_argument(
        "--no-times",
        dest="save_times",
        action="store_false",
        help="Do not save source ERA5 timestamps (default).",
    )
    parser.set_defaults(save_model_bundle=True)
    parser.add_argument(
        "--save-model-bundle",
        dest="save_model_bundle",
        action="store_true",
        help="Save selected AE weights, model kwargs, and config (default).",
    )
    parser.add_argument(
        "--no-model-bundle",
        dest="save_model_bundle",
        action="store_false",
        help="Skip writing the AE model bundle.",
    )
    parser.add_argument(
        "--model-bundle-name",
        type=str,
        default="ae_model.pt",
        help="Filename for the saved AE model bundle.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print progress every N local batches per split on rank 0 when tqdm is disabled.",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show tqdm progress bars for completed latent files (default).",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(config: dict[str, Any], checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    checkpoint_dir = Path(config["train"]["output_dir"]) / "checkpoints"
    checkpoint_path = train_lib.find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not find any epoch_*.pt checkpoints under {checkpoint_dir}. "
            "Pass --checkpoint explicitly."
        )
    return checkpoint_path


def select_model_config(
    checkpoint: Any,
    config: dict[str, Any],
    model_config: str,
) -> dict[str, Any]:
    ckpt_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if model_config == "file":
        return config["model"]
    if model_config == "checkpoint":
        if not ckpt_config or "model" not in ckpt_config:
            raise KeyError("Checkpoint has no embedded model config; use --model-config file.")
        return ckpt_config["model"]
    # auto: prefer the checkpoint config (guaranteed to match the saved weights)
    if ckpt_config and "model" in ckpt_config:
        return ckpt_config["model"]
    return config["model"]


def build_model(
    checkpoint: Any,
    config: dict[str, Any],
    datamodule: ERA5VideoDataModule,
    model_config: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_source = select_model_config(checkpoint, config, model_config)
    model_cfg = train_lib.prepare_model_config({"model": dict(model_source)}, datamodule)
    train_lib.validate_latent_geometry(model_cfg, datamodule)
    model_cfg.pop("expected_latent_shape", None)
    model = train_lib.build_autoencoder(model_cfg, datamodule)
    return model, model_cfg


def load_weights_into_model(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    use_ema: bool,
    checkpoint_path: Path,
) -> str:
    weights = checkpoint.get("weights", checkpoint.get("model"))
    if weights is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no model weights.")
    if use_ema:
        ema = checkpoint.get("ema")
        if ema and ema.get("shadow"):
            shadow = train_lib.normalize_state_dict_keys(ema["shadow"])
            state = model.state_dict()
            kept = [name for name in state if name not in shadow]
            state.update({name: tensor for name, tensor in shadow.items() if name in state})
            model.load_state_dict(state)
            return "ema" if not kept else f"ema(+{len(kept)} raw tensors)"
        if train_lib.is_main_process():
            print("[note] --use-ema set but checkpoint has no EMA weights; using raw weights.")
    model.load_state_dict(train_lib.normalize_state_dict_keys(weights))
    return "raw"


def resolve_output_root(
    output_root_arg: str | None,
    base_dir_arg: str,
    name_arg: str | None,
    config_arg: str,
) -> Path:
    if output_root_arg is not None:
        return Path(output_root_arg)
    run_name = name_arg if name_arg is not None else Path(config_arg).stem
    return Path(base_dir_arg) / run_name


def resolve_precision(args_precision: str, config: dict[str, Any], device: torch.device) -> str:
    precision = (
        str(config.get("train", {}).get("precision", "fp32")).lower()
        if args_precision == "config"
        else args_precision
    )
    if device.type != "cuda":
        return "fp32"
    if precision == "bf16":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            if train_lib.is_main_process():
                print(
                    "[note] bf16 latent encoding is only enabled on A100/H100-class GPUs. "
                    f"Detected {torch.cuda.get_device_name(device)}; falling back to fp32."
                )
            return "fp32"
    return precision


def apply_data_overrides(data_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    data_cfg = dict(data_cfg)
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
        data_cfg["val_batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
        data_cfg["val_num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        data_cfg["prefetch_factor"] = args.prefetch_factor
    if args.pin_memory is not None:
        data_cfg["pin_memory"] = args.pin_memory
    if args.persistent_workers is not None:
        data_cfg["persistent_workers"] = args.persistent_workers
    data_cfg["drop_last"] = False
    return data_cfg


def get_dataset_for_split(datamodule: ERA5VideoDataModule, split: str):
    if split == "train":
        return datamodule.data_train
    if split == "val":
        return datamodule.data_val
    if split == "test":
        return datamodule.data_test
    raise ValueError(f"Unsupported split: {split}")


def source_filename_for_index(dataset, index: int) -> str:
    return Path(dataset.files[index]).name


def latent_file_path(output_root: Path, split: str, dataset, index: int) -> Path:
    return output_root / split / source_filename_for_index(dataset, index)


def try_read_completed_source_index(latent_path: Path) -> tuple[int | None, bool]:
    try:
        with h5py.File(latent_path, "r") as handle:
            attr_index = handle.attrs.get("source_index")
            if attr_index is not None:
                return int(attr_index), True

            input_group = handle.get("input")
            if input_group is not None and "source_index" in input_group:
                return int(input_group["source_index"][()]), True
    except (OSError, KeyError, TypeError, ValueError):
        return None, False

    return None, True


def completed_indexes_for_split(
    *,
    output_root: Path,
    split: str,
    dataset,
    indexes: list[int],
) -> set[int]:
    split_dir = output_root / split
    if not split_dir.is_dir() or not indexes:
        return set()

    requested_indexes = {int(index) for index in indexes}
    fallback_name_to_index = {
        source_filename_for_index(dataset, index): int(index)
        for index in indexes
    }
    completed: set[int] = set()

    for latent_path in split_dir.glob("*.h5"):
        source_index, was_readable = try_read_completed_source_index(latent_path)
        if source_index is not None:
            if source_index in requested_indexes:
                completed.add(source_index)
            continue

        if was_readable:
            fallback_index = fallback_name_to_index.get(latent_path.name)
            if fallback_index is not None:
                completed.add(fallback_index)

    return completed


def selected_indexes_for_split(
    *,
    dataset_length: int,
    dataset,
    output_root: Path,
    split: str,
    start_index: int,
    end_index: int | None,
    limit_samples: int | None,
    overwrite: bool,
    rank: int,
    world_size: int,
) -> tuple[list[int], int, int | None]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")

    stop = dataset_length if end_index is None else min(end_index, dataset_length)
    indexes = list(range(min(start_index, dataset_length), stop))
    if limit_samples is not None:
        if limit_samples < 1:
            raise ValueError("--limit-samples must be at least 1 when provided.")
        indexes = indexes[:limit_samples]

    before_resume_filter = len(indexes)
    first_pending_index: int | None = indexes[0] if indexes else None
    if not overwrite:
        completed_indexes = completed_indexes_for_split(
            output_root=output_root,
            split=split,
            dataset=dataset,
            indexes=indexes,
        )
        indexes = [index for index in indexes if index not in completed_indexes]
        first_pending_index = indexes[0] if indexes else None

    skipped_existing = before_resume_filter - len(indexes)
    return (
        indexes[rank::world_size],
        skipped_existing if rank == 0 else 0,
        first_pending_index if rank == 0 else None,
    )


def make_dataloader(
    dataset,
    datamodule: ERA5VideoDataModule,
    split: str,
    indexes: list[int],
) -> DataLoader:
    batch_size = datamodule.batch_size if split == "train" else datamodule.val_batch_size
    num_workers = datamodule.num_workers if split == "train" else datamodule.val_num_workers
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        **datamodule._loader_kwargs(num_workers_override=num_workers),
    }
    return DataLoader(IndexedDataset(dataset, indexes), **kwargs)


def hdf5_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {
        "compression": "gzip",
        "compression_opts": int(args.compression_level),
        "shuffle": True,
    }


def cast_latent(array: np.ndarray, dtype_name: str) -> np.ndarray:
    dtype = np.float16 if dtype_name == "float16" else np.float32
    return array.astype(dtype, copy=False)


def source_files_for_window(dataset, start_index: int) -> list[str]:
    return [
        Path(dataset.files[start_index + offset]).name
        for offset in range(dataset.timesteps)
    ]


def write_latent_file(
    *,
    output_root: Path,
    split: str,
    index: int,
    latent: np.ndarray,
    dataset,
    time_cache: SplitTimeCache | None,
    args: argparse.Namespace,
    source_root: str,
    rank: int,
) -> None:
    output_path = latent_file_path(output_root, split, dataset, index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.rank{rank}.tmp")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    dataset_kwargs = hdf5_dataset_kwargs(args)

    with h5py.File(tmp_path, "w") as handle:
        handle.attrs["format"] = LATENT_FORMAT_VERSION
        handle.attrs["split"] = split
        handle.attrs["source_root"] = source_root
        handle.attrs["source_index"] = int(index)
        handle.attrs["source_file"] = source_filename_for_index(dataset, index)
        handle.attrs["source_timesteps"] = int(dataset.timesteps)
        handle.attrs["save_dtype"] = args.save_dtype

        input_group = handle.create_group("input")
        input_group.create_dataset(
            "latent",
            data=cast_latent(latent, args.save_dtype),
            **dataset_kwargs,
        )
        input_group.create_dataset("source_index", data=np.int64(index))
        input_group.create_dataset(
            "source_files",
            data=np.asarray(source_files_for_window(dataset, index), dtype=object),
            dtype=string_dtype,
        )
        if time_cache is not None:
            input_group.create_dataset(
                "time",
                data=np.asarray(time_cache.window(index), dtype=object),
                dtype=string_dtype,
            )

    tmp_path.replace(output_path)


def complete_write_futures(
    pending_writes: set[Future],
    progress_bar,
    *,
    block: bool,
) -> int:
    if not pending_writes:
        return 0

    if block:
        done, _ = wait(pending_writes)
    else:
        done = {future for future in pending_writes if future.done()}

    for future in done:
        future.result()
    pending_writes.difference_update(done)
    progress_bar.update(len(done))
    return len(done)


def wait_for_write_capacity(
    pending_writes: set[Future],
    progress_bar,
    max_pending_writes: int,
) -> int:
    completed = 0
    while len(pending_writes) >= max_pending_writes:
        done, _ = wait(pending_writes, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
        pending_writes.difference_update(done)
        progress_bar.update(len(done))
        completed += len(done)
    return completed


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def checkpoint_summary(checkpoint: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "epoch",
        "global_step",
        "best_val_rmse",
        "best_epoch",
        "val_metrics",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def save_model_artifacts(
    *,
    output_root: Path,
    model: torch.nn.Module,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    args: argparse.Namespace,
    data_cfg: dict[str, Any],
    precision: str,
    weights_source: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    impl = str(model_cfg.get("impl", "default"))

    with (output_root / "ae_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    run_metadata = {
        "format": LATENT_FORMAT_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_summary": checkpoint_summary(checkpoint),
        "use_ema": bool(args.use_ema),
        "weights_source": weights_source,
        "splits": list(args.splits),
        "save_dtype": args.save_dtype,
        "precision": precision,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "model_impl": impl,
        "data": data_cfg,
        "model_config": model_cfg,
        "output_layout": {
            "split_folder": "<output_root>/<split>",
            "sample_file": "<source_initial_era5_filename>.h5",
            "latent_dataset": "input/latent",
        },
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, default=str)
        handle.write("\n")

    if args.save_model_bundle:
        bundle = {
            "format": LATENT_FORMAT_VERSION,
            "model_class": f"model.Autoencoder3D (impl={impl})",
            "model_state_dict": cpu_state_dict(model),
            "model_config": model_cfg,
            "hyper_parameters": model_cfg,
            "source_config": config,
            "checkpoint": str(checkpoint_path),
            "checkpoint_summary": checkpoint_summary(checkpoint),
            "use_ema": bool(args.use_ema),
            "weights_source": weights_source,
        }
        encoder = getattr(model, "encoder", None)
        decoder = getattr(model, "decoder", None)
        if encoder is not None:
            bundle["encoder_state_dict"] = cpu_state_dict(encoder)
        if decoder is not None:
            bundle["decoder_state_dict"] = cpu_state_dict(decoder)
        torch.save(bundle, output_root / args.model_bundle_name)


def reduce_int(value: int, device: torch.device) -> int:
    if not train_lib.is_distributed():
        return int(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def encode_split(
    *,
    split: str,
    dataset,
    datamodule: ERA5VideoDataModule,
    model: torch.nn.Module,
    output_root: Path,
    device: torch.device,
    precision: str,
    source_root: str,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> dict[str, int]:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    indexes, skipped_existing, first_pending_index = selected_indexes_for_split(
        dataset_length=len(dataset),
        dataset=dataset,
        output_root=output_root,
        split=split,
        start_index=args.start_index,
        end_index=args.end_index,
        limit_samples=args.limit_samples,
        overwrite=args.overwrite,
        rank=rank,
        world_size=world_size,
    )
    dataloader = make_dataloader(
        dataset=dataset,
        datamodule=datamodule,
        split=split,
        indexes=indexes,
    )
    time_cache = SplitTimeCache(dataset) if args.save_times else None
    written = 0

    if train_lib.is_main_process():
        print(
            f"[info] split={split} samples={len(dataset):,} "
            f"local_to_encode={len(indexes):,}"
        )
        if skipped_existing > 0:
            if first_pending_index is None:
                print(
                    f"[resume] split={split} found {skipped_existing:,} existing latent files. "
                    "Nothing left to encode."
                )
            else:
                print(
                    f"[resume] split={split} found {skipped_existing:,} existing latent files. "
                    f"Continuing from source_index={first_pending_index}."
                )

    progress_desc = f"{split} files"
    if world_size > 1:
        progress_desc = f"{split} rank {rank} files"

    write_workers = max(1, int(args.write_workers))
    max_pending_writes = args.max_pending_writes
    if max_pending_writes is None:
        max_pending_writes = max(1, write_workers * 4)
    max_pending_writes = max(1, int(max_pending_writes))
    write_executor = (
        ThreadPoolExecutor(max_workers=write_workers)
        if write_workers > 1
        else None
    )
    pending_writes: set[Future] = set()

    try:
        with tqdm(
            total=len(indexes),
            desc=progress_desc,
            unit="file",
            dynamic_ncols=True,
            position=rank if world_size > 1 else 0,
            disable=not args.progress,
        ) as progress_bar:
            with torch.inference_mode():
                for batch_idx, (batch_indexes, batch) in enumerate(dataloader, start=1):
                    batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
                    with train_lib.autocast_context(device, precision):
                        latents = model.encode(batch)

                    latents = latents.float().cpu().numpy()
                    batch_indexes_np = batch_indexes.cpu().numpy()

                    for row, index in enumerate(batch_indexes_np):
                        write_kwargs = {
                            "output_root": output_root,
                            "split": split,
                            "index": int(index),
                            "latent": latents[row],
                            "dataset": dataset,
                            "time_cache": time_cache,
                            "args": args,
                            "source_root": source_root,
                            "rank": rank,
                        }
                        if write_executor is None:
                            write_latent_file(**write_kwargs)
                            written += 1
                            progress_bar.update(1)
                        else:
                            pending_writes.add(write_executor.submit(write_latent_file, **write_kwargs))
                            written += wait_for_write_capacity(
                                pending_writes,
                                progress_bar,
                                max_pending_writes,
                            )

                    if write_executor is not None:
                        written += complete_write_futures(
                            pending_writes,
                            progress_bar,
                            block=False,
                        )

                    if (
                        train_lib.is_main_process()
                        and not args.progress
                        and args.log_every > 0
                        and batch_idx % args.log_every == 0
                    ):
                        print(
                            f"[progress] split={split} local_batches={batch_idx:,}/"
                            f"{len(dataloader):,} local_written={written:,}"
                        )

            if write_executor is not None:
                written += complete_write_futures(
                    pending_writes,
                    progress_bar,
                    block=True,
                )
    finally:
        if write_executor is not None:
            write_executor.shutdown(wait=True)

    return {
        "dataset_samples": len(dataset),
        "local_requested": len(indexes),
        "local_written": written,
        "local_skipped_existing": skipped_existing,
    }


def write_manifest(
    *,
    output_root: Path,
    split_stats: dict[str, dict[str, int]],
    args: argparse.Namespace,
    checkpoint_path: Path,
    precision: str,
    world_size: int,
) -> None:
    manifest = {
        "format": LATENT_FORMAT_VERSION,
        "checkpoint": str(checkpoint_path),
        "use_ema": bool(args.use_ema),
        "precision": precision,
        "save_dtype": args.save_dtype,
        "world_size": world_size,
        "splits": split_stats,
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    distributed_info = train_lib.init_distributed_mode(args.device or "auto")
    rank = int(distributed_info["rank"])
    world_size = int(distributed_info["world_size"])
    device = distributed_info["device"]

    try:
        config = train_lib.load_config(args.config)
        checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
        checkpoint = train_lib.torch_load_checkpoint(checkpoint_path, map_location="cpu")
        data_cfg = apply_data_overrides(config["data"], args)
        precision = resolve_precision(args.precision, config, device)

        train_lib.configure_cuda_runtime(config.get("train", {}), device)
        datamodule = ERA5VideoDataModule(**data_cfg)
        datamodule.setup()

        model, model_cfg = build_model(checkpoint, config, datamodule, args.model_config)
        weights_source = load_weights_into_model(
            model,
            checkpoint,
            use_ema=args.use_ema,
            checkpoint_path=checkpoint_path,
        )

        output_root = resolve_output_root(
            args.output_root,
            args.base_dir,
            args.name,
            args.config,
        )
        if train_lib.is_main_process():
            save_model_artifacts(
                output_root=output_root,
                model=model,
                model_cfg=model_cfg,
                config=config,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                args=args,
                data_cfg=data_cfg,
                precision=precision,
                weights_source=weights_source,
            )
            print(f"[info] writing latents to {output_root}")
            print(
                f"[info] checkpoint={checkpoint_path} "
                f"weights={weights_source} "
                f"device={device} precision={precision}"
            )
        train_lib.barrier()

        model = model.to(device)
        model.eval()

        source_root = str(datamodule.dataset_kwargs["root"])
        local_split_stats: dict[str, dict[str, int]] = {}
        for split in args.splits:
            dataset = get_dataset_for_split(datamodule, split)
            if dataset is None:
                if train_lib.is_main_process():
                    print(f"[skip] split={split} is not available.")
                local_split_stats[split] = {
                    "dataset_samples": 0,
                    "local_requested": 0,
                    "local_written": 0,
                    "local_skipped_existing": 0,
                }
                continue

            local_split_stats[split] = encode_split(
                split=split,
                dataset=dataset,
                datamodule=datamodule,
                model=model,
                output_root=output_root,
                device=device,
                precision=precision,
                source_root=source_root,
                args=args,
                rank=rank,
                world_size=world_size,
            )

        global_split_stats: dict[str, dict[str, int]] = {}
        for split, stats in local_split_stats.items():
            global_split_stats[split] = {
                "dataset_samples": stats["dataset_samples"],
                "requested": reduce_int(stats["local_requested"], device),
                "written": reduce_int(stats["local_written"], device),
                "skipped_existing": reduce_int(stats["local_skipped_existing"], device),
            }
            completed_after_run = (
                global_split_stats[split]["written"]
                + global_split_stats[split]["skipped_existing"]
            )
            global_split_stats[split]["completed_after_run"] = min(
                global_split_stats[split]["dataset_samples"],
                completed_after_run,
            )
            global_split_stats[split]["remaining_after_run"] = max(
                0,
                global_split_stats[split]["dataset_samples"]
                - global_split_stats[split]["completed_after_run"],
            )

        train_lib.barrier()
        if train_lib.is_main_process():
            write_manifest(
                output_root=output_root,
                split_stats=global_split_stats,
                args=args,
                checkpoint_path=checkpoint_path,
                precision=precision,
                world_size=world_size,
            )
            print(f"[done] wrote latent manifest to {output_root / 'manifest.json'}")
    finally:
        train_lib.cleanup_distributed()


if __name__ == "__main__":
    main()
