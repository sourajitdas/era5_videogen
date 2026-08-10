from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+),([^}]+)\}$")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _resolve_special_values(value):
    if isinstance(value, dict):
        return {key: _resolve_special_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_special_values(item) for item in value]
    if isinstance(value, str):
        value = value.replace("${hydra:runtime.cwd}", str(PROJECT_ROOT))
        match = ENV_PATTERN.fullmatch(value)
        if match:
            env_name, default = match.groups()
            return os.environ.get(env_name, default)
    return value


def resolve_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg.pop("_target_", None)
    dataset_cfg = _resolve_special_values(dataset_cfg)

    root = dataset_cfg.get("root")
    if isinstance(root, str) and root:
        dataset_cfg["root"] = str(_resolve_path(root))
    return dataset_cfg


class ERA5WindowDataset(Dataset):
    """ERA5 sliding-window dataset that returns only weather target windows."""

    def __init__(
        self,
        root: str,
        variables: list[str],
        split: str = "train",
        timesteps: int = 14,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if timesteps < 1:
            raise ValueError("timesteps must be at least 1")

        self.root = root
        self.split = split
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {os.path.join(root, split)}")
        if len(self.files) < timesteps:
            raise ValueError(
                f"Not enough files for timesteps={timesteps} in split '{split}': "
                f"found {len(self.files)} files under {os.path.join(root, split)}"
            )

        self.variables = list(variables)
        self.timesteps = timesteps

        # Sliding windows overlap by timesteps-1 files, so a small per-worker
        # cache lets consecutive windows reuse already-read frames instead of
        # re-opening the same .h5 file ~timesteps times. FIFO eviction matches
        # the sequential scan order. Tunable/disable-able via env (0 disables).
        self._file_cache_size = int(os.environ.get("ERA5_FILE_CACHE_SIZE", "64"))
        self._file_cache: dict[str, np.ndarray] = {}
        self._file_cache_order: list[str] = []

        means = np.load(os.path.join(root, "normalize_mean.npz"))
        stds = np.load(os.path.join(root, "normalize_std.npz"))
        self.means = np.stack([means[name] for name in self.variables], axis=0).reshape(
            -1,
            1,
            1,
        )
        self.stds = np.stack([stds[name] for name in self.variables], axis=0).reshape(
            -1,
            1,
            1,
        )

        with h5py.File(self.files[0], "r") as handle:
            self.shape = (len(self.variables), *handle["input"][self.variables[0]].shape)

    @property
    def n_channels(self) -> int:
        return len(self.variables)

    @property
    def out_channels(self) -> int:
        return len(self.variables)

    @property
    def num_frames(self) -> int:
        return self.timesteps

    @property
    def img_resolution(self) -> tuple[int, int]:
        return self.shape[1], self.shape[2]

    @property
    def sample_shape(self) -> tuple[int, int, int, int]:
        return len(self.variables), self.timesteps, *self.img_resolution

    def get_lat_lon(self) -> tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as handle:
            timestamp = handle["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        sample = np.empty((len(variables), *self.img_resolution), dtype=np.float32)
        with h5py.File(path, "r") as handle:
            group = handle["input"]
            for idx, variable_name in enumerate(variables):
                group[variable_name].read_direct(sample[idx], np.s_[:], np.s_[:])
        return sample

    def _load_file_cached(self, path: str) -> np.ndarray:
        """Return the raw (unstandardized) frame for ``path``, reusing a cached
        copy when available. Callers must not mutate the returned array in place."""
        if self._file_cache_size <= 0:
            return self._load_file(path, self.variables)
        cached = self._file_cache.get(path)
        if cached is not None:
            return cached
        sample = self._load_file(path, self.variables)
        self._file_cache[path] = sample
        self._file_cache_order.append(path)
        if len(self._file_cache_order) > self._file_cache_size:
            evicted = self._file_cache_order.pop(0)
            if evicted != path:
                self._file_cache.pop(evicted, None)
        return sample

    def _standardize(self, sample: np.ndarray) -> np.ndarray:
        np.subtract(sample, self.means, out=sample)
        np.divide(sample, self.stds, out=sample)
        return sample

    def _unstandardize(self, sample: np.ndarray) -> np.ndarray:
        return sample * self.stds[:, None, :, :] + self.means[:, None, :, :]

    def denormalize(self, sample: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        if sample.ndim == 4:
            return self._unstandardize(sample)
        if sample.ndim == 5:
            return sample * self.stds[None, :, None, :, :] + self.means[None, :, None, :, :]
        raise ValueError("Expected shape (C, T, H, W) or (B, C, T, H, W).")

    def __len__(self) -> int:
        return max(0, len(self.files) - self.timesteps + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        target = np.empty(
            (len(self.variables), self.timesteps, *self.img_resolution),
            dtype=np.float32,
        )
        for step in range(self.timesteps):
            frame = self._load_file_cached(self.files[idx + step])
            # Standardize into the target slice so the cached frame is never
            # mutated (equivalent to _standardize: subtract mean, divide std).
            dst = target[:, step]
            np.subtract(frame, self.means, out=dst)
            np.divide(dst, self.stds, out=dst)
        return torch.from_numpy(target)


class ERA5VideoDataModule:
    def __init__(
        self,
        dataset: dict[str, Any] | None = None,
        batch_size: int = 1,
        val_batch_size: int | None = None,
        val_num_workers: int | None = None,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ):
        self.dataset = dict(dataset or {})
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.val_num_workers = val_num_workers
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

        self.dataset_kwargs = resolve_dataset_config(self.dataset)
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self) -> None:
        if self.data_train is not None:
            return

        self.data_train = ERA5WindowDataset(split="train", **self.dataset_kwargs)
        root = Path(self.dataset_kwargs["root"])
        if (root / "val").exists():
            self.data_val = ERA5WindowDataset(split="val", **self.dataset_kwargs)
        if (root / "test").exists():
            self.data_test = ERA5WindowDataset(split="test", **self.dataset_kwargs)

    def _make_sampler(
        self,
        dataset,
        distributed: bool,
        rank: int,
        world_size: int,
        shuffle: bool,
        drop_last: bool,
    ):
        if not distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def _loader_kwargs(self, num_workers_override: int | None = None) -> dict[str, Any]:
        num_workers = self.num_workers if num_workers_override is None else num_workers_override
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers if num_workers > 0 else False,
        }
        prefetch_factor = self.prefetch_factor if num_workers > 0 else None
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        return kwargs

    def train_dataloader(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        sampler = self._make_sampler(
            self.data_train,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            drop_last=self.drop_last,
        )
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=self.drop_last,
            **self._loader_kwargs(),
        )

    def val_dataloader(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int | None = None,
    ) -> DataLoader | None:
        if self.data_val is None:
            return None
        sampler = self._make_sampler(
            self.data_val,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            drop_last=False,
        )
        if num_workers is None:
            num_workers = self.val_num_workers
        return DataLoader(
            self.data_val,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            **self._loader_kwargs(num_workers_override=num_workers),
        )

    def test_dataloader(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int | None = None,
    ) -> DataLoader | None:
        if self.data_test is None:
            return None
        sampler = self._make_sampler(
            self.data_test,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            drop_last=False,
        )
        if num_workers is None:
            num_workers = self.val_num_workers
        return DataLoader(
            self.data_test,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            **self._loader_kwargs(num_workers_override=num_workers),
        )

    def get_lat_lon(self):
        if self.data_train is None:
            self.setup()
        return self.data_train.get_lat_lon()

    def denormalize(self, sample):
        if self.data_train is None:
            self.setup()
        return self.data_train.denormalize(sample)

    @property
    def variables(self) -> list[str]:
        if self.data_train is None:
            self.setup()
        return list(self.data_train.variables)

    @property
    def n_channels(self) -> int:
        if self.data_train is None:
            self.setup()
        return self.data_train.n_channels

    @property
    def img_resolution(self) -> tuple[int, int]:
        if self.data_train is None:
            self.setup()
        return self.data_train.img_resolution

    @property
    def num_frames(self) -> int:
        if self.data_train is None:
            self.setup()
        return self.data_train.num_frames
