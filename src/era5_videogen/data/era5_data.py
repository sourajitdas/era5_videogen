import os
from glob import glob
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ERA5UnconditionalDataset(Dataset):
    """Simple ERA5 dataset with optional static conditioning variables.

    Targets are returned as ``(C, T, H, W)``.
    If ``condition_variables`` are provided, each item is returned as
    ``(condition, target)`` where condition is ``(C_cond, H, W)``.
    """

    def __init__(
        self,
        root: str,
        variables: list[str],
        split: str = "train",
        timesteps: int = 14,
        condition_variables: list[str] | None = None,
        condition_root: str | None = None,
        normalize_condition_variables: bool = True,
    ):
        super().__init__()
        if split not in {"train", "test", "val"}:
            raise ValueError("split must be 'train', 'test', or 'val'")
        if timesteps < 1:
            raise ValueError("timesteps must be at least 1")

        self.root = root
        self.split = split
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        if not self.files:
            raise FileNotFoundError(
                f"No .h5 files found in {os.path.join(root, split)}"
            )

        self.variables = list(variables)
        self.timesteps = timesteps
        self.condition_variables = list(condition_variables or [])
        self.condition_root = condition_root or root
        self.normalize_condition_variables = normalize_condition_variables

        means = np.load(os.path.join(root, "normalize_mean.npz"))
        stds = np.load(os.path.join(root, "normalize_std.npz"))
        self.means = np.stack([means[v] for v in self.variables], axis=0).reshape(
            -1, 1, 1
        )
        self.stds = np.stack([stds[v] for v in self.variables], axis=0).reshape(
            -1, 1, 1
        )

        with h5py.File(self.files[0], "r") as f:
            self.shape = (len(self.variables), *f["input"][self.variables[0]].shape)
        self._condition_array = self._load_condition_array()

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
    def cond_channels(self) -> int:
        if self._condition_array is None:
            return 0
        return self._condition_array.shape[0]

    @property
    def condition_channels(self) -> int:
        return self.cond_channels

    @property
    def img_resolution(self) -> Tuple[int, int]:
        return self.shape[1], self.shape[2]

    @property
    def sample_shape(self) -> Tuple[int, int, int, int]:
        return len(self.variables), self.timesteps, *self.img_resolution

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        x = np.empty((len(variables), *self.img_resolution), dtype=np.float32)
        with h5py.File(path, "r") as f:
            group = f["input"]
            for idx, var_name in enumerate(variables):
                group[var_name].read_direct(x[idx], np.s_[:], np.s_[:])
        return x

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        np.subtract(x, self.means, out=x)
        np.divide(x, self.stds, out=x)
        return x

    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        return x * self.stds[:, None, :, :] + self.means[:, None, :, :]

    def denormalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4:
            return self._unstandardize(x)
        if x.ndim == 5:
            return x * self.stds[None, :, None, :, :] + self.means[None, :, None, :, :]
        raise ValueError("Expected target with shape (C, T, H, W) or (B, C, T, H, W).")

    def get_condition(self) -> torch.Tensor | None:
        if self._condition_array is None:
            return None
        return torch.from_numpy(self._condition_array.copy())

    def _candidate_condition_paths(self, var_name: str) -> list[str]:
        return [
            os.path.join(self.condition_root, f"{var_name}.npy"),
            os.path.join(self.condition_root, "static", f"{var_name}.npy"),
            os.path.join(self.condition_root, "constants", f"{var_name}.npy"),
            os.path.join(self.root, f"{var_name}.npy"),
            os.path.join(self.root, "static", f"{var_name}.npy"),
            os.path.join(self.root, "constants", f"{var_name}.npy"),
        ]

    def _load_condition_variable(self, var_name: str) -> np.ndarray:
        for path in self._candidate_condition_paths(var_name):
            if os.path.exists(path):
                return np.load(path).astype(np.float32)

        with h5py.File(self.files[0], "r") as f:
            if "input" in f and var_name in f["input"]:
                return np.array(f["input"][var_name], dtype=np.float32)
            if var_name in f:
                return np.array(f[var_name], dtype=np.float32)

        raise FileNotFoundError(
            f"Could not find conditional variable '{var_name}' in .npy files or the first HDF5 sample."
        )

    def _load_condition_array(self) -> np.ndarray | None:
        if not self.condition_variables:
            return None

        h, w = self.img_resolution
        condition_arrays = []
        for var_name in self.condition_variables:
            arr = self._load_condition_variable(var_name)

            if arr.ndim == 2:
                arr = arr[None, :, :]
            elif arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0][None, :, :]
            else:
                raise ValueError(
                    f"Unsupported conditional variable shape {arr.shape} for '{var_name}'."
                )

            if arr.shape[-2:] != (h, w):
                raise ValueError(
                    f"Conditional variable '{var_name}' has shape {arr.shape}, expected (*, {h}, {w})."
                )

            condition_arrays.append(arr.astype(np.float32))

        condition = np.concatenate(condition_arrays, axis=0)

        if self.normalize_condition_variables:
            means = np.load(os.path.join(self.root, "normalize_mean.npz"))
            stds = np.load(os.path.join(self.root, "normalize_std.npz"))
            cond_means = np.stack(
                [means[v] for v in self.condition_variables], axis=0
            ).reshape(-1, 1, 1)
            cond_stds = np.stack(
                [stds[v] for v in self.condition_variables], axis=0
            ).reshape(-1, 1, 1)
            condition = (condition - cond_means) / cond_stds

        return condition

    def __len__(self) -> int:
        return max(0, len(self.files) - self.timesteps + 1)

    def __getitem__(self, idx: int):
        target = np.empty(
            (len(self.variables), self.timesteps, *self.img_resolution),
            dtype=np.float32,
        )
        for step in range(self.timesteps):
            frame = self._load_file(self.files[idx + step], self.variables)
            self._standardize(frame)
            target[:, step] = frame

        target = torch.from_numpy(target)

        if self._condition_array is None:
            return target

        condition = torch.from_numpy(self._condition_array)
        return condition, target

if __name__ == "__main__":
    import time
    from omegaconf import OmegaConf

    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parents[1]
    config_path = package_root / "configs" / "data" / "era5_uncond.yaml"
    cfg = OmegaConf.load(config_path)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=False)
    dataset_cfg.pop("_target_", None)

    root = dataset_cfg.get("root")
    if isinstance(root, str) and "${hydra:runtime.cwd}" in root:
        dataset_cfg["root"] = root.replace("${hydra:runtime.cwd}", str(repo_root))

    condition_root = dataset_cfg.get("condition_root")
    if isinstance(condition_root, str) and "${hydra:runtime.cwd}" in condition_root:
        dataset_cfg["condition_root"] = condition_root.replace(
            "${hydra:runtime.cwd}", str(repo_root)
        )

    dataset = ERA5UnconditionalDataset(**dataset_cfg)

    print("config:", config_path)
    print("sample_shape:", dataset.sample_shape)
    print("model_channels:", dataset.n_channels)
    print("num_frames:", dataset.num_frames)
    print("condition_channels:", dataset.condition_channels)
    print("img_resolution:", dataset.img_resolution)

    batch_size = 1
    num_workers = 0
    num_batches = 3

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    times = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        start_time = time.time()
        if isinstance(batch, (tuple, list)):
            condition, target = batch
            condition = condition.to(torch.float32)
            target = target.to(torch.float32)
            print(f"batch {batch_idx + 1} condition:", condition.shape)
            print(f"batch {batch_idx + 1} target:", target.shape)
        else:
            target = batch.to(torch.float32)
            print(f"batch {batch_idx + 1} target:", target.shape)

        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Batch {batch_idx + 1}: {times[-1]:.4f} seconds")

    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage time per batch ({len(times)} batches): {avg_time:.4f} seconds")
