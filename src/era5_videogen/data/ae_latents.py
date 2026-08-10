import json
import os
from glob import glob
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


# Default filename for the per-channel normalization statistics, stored next to
# the latent splits (i.e. ``<root>/stats.yaml``). Produced by
# ``scripts/compute_ae_latent_stats.py``.
DEFAULT_STATS_FILENAME = "stats.yaml"


class ERA5AELatentDataset(Dataset):
    """ERA5 deterministic *autoencoder* latent dataset.

    Each HDF5 file holds a single latent tensor under ``input/latent`` with shape
    ``(C, T, H, W)`` (no posterior mean/logvar -- the encoder is deterministic).

    Unlike the VAE latents, the AE latents are not globally scaled at save time,
    so this dataset standardizes each latent **channel** to zero mean / unit
    variance using statistics in ``stats.yaml`` (see
    ``scripts/compute_ae_latent_stats.py``). Generation undoes this with
    :meth:`unnormalize_latents` before decoding.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        normalize: bool = True,
        stats_path: str | None = None,
        latent_key: str = "input/latent",
        source_root: str | None = None,
        ae_model_path: str | None = None,
    ):
        super().__init__()
        if split not in {"train", "test", "val"}:
            raise ValueError("split must be 'train', 'test', or 'val'")

        self.root = root
        self.split = split
        self.latent_key = latent_key
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        if not self.files:
            raise FileNotFoundError(
                f"No .h5 files found in {os.path.join(root, split)}"
            )

        self.normalize = bool(normalize)
        self.condition_variables = []
        # The AE is deterministic; this is NOT a VAE latent dataset. The flag is
        # read by generate.py to decide default decoding behaviour.
        self.is_vae_latent_dataset = False
        self.is_ae_latent_dataset = True
        self.is_latent_dataset = True

        self.metadata = self._load_metadata()
        self.source_root = source_root or self._infer_source_root()
        self.variables = self._infer_variables()
        self.ae_model_path = ae_model_path or self._infer_ae_model_path()

        with h5py.File(self.files[0], "r") as f:
            self.shape = tuple(int(v) for v in f[self.latent_key].shape)
            self.source_timesteps = int(f.attrs.get("source_timesteps", self.shape[1]))

        self.stats_path = stats_path or os.path.join(self.root, DEFAULT_STATS_FILENAME)
        self._channel_mean, self._channel_std = self._load_latent_stats()
        self._weather_means, self._weather_stds = self._load_weather_stats()

    # ------------------------------------------------------------------
    # Shape / geometry properties (mirror the other latent datasets so the
    # training and generation code can stay dataset-agnostic).
    # ------------------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return self.shape[0]

    @property
    def out_channels(self) -> int:
        return self.shape[0]

    @property
    def num_frames(self) -> int:
        return self.shape[1]

    @property
    def cond_channels(self) -> int:
        return 0

    @property
    def condition_channels(self) -> int:
        return self.cond_channels

    @property
    def img_resolution(self) -> Tuple[int, int]:
        return self.shape[2], self.shape[3]

    @property
    def sample_shape(self) -> Tuple[int, int, int, int]:
        return self.shape

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        root = self.source_root or self.root
        lat = np.load(os.path.join(root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64 | None:
        with h5py.File(self.files[idx], "r") as f:
            group = f["input"] if "input" in f else f
            # Prefer an explicit ISO timestamp if the latent file ever stores one.
            if "time" in group:
                value = group["time"]
                timestamp = value[0] if value.shape else value[()]
                if isinstance(timestamp, bytes):
                    timestamp = timestamp.decode("utf-8")
                try:
                    return np.datetime64(str(timestamp))
                except ValueError:
                    return None
            # Otherwise resolve the timestamp from the referenced source ERA5
            # file. ``source_files`` holds source *filenames* (e.g.
            # ``1979_0000.h5``), not datetimes, so the latent's initial time is
            # read from that file's ``input/time`` dataset.
            if "source_files" in group:
                value = group["source_files"]
                name = value[0] if value.shape else value[()]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                return self._source_time(str(name))
        return None

    def _source_time(self, source_name: str) -> np.datetime64 | None:
        """Read the ISO timestamp from a referenced source ERA5 file.

        Returns ``None`` (rather than raising) if the source root/file is not
        available or does not contain an ``input/time`` dataset.
        """
        if not self.source_root:
            return None
        name = os.path.basename(str(source_name))
        candidates = [
            os.path.join(self.source_root, self.split, name),
            os.path.join(self.source_root, name),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with h5py.File(path, "r") as sf:
                    sgroup = sf["input"] if "input" in sf else sf
                    if "time" not in sgroup:
                        return None
                    value = sgroup["time"]
                    timestamp = value[0] if value.shape else value[()]
                    if isinstance(timestamp, bytes):
                        timestamp = timestamp.decode("utf-8")
                    return np.datetime64(str(timestamp))
            except (OSError, ValueError):
                return None
        return None

    def get_condition(self) -> torch.Tensor | None:
        return None

    # ------------------------------------------------------------------
    # Per-channel latent normalization.
    # ------------------------------------------------------------------
    def _broadcast_stats(self, ndim: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._channel_mean is None or self._channel_std is None:
            raise RuntimeError(
                "Latent statistics are not loaded. Either set normalize=False or "
                f"create {self.stats_path} via scripts/compute_ae_latent_stats.py."
            )
        if ndim == 4:  # (C, T, H, W)
            shape = (-1, 1, 1, 1)
        elif ndim == 5:  # (B, C, T, H, W)
            shape = (1, -1, 1, 1, 1)
        else:
            raise ValueError(
                "Expected latent with shape (C, T, H, W) or (B, C, T, H, W)."
            )
        return self._channel_mean.reshape(shape), self._channel_std.reshape(shape)

    def normalize_latents(self, x):
        if not self.normalize:
            return x
        is_tensor = isinstance(x, torch.Tensor)
        mean, std = self._broadcast_stats(x.ndim)
        if is_tensor:
            mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
            std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device)
            return (x - mean_t) / std_t
        return (x - mean) / std

    def unnormalize_latents(self, x):
        if not self.normalize:
            return x
        is_tensor = isinstance(x, torch.Tensor)
        mean, std = self._broadcast_stats(x.ndim)
        if is_tensor:
            mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
            std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device)
            return x * std_t + mean_t
        return x * std + mean

    def denormalize(self, x) -> np.ndarray:
        """Undo latent standardization or denormalize decoded ERA5 fields.

        - Latent-shaped arrays (``C == n_channels``) are mapped back to the raw
          AE latent space via :meth:`unnormalize_latents`.
        - Decoded weather-shaped arrays (``C == len(variables)``) are mapped back
          to physical units using the source ERA5 mean/std files.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected array with shape (C, T, H, W) or (B, C, T, H, W)."
            )

        channel_axis = 0 if x.ndim == 4 else 1
        channels = x.shape[channel_axis]

        if channels == self.n_channels:
            return np.asarray(self.unnormalize_latents(x), dtype=np.float32)

        if self._weather_means is None or self._weather_stds is None:
            return x

        weather_channels = len(self.variables)
        if channels != weather_channels:
            return x

        if x.ndim == 4:
            return (
                x * self._weather_stds[:, None, :, :]
                + self._weather_means[:, None, :, :]
            )
        return (
            x * self._weather_stds[None, :, None, :, :]
            + self._weather_means[None, :, None, :, :]
        )

    # ------------------------------------------------------------------
    # AE decoder (for generation: latents -> normalized ERA5 fields).
    # ------------------------------------------------------------------
    def build_decoder(self, device: torch.device, ae_model_path: str | None = None):
        from era5_videogen.data.ae_model import load_ae_decoder

        path = self._resolve_ae_model_path(ae_model_path)
        return load_ae_decoder(path, device=device, source_root=self.source_root)

    def build_autoencoder(self, device: torch.device, ae_model_path: str | None = None):
        """Load the full AE (encoder + decoder), e.g. for posterior sampling."""
        from era5_videogen.data.ae_model import load_ae_model

        path = self._resolve_ae_model_path(ae_model_path)
        return load_ae_model(path, device=device, source_root=self.source_root)

    def _resolve_ae_model_path(self, ae_model_path: str | None = None) -> str:
        path = ae_model_path or self.ae_model_path
        if path is None:
            raise FileNotFoundError(
                "AE decoding requested, but no ae_model_path was provided and the "
                "dataset could not locate ae_model.pt under the latent root."
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"AE model bundle not found: {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Source ERA5 video (ground truth in standardized units), resolved via the
    # latent file's ``input/source_files`` references.
    # ------------------------------------------------------------------
    def get_source_video(self, idx: int, normalize: bool = True) -> torch.Tensor:
        """Load the source ERA5 video for latent ``idx`` as ``(C, T, H, W)``.

        ``C`` follows :attr:`variables` order and ``T`` is the number of source
        timesteps (the AE's temporal upsampling of :attr:`num_frames`). When
        ``normalize`` is set the fields are z-scored with the same per-variable
        statistics used for AE training, i.e. the units the decoder outputs.
        """
        if not self.source_root:
            raise RuntimeError(
                "Cannot load the source video: dataset has no source_root."
            )
        if not self.variables:
            raise RuntimeError(
                "Cannot load the source video: dataset metadata lists no variables."
            )

        with h5py.File(self.files[idx], "r") as f:
            group = f["input"] if "input" in f else f
            if "source_files" not in group:
                raise KeyError(
                    f"{self.files[idx]} does not record input/source_files."
                )
            names = [
                (n.decode("utf-8") if isinstance(n, bytes) else str(n))
                for n in np.atleast_1d(group["source_files"][...])
            ]

        frames = []
        for name in names:
            path = self._resolve_source_file(os.path.basename(name))
            with h5py.File(path, "r") as sf:
                sgroup = sf["input"] if "input" in sf else sf
                frame = np.stack(
                    [
                        np.squeeze(np.asarray(sgroup[v][...], dtype=np.float32))
                        for v in self.variables
                    ],
                    axis=0,
                )
            frames.append(frame)
        video = np.stack(frames, axis=1)  # (C, T, H, W)

        if normalize:
            if self._weather_means is None or self._weather_stds is None:
                raise RuntimeError(
                    "Weather normalization stats not found under source_root."
                )
            video = (video - self._weather_means[:, None]) / self._weather_stds[:, None]

        return torch.from_numpy(np.ascontiguousarray(video, dtype=np.float32))

    def _resolve_source_file(self, name: str) -> str:
        candidates = [
            os.path.join(self.source_root, self.split, name),
            os.path.join(self.source_root, name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Source ERA5 file '{name}' not found under {self.source_root} "
            f"(tried split '{self.split}' and root)."
        )

    # ------------------------------------------------------------------
    # Metadata / stats loading helpers.
    # ------------------------------------------------------------------
    def _load_metadata(self) -> dict:
        path = Path(self.root) / "metadata.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_source_root(self) -> str | None:
        with h5py.File(self.files[0], "r") as f:
            source_root = f.attrs.get("source_root")
        if source_root is not None:
            if isinstance(source_root, bytes):
                source_root = source_root.decode("utf-8")
            return str(source_root)

        dataset_cfg = self.metadata.get("data", {}).get("dataset", {})
        source_root = dataset_cfg.get("root")
        if isinstance(source_root, str) and source_root.startswith("${oc.env:"):
            default = source_root.rsplit(",", 1)[-1].rstrip("}")
            return os.environ.get("ERA5_DATA_ROOT", default)
        return source_root

    def _infer_variables(self) -> list[str]:
        dataset_cfg = self.metadata.get("data", {}).get("dataset", {})
        variables = dataset_cfg.get("variables", [])
        return list(variables)

    def _infer_ae_model_path(self) -> str | None:
        for name in ("ae_model.pt", "vae_model.pt"):
            path = os.path.join(self.root, name)
            if os.path.exists(path):
                return path
        return None

    def _load_latent_stats(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.normalize:
            return None, None
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                f"Latent stats file not found: {self.stats_path}. Run "
                "scripts/compute_ae_latent_stats.py to generate it, or pass "
                "normalize=False to use the raw latents."
            )
        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats = yaml.safe_load(f)

        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        if mean.shape != (self.n_channels,) or std.shape != (self.n_channels,):
            raise ValueError(
                f"stats.yaml mean/std must have shape ({self.n_channels},), got "
                f"mean={mean.shape}, std={std.shape}."
            )
        if not np.all(std > 0):
            raise ValueError("stats.yaml contains non-positive std values.")
        return mean, std

    def _load_weather_stats(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.source_root or not self.variables:
            return None, None

        mean_path = os.path.join(self.source_root, "normalize_mean.npz")
        std_path = os.path.join(self.source_root, "normalize_std.npz")
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            return None, None

        means = np.load(mean_path)
        stds = np.load(std_path)
        weather_means = np.stack([means[v] for v in self.variables], axis=0).reshape(
            -1, 1, 1
        )
        weather_stds = np.stack([stds[v] for v in self.variables], axis=0).reshape(
            -1, 1, 1
        )
        return weather_means, weather_stds

    def _load_latent(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            return np.array(f[self.latent_key], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        target = self.normalize_latents(self._load_latent(self.files[idx]))
        return torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load one ERA5 AE latent batch and print basic stats."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/latents",
        help="Root directory containing train/val/test latent HDF5 folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Load raw latents without per-channel standardization.",
    )
    args = parser.parse_args()

    dataset = ERA5AELatentDataset(
        root=args.root,
        split=args.split,
        normalize=not args.no_normalize,
    )
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = False
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(**dataloader_kwargs)

    batch = next(iter(dataloader)).to(torch.float32)

    print("root:", dataset.root)
    print("split:", dataset.split)
    print("normalize:", dataset.normalize)
    print("stats_path:", dataset.stats_path)
    print("dataset_size:", len(dataset))
    print("sample_shape:", dataset.sample_shape)
    print("img_resolution:", dataset.img_resolution)
    print("num_frames:", dataset.num_frames)
    print("n_channels:", dataset.n_channels)
    print("ae_model_path:", dataset.ae_model_path)
    print("source_root:", dataset.source_root)
    print("num_variables:", len(dataset.variables))
    print("batch_shape:", tuple(batch.shape))
    print("batch_mean:", float(batch.mean()))
    print("batch_std:", float(batch.std(unbiased=False)))

    reduce_dims = (0, 2, 3, 4)
    channel_mean = batch.mean(dim=reduce_dims)
    channel_std = batch.std(dim=reduce_dims, unbiased=False)
    print("per-channel mean (first 5):", channel_mean[:5].tolist())
    print("per-channel std  (first 5):", channel_std[:5].tolist())
