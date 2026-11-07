#!/usr/bin/env python3
"""Validate the transferred Laura bundle and write a NERSC runtime config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
from omegaconf import OmegaConf


EXPECTED = {
    "config": "4c503b82d1f1d5a03f5df962753186fcea8bdcda48327779eb2266e65fbeda19",
    "checkpoint": "490c3b8c71e6ecae79c210949a1ef32db7cbded6f55114bc87c821a90106a93b",
    "autoencoder": "87573b61a72d192865d1d5cc55dc118597aefb653a31a49688a68cd9088b7fa6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=932)
    parser.add_argument(
        "--verify-model-hashes",
        action="store_true",
        help="Hash the 7.8-GB prior and AE as well as the small config.",
    )
    parser.add_argument(
        "--verify-all-checksums",
        action="store_true",
        help="Verify every entry in the transferred SHA256SUMS.txt.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_checksum_list(bundle: Path) -> None:
    checksum_file = require(bundle / "SHA256SUMS.txt")
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.lstrip("* ")
        path = bundle / rel
        actual = sha256(require(path))
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {path}: {actual} != {expected}")
        checked += 1
    print(f"verified {checked} transferred checksums")


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    latent_root = bundle / "data" / "latent_root"
    source_root = bundle / "data" / "source_root"
    observations = bundle / "data" / "observations"
    config_path = require(bundle / "models" / "dj_config.yaml")
    checkpoint = require(bundle / "models" / "dj_checkpoint-019901.pt")
    autoencoder = require(args.autoencoder.resolve())
    require(observations / "global_joint_2020_year_superob.pkl")
    require(observations / "global_joint_2020_year.pkl")

    source_files = sorted((source_root / "test").glob("2020_*.h5"))
    control_samples = sorted((bundle / "control_k8").glob("fc_seed*/sample_0932/samples.npy"))
    latent_files = sorted((latent_root / "test").glob("*.h5"))
    if len(source_files) != 32:
        raise RuntimeError(f"expected 32 source frames, found {len(source_files)}")
    if len(control_samples) != 8:
        raise RuntimeError(
            f"expected 8 independent control samples, found {len(control_samples)}"
        )
    if len(latent_files) != 1:
        raise RuntimeError(f"expected one Laura latent, found {len(latent_files)}")

    with h5py.File(latent_files[0], "r") as handle:
        group = handle["input"] if "input" in handle else handle
        source_index = int(group["source_index"][()])
        referenced = list(group["source_files"][...])
    if source_index != args.source_index:
        raise RuntimeError(
            f"latent source_index is {source_index}, expected {args.source_index}"
        )
    if len(referenced) != 32:
        raise RuntimeError(f"latent references {len(referenced)} frames, expected 32")

    config_hash = sha256(config_path)
    if config_hash != EXPECTED["config"]:
        raise RuntimeError(f"reference config hash mismatch: {config_hash}")
    if args.verify_model_hashes:
        for label, path in (("checkpoint", checkpoint), ("autoencoder", autoencoder)):
            actual = sha256(path)
            if actual != EXPECTED[label]:
                raise RuntimeError(f"{label} hash mismatch: {actual}")
            print(f"verified {label} hash")
    if args.verify_all_checksums:
        verify_checksum_list(bundle)

    cfg = OmegaConf.load(config_path)
    cfg.data.dataset.root = str(latent_root)
    cfg.data.dataset.source_root = str(source_root)
    cfg.data.dataset.ae_model_path = str(autoencoder)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output_config)

    print(f"Laura bundle:       {bundle}")
    print(f"latent dataset idx: 0 (source_index={source_index})")
    print(f"source frames:      {len(source_files)}")
    print(f"independent control members: {len(control_samples)}")
    print(f"checkpoint:         {checkpoint}")
    print(f"autoencoder:        {autoencoder}")
    print(f"runtime config:     {args.output_config}")


if __name__ == "__main__":
    main()
