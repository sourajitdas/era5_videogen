"""Build the bundled demo dataset from a full ERA5 / latent / observation corpus.

The repository ships a slice of the data used in the paper: one of the eight
test windows of the evaluation protocol, and the station observations covering
it. This script produced that slice, and is also how you extend the demo once
you have the full corpus (pass more --windows).

Windows have stride 1, so window w covers ERA5 source frames w..w+31.

Example
-------
python scripts/prepare_demo_data.py \
    --latent_root /path/to/era5_ae/128_8_32_64_surfcond \
    --source_root /path/to/1.40625deg_from_full_res_1_step_6hr_h5df \
    --obs_dir /path/to/data_obs --out data --windows 93
"""

import argparse
import json
import os
import pickle
import shutil
from glob import glob
from pathlib import Path

import h5py
import numpy as np

# Frame-indexed observation lists covering 2020 at 6 h resolution. A leading
# slice keeps every frame index identical to the full file, so the sampling
# commands need no offset.
OBS_FILES = [
    # The joint file is the merged network (IGRA soundings + ISD land surface +
    # ICOADS marine surface, super-obbed to one report per grid cell) and is what
    # the observation-to-forecast experiments assimilate. The three single-source
    # files are shipped as well so each network can be used on its own.
    ("global_joint_2020_all_year_superob.pkl", "joint_superob_2020_demo.pkl"),
    ("igra_2020_all.pkl", "igra_2020_demo.pkl"),
    ("isd_2020_year.pkl", "isd_2020_demo.pkl"),
    ("icoads_2020_year.pkl", "icoads_2020_demo.pkl"),
]

# Static fields the surface-conditioned decoder reads at run time.
STATIC_VARIABLES = ["land_sea_mask", "geopotential_at_surface"]


def blank_frame(entry):
    """An observation frame with the same structure but no reports."""
    queries, values = entry
    return [[[np.zeros((0, 2), np.float32) for _ in group] for group in queries],
            [[np.zeros((0,), np.float32) for _ in group] for group in values]]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent_root", required=True,
                   help="Latent dataset directory (holds stats.yaml, metadata.json, test/).")
    p.add_argument("--source_root", required=True,
                   help="Raw ERA5 directory (holds normalize_*.npz, lat/lon.npy, test/).")
    p.add_argument("--obs_dir", default=None, help="Directory of observation pickles.")
    p.add_argument("--out", default="data", help="Output directory of the demo bundle.")
    p.add_argument("--windows", type=int, nargs="+", default=[93],
                   help="Test-split window indices to include.")
    p.add_argument("--split", default="test")
    p.add_argument("--compression_level", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    lat_out, era_out = out / "latents", out / "era5"
    (lat_out / args.split).mkdir(parents=True, exist_ok=True)
    (era_out / args.split).mkdir(parents=True, exist_ok=True)
    (era_out / "train").mkdir(parents=True, exist_ok=True)

    meta = json.load(open(Path(args.latent_root) / "metadata.json"))
    variables = list(meta["data"]["dataset"]["variables"])
    print(f"{len(variables)} modelled variables")

    # ---- latents ------------------------------------------------------------
    latent_files = sorted(glob(os.path.join(args.latent_root, args.split, "*.h5")))
    frames: set[str] = set()
    for w in args.windows:
        src = latent_files[w]
        dst = lat_out / args.split / os.path.basename(src)
        shutil.copyfile(src, dst)
        with h5py.File(dst, "r+") as f:
            # Point the copy at the bundled ERA5 slice rather than the machine it
            # was encoded on (an explicit source_root in the config still wins).
            f.attrs["source_root"] = str(era_out)
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in np.atleast_1d(f["input/source_files"][...])]
        frames.update(os.path.basename(n) for n in names)
        print(f"  window {w}: {os.path.basename(src)} <- {len(names)} source frames")

    for name in ["stats.yaml", "metadata.json", "manifest.json"]:
        src = Path(args.latent_root) / name
        if src.exists():
            shutil.copyfile(src, lat_out / name)

    # ---- ERA5 frames --------------------------------------------------------
    for name in ["lat.npy", "lon.npy", "normalize_mean.npz", "normalize_std.npz"]:
        shutil.copyfile(Path(args.source_root) / name, era_out / name)

    kw = dict(compression="gzip", compression_opts=args.compression_level)
    total = 0
    frames = sorted(frames)
    for i, name in enumerate(frames):
        with h5py.File(Path(args.source_root) / args.split / name, "r") as fin, \
                h5py.File(era_out / args.split / name, "w") as fout:
            gin = fin["input"] if "input" in fin else fin
            gout = fout.create_group("input")
            for v in variables:
                gout.create_dataset(
                    v, data=np.squeeze(np.asarray(gin[v][...], dtype=np.float32)), **kw)
        total += (era_out / args.split / name).stat().st_size
        if (i + 1) % 8 == 0 or i == len(frames) - 1:
            print(f"  frame {i + 1}/{len(frames)}  ({total / 1e6:.0f} MB)")

    src = sorted(glob(os.path.join(args.source_root, "train", "*.h5")))[0]
    with h5py.File(src, "r") as fin, h5py.File(era_out / "train" / "static.h5", "w") as fout:
        gin = fin["input"] if "input" in fin else fin
        gout = fout.create_group("input")
        for v in STATIC_VARIABLES:
            gout.create_dataset(
                v, data=np.squeeze(np.asarray(gin[v][...], dtype=np.float32)), **kw)
    print(f"  static conditioning fields from {os.path.basename(src)}")

    # ---- observations -------------------------------------------------------
    if args.obs_dir:
        obs_out = out / "obs"
        obs_out.mkdir(parents=True, exist_ok=True)
        # Keep the list length so every frame index matches the full file, but
        # empty the frames before the first window: the released slice carries
        # observations only for the windows it ships.
        n_first, n_keep = min(args.windows), max(args.windows) + 32
        for src_name, dst_name in OBS_FILES:
            src = Path(args.obs_dir) / src_name
            if not src.exists():
                print(f"  skipping missing {src_name}")
                continue
            with open(src, "rb") as f:
                data = pickle.load(f)
            subset = [blank_frame(e) if i < n_first else e
                      for i, e in enumerate(data[:n_keep])]
            with open(obs_out / dst_name, "wb") as f:
                pickle.dump(subset, f, protocol=4)
            print(f"  {dst_name}: frames {n_first}-{n_keep - 1} of {len(data)}, "
                  f"{(obs_out / dst_name).stat().st_size / 1e6:.0f} MB")
        chan = Path(args.obs_dir) / "igra_2020_all_channels.json"
        if chan.exists():
            shutil.copyfile(chan, obs_out / "igra_2020_all_channels.json")

    print(f"\ndemo bundle written to {out.resolve()}")


if __name__ == "__main__":
    main()
