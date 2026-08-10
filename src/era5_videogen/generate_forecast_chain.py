"""Autoregressive multi-window forecast: chain 8-day windows into a ~15-16 day
forecast by using the previous window's last generated frame(s) as the dense
initial condition of the next window.

Window 1 is an existing obs->forecast run (e.g. the 2-day-window test_superob
run): its generated trailing ``--ic_frames`` frames are standardized and act as
a dense all-channel observation on the leading frames of window 2, sampled with
the same guided TrigFlow sampler. Repeat for ``--num_windows``. With 32-frame
(8-day) windows and a 1-frame overlap, two windows span 15.75 days.

NO ERA5 LEAKAGE: the continuation windows condition only on the model's own
previous output. ERA5 truth is loaded after sampling, for evaluation files.

Example
-------
python -m era5_videogen.generate_forecast_chain \
    --diffusion_config results/.../config.yaml \
    --diffusion_checkpoint results/.../checkpoint.pt \
    --window1_dir outputs/.../obs_to_forecast/test_superob/fc_seed17 \
    --sample_index 60 300 --guidance_scale 0.4 --guide_norm dsg
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from era5_videogen.generate import (
    _build_network,
    _checkpoint_state_dict,
    _condition_batch,
    _is_trigflow_loss,
    _load_dataset,
    clamp_generated_humidity,
)
from era5_videogen.generate_posterior import NormalizedLatentDecoder
from era5_videogen.generate_posterior_igra import _source_index
from era5_videogen.generating.conditioning_methods import (
    DenseFrameConditionOperator,
    IGRAVideoConditioning,
)
from era5_videogen.generating.diffusion import PosteriorDiffusionSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chain forecast windows autoregressively (dense-IC guidance)."
    )
    parser.add_argument("--diffusion_config", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--ae_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--window1_dir",
        type=str,
        required=True,
        help="Existing obs->forecast run dir (one seed) holding "
        "sample_XXXX/samples.npy for every --sample_index.",
    )
    parser.add_argument("--sample_index", type=int, nargs="+", required=True)
    parser.add_argument(
        "--num_windows", type=int, default=2,
        help="Total windows including window 1 (2 -> 15.75-day span).")
    parser.add_argument(
        "--ic_frames", type=int, default=1,
        help="Trailing frames of the previous window used as the dense IC.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    # Guidance (defaults mirror the o2f test runs).
    parser.add_argument("--guidance_scale", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=2e-6)
    parser.add_argument("--meas_std", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--guide_norm", choices=["none", "dsg"], default="dsg")
    parser.add_argument("--corrector_steps", type=int, default=1)
    parser.add_argument("--corrector_snr", type=float, default=0.3)
    parser.add_argument("--corrector_max_sigma", type=float, default=3.0)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--no_pp", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    if args.num_windows < 2:
        parser.error("--num_windows must be at least 2 (window 1 already exists)")
    if args.ic_frames < 1:
        parser.error("--ic_frames must be at least 1")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.diffusion_config)
    if not _is_trigflow_loss(cfg):
        raise ValueError("generate_forecast_chain.py expects a TrigFlow run.")
    sigma_data = float(cfg.loss.get("sigma_data", 1.0))
    sigma_min = float(cfg.loss.get("sigma_min", 0.002))
    sigma_max = float(cfg.loss.get("sigma_max", 80.0))

    output_root = Path(args.output_dir) / args.name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata from split={args.split}")
    dataset = _load_dataset(cfg, split=args.split)
    if not getattr(dataset, "is_ae_latent_dataset", False):
        raise ValueError("Expected an AE latent dataset.")

    print("Building network...")
    net = _build_network(cfg, dataset, device=device)
    net.load_state_dict(_checkpoint_state_dict(args.diffusion_checkpoint, device))
    net.eval().requires_grad_(False)

    print("Loading AE decoder...")
    decoder = dataset.build_decoder(device, ae_model_path=args.ae_model_path)
    decoder.requires_grad_(False)
    decode_fn = NormalizedLatentDecoder(decoder=decoder, dataset=dataset)

    latent_shape = tuple(dataset.sample_shape)
    num_source_frames = int(dataset.source_timesteps)
    frames_new = num_source_frames - args.ic_frames

    # physical <-> standardized stats for the IC handoff
    means, stds = dataset._weather_means, dataset._weather_stds  # (C, 1, 1)
    if means is None or stds is None:
        raise RuntimeError("Dataset is missing ERA5 normalize_mean/std stats.")

    operator = DenseFrameConditionOperator(obs_frames=list(range(args.ic_frames)))
    conditioning = IGRAVideoConditioning(
        operator=operator,
        decode_fn=decode_fn,
        scale=args.guidance_scale,
        meas_std=args.meas_std,
        gamma=args.gamma,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        guide_norm=args.guide_norm,
    )
    sampler = PosteriorDiffusionSampler(net, conditioning)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    condition = _condition_batch(dataset, 1, device)

    for idx in args.sample_index:
        w1 = Path(args.window1_dir) / f"sample_{idx:04d}" / "samples.npy"
        arr = np.load(w1)
        prev_phys = arr.reshape(-1, *arr.shape[-4:])[0].astype(np.float32)  # (C,T,H,W)
        src0 = _source_index(dataset, idx)
        print(f"=== chain from sample_index {idx} (source {src0}) ===")

        for w in range(1, args.num_windows):
            idx_w = idx + w * frames_new
            if idx_w >= len(dataset):
                raise RuntimeError(
                    f"window {w + 1} needs dataset index {idx_w} but the "
                    f"{args.split} split only has {len(dataset)} windows; "
                    f"reduce --num_windows for sample_index {idx}"
                )
            src_w = _source_index(dataset, idx_w)
            if src_w != src0 + w * frames_new:
                raise RuntimeError(
                    f"window stride mismatch: dataset[{idx_w}] has source {src_w}, "
                    f"expected {src0 + w * frames_new}"
                )
            ic_phys = prev_phys[:, num_source_frames - args.ic_frames:]  # (C,F,H,W)
            ic_std = (ic_phys - means[:, None]) / stds[:, None]
            ic_t = torch.as_tensor(
                np.ascontiguousarray(ic_std.transpose(1, 0, 2, 3)),
                device=device, dtype=torch.float32,
            )  # (F, C, H, W)
            measurement = [ic_t, None]

            latents = torch.randn((1, *latent_shape), generator=generator,
                                  device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            x0 = sampler.pos_trigflow_sampler(
                latents=latents,
                measurement=measurement,
                condition=condition,
                num_steps=args.num_steps,
                sigma_data=sigma_data,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=args.rho,
                use_pp=not args.no_pp,
                guide_through_net=True,
                log_every=args.log_every,
                corrector_steps=args.corrector_steps,
                corrector_snr=args.corrector_snr,
                corrector_max_sigma=args.corrector_max_sigma,
                noise_scale=args.noise_scale,
            ).detach()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            secs = time.perf_counter() - t0

            with torch.no_grad():
                decoded = decode_fn(x0).float()
            samples_phys = clamp_generated_humidity(
                dataset.denormalize(decoded), list(dataset.variables))

            # evaluation only: ERA5 truth enters here, after sampling
            truth_std = dataset.get_source_video(idx_w).unsqueeze(0)
            truth_phys = dataset.denormalize(truth_std)[0]

            wdir = output_root / f"sample_{idx:04d}" / f"window_{w + 1}"
            wdir.mkdir(parents=True, exist_ok=True)
            np.save(wdir / "samples.npy", samples_phys.astype(np.float32))
            np.save(wdir / "ground_truth.npy", truth_phys.astype(np.float32))
            ic_err = float(
                np.sqrt(((samples_phys[0][:, :args.ic_frames] - ic_phys) ** 2).mean()))
            print(f"  window {w + 1} (dataset idx {idx_w}): {secs:.1f}s, "
                  f"IC handoff RMSE (phys, all-channel) {ic_err:.4f}")
            json.dump(
                {"dataset_index": idx_w, "source_index": src_w,
                 "sampling_seconds": secs, "ic_handoff_rmse_phys": ic_err},
                open(wdir / "metrics.json", "w"), indent=2)

            prev_phys = samples_phys[0].astype(np.float32)

    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "sampler": "trigflow-forecast-chain",
        "window1_dir": args.window1_dir,
        "sample_index": list(args.sample_index),
        "num_windows": args.num_windows,
        "ic_frames": args.ic_frames,
        "frames_per_window": num_source_frames,
        "frames_new_per_window": frames_new,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "split": args.split,
        "guidance_scale": args.guidance_scale,
        "gamma": args.gamma,
        "meas_std": args.meas_std,
        "grad_clip": args.grad_clip,
        "guide_norm": args.guide_norm,
        "corrector_steps": args.corrector_steps,
        "corrector_snr": args.corrector_snr,
        "corrector_max_sigma": args.corrector_max_sigma,
        "noise_scale": args.noise_scale,
        "checkpoint": args.diffusion_checkpoint,
        "config": args.diffusion_config,
        "variables": list(dataset.variables),
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved chained forecast windows under {output_root}")


if __name__ == "__main__":
    main()
