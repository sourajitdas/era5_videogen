"""Posterior sampling of ERA5 videos conditioned on sparse IGRA radiosonde data.

Reconstructs a full-resolution ERA5 video from real IGRA station observations
(standardized units, station [lat, lon] locations, a subset of variables) at a
few frames, using the trained TrigFlow latent diffusion prior. The likelihood
goes through the frozen AE decoder and the igra_gen-style station
interpolation operator.

NO ERA5 DATA LEAKAGE: the measurement is built exclusively from the IGRA
pickle. The ERA5 ground-truth video is loaded only AFTER sampling, to compute
evaluation metrics and plots.

Example
-------
python -m era5_videogen.generate_posterior_igra \
    --diffusion_config results/.../config.yaml \
    --diffusion_checkpoint results/.../checkpoint.pt \
    --igra_pkl samples_posterior/igra_2020.pkl \
    --sample_index 0 --obs_frames middle:8 --num_steps 50
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import h5py
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
from era5_videogen.generate_posterior import (
    NormalizedLatentDecoder,
    _plot_rmse_vs_time,
    _resolve_plot_variables,
    _rmse_vs_time,
    parse_obs_frames,
)
from era5_videogen.generating.conditioning_methods import (
    IGRAVideoConditioning,
    IGRAVideoOperator,
)
from era5_videogen.generating.diffusion import PosteriorDiffusionSampler

warnings.filterwarnings("ignore", category=UserWarning)

# Variable order of the IGRA pickle's 13 measurement channels (matches
# igra_gen's configs/data/era5_1.4_13.yaml, minus the land_sea_mask condition).
IGRA_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_500",
    "geopotential_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "temperature_500",
    "temperature_850",
    "specific_humidity_500",
    "specific_humidity_850",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Posterior sampling conditioned on IGRA radiosonde observations."
    )
    parser.add_argument("--diffusion_config", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./samples_igra")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample_index", type=int, nargs="+", default=[0])
    parser.add_argument("--num_posterior_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ae_model_path", type=str, default=None)
    # IGRA measurement model.
    parser.add_argument(
        "--obs_variables",
        type=str,
        default=None,
        help="Comma-separated ERA5 variable names matching the pkl channel order "
        "(defaults to the 13 IGRA variables). For ISD: "
        "2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure",
    )
    parser.add_argument(
        "--igra_pkl",
        type=str,
        default="samples_posterior/igra_2020.pkl",
        help="Pickle with igra_data[time][0/1][batch][channel] = queries/values.",
    )
    parser.add_argument(
        "--igra_time_offset",
        type=int,
        default=0,
        help="Offset added to (source_index + frame) when indexing the pickle.",
    )
    parser.add_argument(
        "--obs_frames",
        type=str,
        default="middle:8",
        help="Observed source frames: 'middle:K', 'every:K', 'lo-hi', or 'a,b,c'.",
    )
    # Guidance.
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-2,
        help="SDA annealing coefficient: var = meas_std^2 + gamma * sigma^2.",
    )
    parser.add_argument(
        "--meas_std",
        type=float,
        default=1e-2,
        help="Assumed observation noise std (standardized units) in the likelihood.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Efficiency knobs (mirror generate_posterior.py).
    parser.add_argument("--corrector_steps", type=int, default=0)
    parser.add_argument("--corrector_snr", type=float, default=0.3)
    parser.add_argument("--corrector_max_sigma", type=float, default=float("inf"))
    parser.add_argument("--corrector_min_sigma", type=float, default=0.0)
    parser.add_argument("--corrector_steps2", type=int, default=0)
    parser.add_argument("--corrector_max_sigma2", type=float, default=0.0)
    parser.add_argument("--guide_max_sigma", type=float, default=float("inf"))
    parser.add_argument("--guide_min_sigma", type=float, default=0.0)
    parser.add_argument("--guide_reuse", type=int, default=1)
    parser.add_argument("--guide_momentum", type=float, default=0.0)
    parser.add_argument("--restart_k", type=int, default=0)
    parser.add_argument("--restart_sigma", type=float, default=1.0)
    parser.add_argument("--restart_steps", type=int, default=10)
    parser.add_argument("--guide_norm", choices=["none", "dsg"], default="none")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument(
        "--no_net_grad",
        action="store_true",
        help="Differentiate the likelihood through the decoder only.",
    )
    parser.add_argument("--no_pp", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--plot_variables", type=str, default="t2m,u10m,q1000")

    args = parser.parse_args()
    if args.num_steps < 2:
        parser.error("--num_steps must be at least 2")
    if args.num_posterior_samples < 1:
        parser.error("--num_posterior_samples must be at least 1")
    return args


def _source_index(dataset, idx: int) -> int:
    with h5py.File(dataset.files[idx], "r") as f:
        group = f["input"] if "input" in f else f
        if "source_index" in group:
            return int(np.asarray(group["source_index"]))
    raise KeyError(f"{dataset.files[idx]} does not record input/source_index.")


def _build_igra_measurement(
    igra_data, src_idx: int, obs_frames: list[int], time_offset: int
):
    """Per-frame [queries, values] lists from the pickle. IGRA only — no ERA5."""
    q_frames, v_frames, counts = [], [], []
    for t in obs_frames:
        k = src_idx + t + time_offset
        if k < 0 or k >= len(igra_data):
            raise IndexError(
                f"IGRA index {k} (source {src_idx} + frame {t} + offset "
                f"{time_offset}) outside pickle range [0, {len(igra_data) - 1}]"
            )
        q_frames.append(igra_data[k][0][0])
        v_frames.append(igra_data[k][1][0])
        counts.append(int(sum(np.asarray(v).size for v in igra_data[k][1][0])))
    return [q_frames, v_frames], counts


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cfg = OmegaConf.load(args.diffusion_config)
    if not _is_trigflow_loss(cfg):
        raise ValueError("generate_posterior_igra.py expects a TrigFlow run.")
    sigma_data = float(cfg.loss.get("sigma_data", 1.0))
    sigma_min = args.sigma_min or float(cfg.loss.get("sigma_min", 0.002))
    sigma_max = args.sigma_max or float(cfg.loss.get("sigma_max", 80.0))

    run_name = args.name or (cfg.get("experiment_name", "posterior") + "_igra")
    output_root = Path(args.output_dir) / run_name
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

    print(f"Loading IGRA observations from {args.igra_pkl}")
    with open(args.igra_pkl, "rb") as f:
        igra_data = pickle.load(f)

    latent_shape = tuple(dataset.sample_shape)
    num_source_frames = int(dataset.source_timesteps)
    obs_frames = parse_obs_frames(args.obs_frames, num_source_frames)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    variables = list(dataset.variables)
    obs_vars = (
        [v.strip() for v in args.obs_variables.split(",") if v.strip()]
        if args.obs_variables
        else IGRA_VARIABLES
    )
    missing = [v for v in obs_vars if v not in variables]
    if missing:
        raise ValueError(f"Dataset is missing observation variables: {missing}")
    channel_map = [variables.index(v) for v in obs_vars]
    print(f"observation variables ({len(obs_vars)}): channel_map -> {channel_map}")
    operator = IGRAVideoOperator(obs_frames=obs_frames, channel_map=channel_map)

    for idx in args.sample_index:
        sample_dir = output_root / f"sample_{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== sample_index {idx} ===")

        # ---- measurement: IGRA only (no ERA5 anywhere in here) ----
        src_idx = _source_index(dataset, idx)
        measurement, obs_counts = _build_igra_measurement(
            igra_data, src_idx, obs_frames, args.igra_time_offset
        )
        print(f"  source_index {src_idx}; IGRA points per observed frame: {obs_counts}")

        conditioning = IGRAVideoConditioning(
            operator=operator,
            decode_fn=decode_fn,
            scale=args.guidance_scale,
            meas_std=args.meas_std,
            gamma=args.gamma,
            grad_clip=grad_clip,
            guide_norm=args.guide_norm,
        )
        sampler = PosteriorDiffusionSampler(net, conditioning)
        batch = args.num_posterior_samples
        condition = _condition_batch(dataset, batch, device)

        latents = torch.randn(
            (batch, *latent_shape), generator=generator, device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_start = time.perf_counter()
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
            guide_through_net=not args.no_net_grad,
            log_every=args.log_every,
            corrector_steps=args.corrector_steps,
            corrector_snr=args.corrector_snr,
            corrector_max_sigma=args.corrector_max_sigma,
            corrector_min_sigma=args.corrector_min_sigma,
            corrector_steps2=args.corrector_steps2,
            corrector_max_sigma2=args.corrector_max_sigma2,
            guide_max_sigma=args.guide_max_sigma,
            guide_min_sigma=args.guide_min_sigma,
            guide_reuse=args.guide_reuse,
            guide_momentum=args.guide_momentum,
            noise_scale=args.noise_scale,
        ).detach()
        # Guided restart cycles (see generate_posterior.py).
        for _ in range(args.restart_k):
            x0 = sampler.pos_trigflow_sampler(
                latents=x0,
                measurement=measurement,
                condition=condition,
                num_steps=args.restart_steps,
                sigma_data=sigma_data,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=args.rho,
                use_pp=not args.no_pp,
                guide_through_net=not args.no_net_grad,
                log_every=args.log_every,
                x_warm=x0,
                sigma_start=args.restart_sigma,
                corrector_steps=args.corrector_steps,
                corrector_snr=args.corrector_snr,
                corrector_max_sigma=args.corrector_max_sigma,
                corrector_min_sigma=args.corrector_min_sigma,
                corrector_steps2=args.corrector_steps2,
                corrector_max_sigma2=args.corrector_max_sigma2,
                guide_momentum=args.guide_momentum,
            ).detach()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = time.perf_counter() - sampling_start
        print(f"  sampling wall time: {sampling_seconds:.1f}s for {batch} draw(s)")

        if args.save_latents:
            np.save(sample_dir / "latents.npy", x0.cpu().numpy())

        print("Decoding posterior samples...")
        with torch.no_grad():
            decoded = torch.cat(
                [decode_fn(x0[i : i + 1]).float() for i in range(batch)], dim=0
            )
        # Clamp generated specific humidity at 0 (decoder output only); the
        # ERA5 truth below is denormalized without any clamp.
        samples_phys = clamp_generated_humidity(
            dataset.denormalize(decoded), list(dataset.variables))

        # ---- evaluation only: ERA5 truth enters here, after sampling ----
        truth_std = dataset.get_source_video(idx).unsqueeze(0)
        truth_phys = dataset.denormalize(truth_std)[0]

        np.save(sample_dir / "samples.npy", samples_phys)
        np.save(sample_dir / "ground_truth.npy", truth_phys)

        rmse_samples = np.stack(
            [_rmse_vs_time(samples_phys[b], truth_phys) for b in range(batch)], axis=0
        )
        ens_mean = samples_phys.mean(axis=0)
        rmse_mean = _rmse_vs_time(ens_mean, truth_phys)

        plot_vars = _resolve_plot_variables(args.plot_variables, list(dataset.variables))
        if plot_vars:
            _plot_rmse_vs_time(
                rmse_samples,
                rmse_mean,
                plot_vars,
                obs_frames,
                sample_dir / "rmse_vs_time.png",
            )

        metrics = {
            "obs_frames": obs_frames,
            "igra_points_per_frame": obs_counts,
            "sampling_seconds": sampling_seconds,
            "sampling_seconds_per_draw": sampling_seconds / batch,
            "num_draws": batch,
            "rmse_vs_time": {alias: rmse_mean[ch].tolist() for alias, ch in plot_vars},
            "rmse_obs_frames": {
                alias: float(np.mean(rmse_mean[ch][obs_frames])) for alias, ch in plot_vars
            },
            "rmse_unobs_frames": {
                alias: float(np.mean(np.delete(rmse_mean[ch], obs_frames)))
                for alias, ch in plot_vars
            },
            "rmse_overall_per_var": {
                alias: float(np.mean(rmse_mean[ch])) for alias, ch in plot_vars
            },
        }
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        for alias, ch in plot_vars:
            print(
                f"  {alias:8s} RMSE obs-frames {metrics['rmse_obs_frames'][alias]:.4f}  "
                f"unobs-frames {metrics['rmse_unobs_frames'][alias]:.4f}"
            )

    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "sampler": "trigflow-posterior-igra",
        "sampler_family": getattr(args, "sampler", "guided"),
        "sampler_args": {
            k: getattr(args, k)
            for k in vars(args)
            if k.startswith(("daps_", "diffpir_", "red_", "restart_", "hmc_", "bp_"))
        },
        "measurement": "igra",
        "igra_pkl": args.igra_pkl,
        "igra_time_offset": args.igra_time_offset,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "sigma_data": sigma_data,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": args.rho,
        "split": args.split,
        "sample_index": list(args.sample_index),
        "num_posterior_samples": args.num_posterior_samples,
        "obs_frames_spec": args.obs_frames,
        "obs_frames": obs_frames,
        "guidance_scale": args.guidance_scale,
        "gamma": args.gamma,
        "meas_std": args.meas_std,
        "grad_clip": args.grad_clip,
        "guide_through_net": not args.no_net_grad,
        "corrector_steps": args.corrector_steps,
        "corrector_snr": args.corrector_snr,
        "corrector_max_sigma": args.corrector_max_sigma,
        "corrector_steps2": args.corrector_steps2,
        "corrector_max_sigma2": args.corrector_max_sigma2,
        "guide_max_sigma": args.guide_max_sigma,
        "guide_reuse": args.guide_reuse,
        "guide_momentum": args.guide_momentum,
        "restart_k": args.restart_k,
        "restart_sigma": args.restart_sigma,
        "restart_steps": args.restart_steps,
        "use_pp": not args.no_pp,
        "checkpoint": args.diffusion_checkpoint,
        "config": args.diffusion_config,
        "variables": list(dataset.variables),
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved IGRA posterior samples under {output_root}")


if __name__ == "__main__":
    main()
