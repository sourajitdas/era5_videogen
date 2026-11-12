"""Joint repulsion-guided posterior sampling from real in-situ observations.

This is the sparse-observation analogue of
``generate_posterior_repulsion``.  All K particles are evolved in one batch,
share the reference real IGRA/ISD/ICOADS likelihood, and receive an RBF repulsive score
computed from decoded denoised fields.  ERA5 truth is loaded only after
sampling and is used exclusively for evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
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
from era5_videogen.generate_posterior import (
    NormalizedLatentDecoder,
    _plot_rmse_vs_time,
    _resolve_plot_variables,
    _rmse_vs_time,
    parse_obs_frames,
)
from era5_videogen.generate_posterior_igra import (
    IGRA_VARIABLES,
    _build_igra_measurement,
    _source_index,
)
from era5_videogen.generating.conditioning_methods_repulsion import (
    IGRAVideoConditioning,
    IGRAVideoOperator,
)
from era5_videogen.generating.diffusion_repulsion import PosteriorDiffusionSampler

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion_config", required=True)
    parser.add_argument("--diffusion_checkpoint", required=True)
    parser.add_argument("--ae_model_path", required=True)
    parser.add_argument("--igra_pkl", required=True)
    parser.add_argument("--obs_variables", default=None)
    parser.add_argument("--igra_time_offset", type=int, default=0)
    parser.add_argument("--output_dir", default="./samples_igra_repulsion")
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--sample_index",
        type=int,
        nargs="+",
        default=[0],
        help="Dataset-array index. A one-latent Laura bundle uses index 0, not 932.",
    )
    parser.add_argument("--num_posterior_samples", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--obs_frames", default="16-31")
    parser.add_argument("--guidance_scale", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=2e-6)
    parser.add_argument("--meas_std", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--corrector_steps", type=int, default=1)
    parser.add_argument("--corrector_snr", type=float, default=0.3)
    parser.add_argument("--corrector_max_sigma", type=float, default=3.0)
    parser.add_argument("--corrector_min_sigma", type=float, default=0.0)
    parser.add_argument("--corrector_steps2", type=int, default=0)
    parser.add_argument("--corrector_max_sigma2", type=float, default=0.0)
    parser.add_argument("--guide_max_sigma", type=float, default=float("inf"))
    parser.add_argument("--guide_min_sigma", type=float, default=0.0)
    parser.add_argument("--guide_reuse", type=int, default=1)
    parser.add_argument("--guide_momentum", type=float, default=0.0)
    parser.add_argument("--guide_norm", choices=["none", "dsg"], default="dsg")
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--restart_k", type=int, default=0)
    parser.add_argument("--restart_sigma", type=float, default=1.0)
    parser.add_argument("--restart_steps", type=int, default=10)
    parser.add_argument(
        "--no_net_grad",
        action="store_true",
        help="Use decoder-only likelihood gradients; recommended for K=8 on A100-80GB.",
    )
    parser.add_argument("--no_pp", action="store_true")
    parser.add_argument("--repulsion_strength", type=float, default=0.10)
    parser.add_argument("--repulsion_bandwidth", type=float, default=0.0)
    parser.add_argument("--repulsion_min_sigma", type=float, default=0.05)
    parser.add_argument("--repulsion_max_sigma", type=float, default=5.0)
    parser.add_argument(
        "--repulsion_schedule", choices=["constant", "sine"], default="sine"
    )
    parser.add_argument("--repulsion_spatial_pool", type=int, default=8)
    parser.add_argument("--repulsion_all_frames", action="store_true")
    parser.add_argument(
        "--repulsion_mask_mode",
        choices=["frames", "spatial", "split", "hurricane", "all"],
        default="frames",
        help=(
            "Where decoded-space repulsion acts: 'frames' keeps the original "
            "unobserved-frame-only behavior; 'spatial' also acts on unobserved "
            "pixels and variables inside observed frames; 'split' adds separate "
            "forecast-frame and observed-frame-complement objectives; 'all' uses "
            "every pixel; 'hurricane' repels differentiable storm-track and "
            "intensity descriptors over forecast frames."
        ),
    )
    parser.add_argument(
        "--repulsion_split_beta",
        type=float,
        default=1.0,
        help=(
            "Weight beta on the independently normalized unobserved-pixel "
            "objective when --repulsion_mask_mode=split."
        ),
    )
    parser.add_argument(
        "--repulsion_observation_buffer",
        type=int,
        default=0,
        help=(
            "Additional decoded-grid cells excluded around each bilinear station "
            "footprint when --repulsion_mask_mode is spatial or split."
        ),
    )
    parser.add_argument(
        "--repulsion_grad_norm", choices=["none", "rms"], default="rms"
    )
    parser.add_argument("--hurricane_region_lat", type=float, nargs=2, default=[15.0, 35.0])
    parser.add_argument("--hurricane_region_lon", type=float, nargs=2, default=[-100.0, -65.0])
    parser.add_argument(
        "--hurricane_descriptor_horizon",
        type=int,
        default=0,
        help="Number of earliest forecast frames in the descriptor; 0 uses all.",
    )
    parser.add_argument("--hurricane_pressure_temperature_hpa", type=float, default=2.0)
    parser.add_argument("--hurricane_wind_temperature_ms", type=float, default=2.0)
    parser.add_argument("--hurricane_track_radius_deg", type=float, default=6.0)
    parser.add_argument("--hurricane_position_scale_deg", type=float, default=2.0)
    parser.add_argument("--hurricane_pressure_scale_hpa", type=float, default=10.0)
    parser.add_argument("--hurricane_wind_scale_ms", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=2)
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument("--plot_variables", default="t2m,u10m,q1000")
    args = parser.parse_args()

    if args.num_steps < 2:
        parser.error("--num_steps must be at least 2")
    if args.num_posterior_samples < 2:
        parser.error("joint repulsion requires --num_posterior_samples >= 2")
    if args.repulsion_strength <= 0:
        parser.error("--repulsion_strength must be positive")
    if not 0 < args.repulsion_min_sigma < args.repulsion_max_sigma:
        parser.error("expected 0 < --repulsion_min_sigma < --repulsion_max_sigma")
    if args.repulsion_spatial_pool < 1:
        parser.error("--repulsion_spatial_pool must be at least 1")
    if args.repulsion_observation_buffer < 0:
        parser.error("--repulsion_observation_buffer must be non-negative")
    if args.repulsion_split_beta < 0:
        parser.error("--repulsion_split_beta must be non-negative")
    if args.hurricane_descriptor_horizon < 0:
        parser.error("--hurricane_descriptor_horizon must be non-negative")
    if args.hurricane_region_lat[0] >= args.hurricane_region_lat[1]:
        parser.error("--hurricane_region_lat must be increasing")
    if args.hurricane_region_lon[0] >= args.hurricane_region_lon[1]:
        parser.error("--hurricane_region_lon must be increasing")
    positive_hurricane_args = {
        "--hurricane_pressure_temperature_hpa": args.hurricane_pressure_temperature_hpa,
        "--hurricane_wind_temperature_ms": args.hurricane_wind_temperature_ms,
        "--hurricane_track_radius_deg": args.hurricane_track_radius_deg,
        "--hurricane_position_scale_deg": args.hurricane_position_scale_deg,
        "--hurricane_pressure_scale_hpa": args.hurricane_pressure_scale_hpa,
        "--hurricane_wind_scale_ms": args.hurricane_wind_scale_ms,
    }
    for name, value in positive_hurricane_args.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.repulsion_all_frames:
        if args.repulsion_mask_mode != "frames":
            parser.error(
                "--repulsion_all_frames cannot be combined with an explicit "
                "--repulsion_mask_mode"
            )
        args.repulsion_mask_mode = "all"
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cfg = OmegaConf.load(args.diffusion_config)
    if not _is_trigflow_loss(cfg):
        raise ValueError("Laura repulsion sampling expects a TrigFlow checkpoint")
    sigma_data = float(cfg.loss.get("sigma_data", 1.0))
    sigma_min = args.sigma_min or float(cfg.loss.get("sigma_min", 0.002))
    sigma_max = args.sigma_max or float(cfg.loss.get("sigma_max", 80.0))

    run_name = args.name or f"{cfg.get('experiment_name', 'posterior')}_igra_repulsion"
    output_root = Path(args.output_dir) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata from split={args.split}")
    dataset = _load_dataset(cfg, split=args.split)
    if not getattr(dataset, "is_ae_latent_dataset", False):
        raise ValueError("Expected an AE latent dataset")

    print("Building network...")
    net = _build_network(cfg, dataset, device=device)
    net.load_state_dict(_checkpoint_state_dict(args.diffusion_checkpoint, device))
    net.eval().requires_grad_(False)

    print("Loading AE decoder...")
    decoder = dataset.build_decoder(device, ae_model_path=args.ae_model_path)
    decoder.requires_grad_(False)
    decode_fn = NormalizedLatentDecoder(decoder=decoder, dataset=dataset)

    print(f"Loading in-situ observations from {args.igra_pkl}")
    with open(args.igra_pkl, "rb") as handle:
        igra_data = pickle.load(handle)

    latent_shape = tuple(dataset.sample_shape)
    obs_frames = parse_obs_frames(args.obs_frames, int(dataset.source_timesteps))
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
    hurricane_descriptor_config = None
    if args.repulsion_mask_mode == "hurricane":
        hurricane_variables = [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ]
        missing_hurricane = [name for name in hurricane_variables if name not in variables]
        if missing_hurricane:
            raise ValueError(
                f"Dataset is missing hurricane descriptor variables: {missing_hurricane}"
            )
        if dataset._weather_means is None or dataset._weather_stds is None:
            raise ValueError(
                "Hurricane descriptors require physical weather mean/std files"
            )
        hurricane_channels = [variables.index(name) for name in hurricane_variables]
        latitude, longitude = dataset.get_lat_lon()
        hurricane_descriptor_config = {
            "latitude": latitude,
            "longitude": longitude,
            "region_lat": list(args.hurricane_region_lat),
            "region_lon": list(args.hurricane_region_lon),
            "horizon": args.hurricane_descriptor_horizon,
            "mslp_channel": hurricane_channels[0],
            "u10_channel": hurricane_channels[1],
            "v10_channel": hurricane_channels[2],
            "weather_means": [
                float(np.asarray(dataset._weather_means[index]).squeeze())
                for index in hurricane_channels
            ],
            "weather_stds": [
                float(np.asarray(dataset._weather_stds[index]).squeeze())
                for index in hurricane_channels
            ],
            "pressure_temperature_hpa": args.hurricane_pressure_temperature_hpa,
            "wind_temperature_ms": args.hurricane_wind_temperature_ms,
            "track_radius_deg": args.hurricane_track_radius_deg,
            "position_scale_deg": args.hurricane_position_scale_deg,
            "pressure_scale_hpa": args.hurricane_pressure_scale_hpa,
            "wind_scale_ms": args.hurricane_wind_scale_ms,
        }
        print(
            "hurricane descriptor: "
            f"region lat={args.hurricane_region_lat}, lon={args.hurricane_region_lon}, "
            f"frames={args.hurricane_descriptor_horizon or 'all forecast'}"
        )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    source_indices: list[int] = []

    for dataset_idx in args.sample_index:
        src_idx = _source_index(dataset, dataset_idx)
        source_indices.append(src_idx)
        # Label by the physical ERA5 source index so this output aligns with
        # The reference sample_0932 control directories even though the bundle index is 0.
        sample_dir = output_root / f"sample_{src_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== dataset index {dataset_idx}; source index {src_idx} ===")

        measurement, obs_counts = _build_igra_measurement(
            igra_data, src_idx, obs_frames, args.igra_time_offset
        )
        print(f"  in-situ points per observed frame: {obs_counts}")
        conditioning = IGRAVideoConditioning(
            operator=operator,
            decode_fn=decode_fn,
            scale=args.guidance_scale,
            meas_std=args.meas_std,
            gamma=args.gamma,
            grad_clip=grad_clip,
            guide_norm=args.guide_norm,
            hurricane_descriptor_config=hurricane_descriptor_config,
        )
        sampler = PosteriorDiffusionSampler(net, conditioning)
        batch = args.num_posterior_samples
        condition = _condition_batch(dataset, batch, device)
        latents = torch.randn(
            (batch, *latent_shape), generator=generator, device=device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        common_sampler_args = dict(
            measurement=measurement,
            condition=condition,
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
            repulsion_strength=args.repulsion_strength,
            repulsion_bandwidth=args.repulsion_bandwidth,
            repulsion_min_sigma=args.repulsion_min_sigma,
            repulsion_max_sigma=args.repulsion_max_sigma,
            repulsion_schedule=args.repulsion_schedule,
            repulsion_spatial_pool=args.repulsion_spatial_pool,
            repulsion_unobserved_only=args.repulsion_mask_mode == "frames",
            repulsion_mask_mode=args.repulsion_mask_mode,
            repulsion_observation_buffer=args.repulsion_observation_buffer,
            repulsion_split_beta=args.repulsion_split_beta,
            repulsion_grad_norm=args.repulsion_grad_norm,
        )
        x0 = sampler.pos_trigflow_sampler(
            latents=latents, num_steps=args.num_steps, **common_sampler_args
        ).detach()
        for _ in range(args.restart_k):
            x0 = sampler.pos_trigflow_sampler(
                latents=x0,
                num_steps=args.restart_steps,
                x_warm=x0,
                sigma_start=args.restart_sigma,
                **common_sampler_args,
            ).detach()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = time.perf_counter() - start
        print(f"  sampling wall time: {sampling_seconds:.1f}s for {batch} particles")

        if args.save_latents:
            np.save(sample_dir / "latents.npy", x0.cpu().numpy())
        print("Decoding posterior particles...")
        with torch.no_grad():
            decoded = torch.cat(
                [decode_fn(x0[i : i + 1]).float() for i in range(batch)], dim=0
            )
        samples_phys = clamp_generated_humidity(
            dataset.denormalize(decoded), variables
        )

        # ERA5 is evaluation-only and enters after the posterior sample exists.
        truth_std = dataset.get_source_video(dataset_idx).unsqueeze(0)
        truth_phys = dataset.denormalize(truth_std)[0]
        np.save(sample_dir / "samples.npy", samples_phys)
        np.save(sample_dir / "ground_truth.npy", truth_phys)

        rmse_samples = np.stack(
            [_rmse_vs_time(samples_phys[b], truth_phys) for b in range(batch)]
        )
        rmse_mean = _rmse_vs_time(samples_phys.mean(axis=0), truth_phys)
        plot_vars = _resolve_plot_variables(args.plot_variables, variables)
        if plot_vars:
            _plot_rmse_vs_time(
                rmse_samples,
                rmse_mean,
                plot_vars,
                obs_frames,
                sample_dir / "rmse_vs_time.png",
            )
        metrics = {
            "dataset_index": dataset_idx,
            "source_index": src_idx,
            "obs_frames": obs_frames,
            "igra_points_per_frame": obs_counts,
            "sampling_seconds": sampling_seconds,
            "sampling_seconds_per_particle": sampling_seconds / batch,
            "num_particles": batch,
            "rmse_vs_time": {
                alias: rmse_mean[channel].tolist() for alias, channel in plot_vars
            },
            "rmse_obs_frames": {
                alias: float(np.mean(rmse_mean[channel][obs_frames]))
                for alias, channel in plot_vars
            },
            "rmse_unobs_frames": {
                alias: float(np.mean(np.delete(rmse_mean[channel], obs_frames)))
                for alias, channel in plot_vars
            },
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )

    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "sampler": "trigflow-posterior-in-situ-repulsion",
        "measurement": "igra_isd_icoads_superob",
        "particle_coupling": "global_all_particles",
        "dataset_indices": list(args.sample_index),
        "source_indices": source_indices,
        "num_posterior_samples": args.num_posterior_samples,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "obs_frames_spec": args.obs_frames,
        "obs_frames": obs_frames,
        "obs_variables": obs_vars,
        "igra_pkl": args.igra_pkl,
        "igra_time_offset": args.igra_time_offset,
        "guidance_scale": args.guidance_scale,
        "gamma": args.gamma,
        "meas_std": args.meas_std,
        "grad_clip": args.grad_clip,
        "guide_norm": args.guide_norm,
        "guide_through_net": not args.no_net_grad,
        "corrector_steps": args.corrector_steps,
        "corrector_snr": args.corrector_snr,
        "corrector_max_sigma": args.corrector_max_sigma,
        "noise_scale": args.noise_scale,
        "repulsion_strength": args.repulsion_strength,
        "repulsion_bandwidth": args.repulsion_bandwidth,
        "repulsion_min_sigma": args.repulsion_min_sigma,
        "repulsion_max_sigma": args.repulsion_max_sigma,
        "repulsion_schedule": args.repulsion_schedule,
        "repulsion_spatial_pool": args.repulsion_spatial_pool,
        # Legacy field retained for older analysis scripts.  The explicit mask
        # mode below is authoritative for new runs.
        "repulsion_unobserved_only": args.repulsion_mask_mode == "frames",
        "repulsion_mask_mode": args.repulsion_mask_mode,
        "repulsion_observation_buffer": args.repulsion_observation_buffer,
        "repulsion_split_beta": args.repulsion_split_beta,
        "repulsion_grad_norm": args.repulsion_grad_norm,
        "hurricane_descriptor": {
            "region_lat": list(args.hurricane_region_lat),
            "region_lon": list(args.hurricane_region_lon),
            "horizon": args.hurricane_descriptor_horizon,
            "pressure_temperature_hpa": args.hurricane_pressure_temperature_hpa,
            "wind_temperature_ms": args.hurricane_wind_temperature_ms,
            "track_radius_deg": args.hurricane_track_radius_deg,
            "position_scale_deg": args.hurricane_position_scale_deg,
            "pressure_scale_hpa": args.hurricane_pressure_scale_hpa,
            "wind_scale_ms": args.hurricane_wind_scale_ms,
        },
        "checkpoint": args.diffusion_checkpoint,
        "config": args.diffusion_config,
        "ae_model_path": args.ae_model_path,
        "variables": variables,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved joint in-situ posterior particles under {output_root}")


if __name__ == "__main__":
    main()
