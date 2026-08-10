"""Posterior-sampling super-resolution for the latent video diffusion model.

Reconstructs a full-resolution ERA5 video from low-resolution observations of a
few frames, using the trained (TrigFlow or EDM) latent diffusion prior. The
measurement likelihood goes through the frozen autoencoder decoder
(Latent-DPS, with SDA-style annealed guidance weighting):

    y = A(u_true) + n,    u = D(unnormalize(x0)),    x0 ~ latent diffusion prior

Example
-------
python -m era5_videogen.generate_posterior \
    --diffusion_config results/.../config.yaml \
    --diffusion_checkpoint results/.../network-snapshot.pt \
    --sample_index 0 --obs_downscale 8 --obs_frames middle:8 \
    --num_steps 50 --guidance_scale 1.0
"""

import argparse
import json
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
from era5_videogen.generating.conditioning_methods import (
    LatentVideoConditioning,
    VideoSuperResolutionOperator,
)
from era5_videogen.generating.diffusion import PosteriorDiffusionSampler

warnings.filterwarnings("ignore", category=UserWarning)

# Short aliases for commonly plotted ERA5 variables.
VARIABLE_ALIASES = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "q1000": "specific_humidity_1000",
    "z500": "geopotential_500",
    "t850": "temperature_850",
}


def parse_obs_frames(spec: str, num_frames: int) -> list[int]:
    """Parse an observed-frame spec: 'middle:K', 'start-end', 'a,b,c', 'every:K'."""
    spec = spec.strip()
    if spec.startswith("middle:"):
        k = int(spec.split(":", 1)[1])
        start = (num_frames - k) // 2
        frames = list(range(start, start + k))
    elif spec.startswith("every:"):
        k = int(spec.split(":", 1)[1])
        frames = list(range(0, num_frames, k))
    elif "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        frames = list(range(int(lo), int(hi) + 1))
    else:
        frames = [int(v) for v in spec.split(",") if v.strip() != ""]
    if not frames or any(f < 0 or f >= num_frames for f in frames):
        raise ValueError(f"Invalid obs_frames '{spec}' for {num_frames} frames")
    return frames


class NormalizedLatentDecoder(torch.nn.Module):
    """Differentiable map: normalized diffusion latent -> standardized ERA5 fields."""

    def __init__(self, decoder: torch.nn.Module, dataset):
        super().__init__()
        self.decoder = decoder
        self.dataset = dataset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dataset.unnormalize_latents(x)
        return self.decoder(z)


class NormalizedLatentEncoder(torch.nn.Module):
    """Differentiable map: standardized ERA5 fields -> normalized diffusion latent."""

    def __init__(self, autoencoder: torch.nn.Module, dataset):
        super().__init__()
        self.autoencoder = autoencoder
        self.dataset = dataset

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(fields)
        return self.dataset.normalize_latents(z)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Posterior-sampling video super-resolution with a latent diffusion prior."
    )
    parser.add_argument("--diffusion_config", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./samples_posterior")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--sample_index",
        type=int,
        nargs="+",
        default=[0],
        help="Index/indices of latent files in the split to reconstruct.",
    )
    parser.add_argument(
        "--num_posterior_samples",
        type=int,
        default=1,
        help="Posterior draws per measurement (sampled as one batch).",
    )
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ae_model_path", type=str, default=None)
    # Measurement model.
    parser.add_argument(
        "--obs_frames",
        type=str,
        default="middle:8",
        help="Observed source frames: 'middle:K', 'every:K', 'lo-hi', or 'a,b,c'.",
    )
    parser.add_argument(
        "--obs_downscale",
        type=int,
        default=8,
        help="Spatial downsampling factor of the observations.",
    )
    parser.add_argument(
        "--obs_mode",
        choices=["subsample", "bicubic", "bilinear", "area"],
        default="subsample",
        help="How the low-res observation is produced from the truth.",
    )
    parser.add_argument(
        "--obs_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to the (standardized) observations.",
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
        help="Assumed measurement noise std in the likelihood.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Elementwise clamp on the likelihood gradient (<=0 disables).",
    )
    parser.add_argument(
        "--no_net_grad",
        action="store_true",
        help="Differentiate the likelihood through the decoder only "
        "(FlowChef/MPGD approximation), not through the denoiser network.",
    )
    parser.add_argument(
        "--glue_weight",
        type=float,
        default=0.0,
        help="PSLD-style re-encoding consistency weight (0 disables; needs encoder).",
    )
    parser.add_argument(
        "--weighting",
        choices=["sda", "pigdm"],
        default="sda",
        help="Likelihood variance annealing: SDA (gamma*sigma^2) or piGDM (gamma*r_t^2).",
    )
    parser.add_argument(
        "--corrector_steps", type=int, default=0, help="Langevin corrector steps per level."
    )
    parser.add_argument("--corrector_snr", type=float, default=0.3)
    parser.add_argument(
        "--corrector_max_sigma", type=float, default=float("inf"),
        help="Apply correctors only when sigma <= this (concentrate compute).",
    )
    parser.add_argument("--corrector_min_sigma", type=float, default=0.0)
    # Graded correction: EXTRA corrector steps in a second, lower-noise band.
    parser.add_argument("--corrector_steps2", type=int, default=0)
    parser.add_argument("--corrector_max_sigma2", type=float, default=0.0)
    # Momentum guidance: EMA over likelihood scores across steps.
    parser.add_argument("--guide_momentum", type=float, default=0.0)
    # Ensemble-mean guidance: couple the batch through the mean prediction.
    parser.add_argument(
        "--ens_weight", type=float, default=0.0,
        help="Weight in [0,1] on fitting the batch-mean prediction to the data "
        "(requires num_posterior_samples > 1).",
    )
    # Guided restart cycles (Restart sampling adapted to the guided sampler):
    # after the main pass, forward-diffuse the clean estimate to restart_sigma
    # and re-descend with guidance, restart_k times.
    parser.add_argument("--restart_k", type=int, default=0)
    parser.add_argument("--restart_sigma", type=float, default=1.0)
    parser.add_argument("--restart_steps", type=int, default=10)
    # DSG-style sphere-normalized guidance gradient (arXiv:2402.03201).
    parser.add_argument("--guide_norm", choices=["none", "dsg"], default="none")
    # Back-projection + re-encode hard consistency (DDNM/CM4IR-flavored).
    parser.add_argument("--noise_scale", type=float, default=1.0)
    # ---- InverseBench variable-splitting samplers (arXiv:2503.11043) ----
    parser.add_argument(
        "--guide_max_sigma", type=float, default=float("inf"),
        help="Skip the likelihood gradient (and denoiser backward) when sigma > this.",
    )
    parser.add_argument("--guide_min_sigma", type=float, default=0.0)
    parser.add_argument(
        "--guide_reuse", type=int, default=1,
        help="Recompute the likelihood gradient every N steps, reuse it in between (lazy guidance).",
    )
    # SDEdit-style warm start from encoded observations (no ERA5 truth involved).
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Initialize from the encoded, upsampled observations instead of pure noise.",
    )
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=None,
        help="Reduced starting noise level for --warm_start (e.g. 10).",
    )
    parser.add_argument(
        "--no_pp", action="store_true", help="Use DPM instead of DPM++ updates."
    )
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_latents", action="store_true")
    parser.add_argument(
        "--plot_variables",
        type=str,
        default="t2m,u10m,q1000",
        help="Comma-separated variable aliases/names for the RMSE-vs-time plot.",
    )

    args = parser.parse_args()
    if args.num_steps < 2:
        parser.error("--num_steps must be at least 2")
    if args.num_posterior_samples < 1:
        parser.error("--num_posterior_samples must be at least 1")
    if args.obs_downscale < 1:
        parser.error("--obs_downscale must be >= 1")
    if args.sigma_start is not None and not args.warm_start:
        parser.error("--sigma_start requires --warm_start")
    return args


def _temporal_fill(y_up: torch.Tensor, obs_frames: list[int], num_frames: int) -> torch.Tensor:
    """Linearly interpolate observed (upsampled) frames onto the full time axis.

    y_up: (B, C, T_obs, H, W) -> (B, C, num_frames, H, W). Ends are held
    constant at the nearest observed frame. Uses only the observations.
    """
    out = y_up.new_zeros((*y_up.shape[:2], num_frames, *y_up.shape[-2:]))
    obs = list(obs_frames)
    for t in range(num_frames):
        if t <= obs[0]:
            out[:, :, t] = y_up[:, :, 0]
        elif t >= obs[-1]:
            out[:, :, t] = y_up[:, :, -1]
        else:
            k = max(i for i in range(len(obs)) if obs[i] <= t)
            if obs[k] == t:
                out[:, :, t] = y_up[:, :, k]
            else:
                w = (t - obs[k]) / (obs[k + 1] - obs[k])
                out[:, :, t] = (1 - w) * y_up[:, :, k] + w * y_up[:, :, k + 1]
    return out


def _resolve_plot_variables(spec: str, variables: list[str]) -> list[tuple[str, int]]:
    resolved = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name = VARIABLE_ALIASES.get(token, token)
        if name in variables:
            resolved.append((token, variables.index(name)))
        else:
            print(f"[warn] plot variable '{token}' ({name}) not in dataset variables; skipping")
    return resolved


def _rmse_vs_time(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-channel, per-frame latitude-weighted RMSE. (C, T, H, W) -> (C, T).

    Uses WeatherBench cos-latitude area weighting (evaluation/plotting metric only;
    the training loss is unaffected)."""
    from era5_videogen.utils.metrics import load_lat_weights, weighted_rmse_ct

    return weighted_rmse_ct(pred, truth, load_lat_weights())


def _plot_rmse_vs_time(
    rmse_samples: np.ndarray,
    rmse_mean: np.ndarray,
    plot_vars: list[tuple[str, int]],
    obs_frames: list[int],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(plot_vars)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    t_axis = np.arange(rmse_mean.shape[-1])
    for k, (alias, ch) in enumerate(plot_vars):
        ax = axes[0, k]
        for s in range(rmse_samples.shape[0]):
            ax.plot(t_axis, rmse_samples[s, ch], color="C0", alpha=0.35, lw=0.8)
        ax.plot(t_axis, rmse_mean[ch], color="C0", lw=2.0, label="posterior")
        for f in obs_frames:
            ax.axvline(f, color="0.85", lw=4, zorder=0)
        ax.set_title(alias)
        ax.set_xlabel("frame")
        ax.set_ylabel("RMSE (physical units)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Denormalized RMSE vs time (shaded: observed frames)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cfg = OmegaConf.load(args.diffusion_config)
    use_trigflow = _is_trigflow_loss(cfg)
    sigma_data = float(cfg.loss.get("sigma_data", 1.0))
    sigma_min = args.sigma_min or float(cfg.loss.get("sigma_min", 0.002))
    sigma_max = args.sigma_max or float(cfg.loss.get("sigma_max", 80.0))

    run_name = args.name or (cfg.get("experiment_name", "posterior") + "_posterior_sr")
    output_root = Path(args.output_dir) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata from split={args.split}")
    dataset = _load_dataset(cfg, split=args.split)
    if not getattr(dataset, "is_ae_latent_dataset", False):
        raise ValueError(
            "generate_posterior.py currently supports AE latent datasets "
            "(needs a deterministic decoder and source video access)."
        )

    print("Building network...")
    net = _build_network(cfg, dataset, device=device)
    net.load_state_dict(_checkpoint_state_dict(args.diffusion_checkpoint, device))
    net.eval().requires_grad_(False)

    print("Loading autoencoder...")
    need_encoder = args.glue_weight > 0 or args.warm_start
    autoencoder = None
    if need_encoder:
        autoencoder = dataset.build_autoencoder(device, ae_model_path=args.ae_model_path)
        autoencoder.requires_grad_(False)
        decoder = autoencoder.decode  # bound method: z -> standardized fields
        encode_fn = (
            NormalizedLatentEncoder(autoencoder, dataset)
            if args.glue_weight > 0
            else None
        )
    else:
        decoder = dataset.build_decoder(device, ae_model_path=args.ae_model_path)
        decoder.requires_grad_(False)
        encode_fn = None

    decode_fn = NormalizedLatentDecoder(decoder=decoder, dataset=dataset)

    # Geometry: latent (C, T, H, W) and decoded source video (C', T', H', W').
    latent_shape = tuple(dataset.sample_shape)
    num_source_frames = int(dataset.source_timesteps)
    obs_frames = parse_obs_frames(args.obs_frames, num_source_frames)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    grad_clip = args.grad_clip if args.grad_clip > 0 else None

    all_metrics = {}
    for idx in args.sample_index:
        sample_dir = output_root / f"sample_{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== sample_index {idx} ===")

        # Ground truth in standardized units, (1, C, T, H, W).
        truth_std = dataset.get_source_video(idx).unsqueeze(0).to(device)
        full_shape = tuple(truth_std.shape[-2:])
        in_shape = tuple(s // args.obs_downscale for s in full_shape)

        operator = VideoSuperResolutionOperator(
            target_shape=full_shape,
            in_shape=in_shape,
            obs_frames=obs_frames,
            mode=args.obs_mode,
        )
        measurement = operator.forward(truth_std)
        if args.obs_noise_std > 0:
            measurement = measurement + args.obs_noise_std * torch.randn(
                measurement.shape, generator=generator, device=device
            )

        batch = args.num_posterior_samples
        measurement_b = measurement.expand(batch, -1, -1, -1, -1)

        conditioning = LatentVideoConditioning(
            operator=operator,
            decode_fn=decode_fn,
            scale=args.guidance_scale,
            meas_std=args.meas_std,
            gamma=args.gamma,
            grad_clip=grad_clip,
            encode_fn=encode_fn,
            glue_weight=args.glue_weight,
            weighting=args.weighting,
            sigma_data=sigma_data,
            ens_weight=args.ens_weight,
            guide_norm=args.guide_norm,
        )
        sampler = PosteriorDiffusionSampler(net, conditioning)
        condition = _condition_batch(dataset, batch, device)

        # SDEdit-style warm start: encode the temporally-interpolated, spatially
        # upsampled observations. Built from the measurement only (no truth).
        x_warm = None
        if args.warm_start:
            with torch.no_grad():
                y_up = operator.transpose(measurement)
                pseudo = _temporal_fill(y_up, obs_frames, num_source_frames)
                z_raw = autoencoder.encode(pseudo.to(device))
                x_warm = dataset.normalize_latents(z_raw).expand(batch, -1, -1, -1, -1)

        latents = torch.randn(
            (batch, *latent_shape), generator=generator, device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_start = time.perf_counter()
        if use_trigflow:
            x0 = sampler.pos_trigflow_sampler(
                latents=latents,
                measurement=measurement_b,
                condition=condition,
                num_steps=args.num_steps,
                sigma_data=sigma_data,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=args.rho,
                use_pp=not args.no_pp,
                guide_through_net=not args.no_net_grad,
                log_every=args.log_every,
                x_warm=x_warm,
                sigma_start=args.sigma_start,
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
            )
            # Guided restart cycles: forward-diffuse the clean estimate back to
            # restart_sigma and re-descend with the same guided sampler. The
            # warm-start path (x_warm + sigma_start) performs the exact forward
            # noising x = cos(t0) x0 + sin(t0) sigma_d eps.
            for _ in range(args.restart_k):
                x0 = sampler.pos_trigflow_sampler(
                    latents=x0,
                    measurement=measurement_b,
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
                    noise_scale=args.noise_scale,
                )
        else:
            x0 = sampler.pos_edm_sampler(
                latents=latents,
                measurement=measurement_b,
                condition=condition,
                num_steps=args.num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=args.rho,
            )
        x0 = x0.detach()
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

        # Physical units. Generated samples clamp specific humidity at 0
        # (decoder output only); ERA5 truth is left exactly as denormalized.
        samples_phys = clamp_generated_humidity(
            dataset.denormalize(decoded), list(dataset.variables))  # (B, C, T, H, W)
        truth_phys = dataset.denormalize(truth_std)[0]  # (C, T, H, W)

        np.save(sample_dir / "samples.npy", samples_phys)
        np.save(sample_dir / "ground_truth.npy", truth_phys)
        np.save(sample_dir / "observation_lowres.npy", measurement.cpu().numpy())

        # Per-frame denormalized RMSE.
        rmse_samples = np.stack(
            [_rmse_vs_time(samples_phys[b], truth_phys) for b in range(batch)], axis=0
        )  # (B, C, T)
        ens_mean = samples_phys.mean(axis=0)
        rmse_mean = _rmse_vs_time(ens_mean, truth_phys)  # (C, T)

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
            "obs_shape": list(measurement.shape),
            "sampling_seconds": sampling_seconds,
            "sampling_seconds_per_draw": sampling_seconds / batch,
            "num_draws": batch,
            "rmse_vs_time": {
                alias: rmse_mean[ch].tolist() for alias, ch in plot_vars
            },
            "rmse_obs_frames": {
                alias: float(np.mean(rmse_mean[ch][obs_frames])) for alias, ch in plot_vars
            },
            "rmse_unobs_frames": {
                alias: float(
                    np.mean(np.delete(rmse_mean[ch], obs_frames))
                )
                for alias, ch in plot_vars
            },
            "rmse_overall_per_var": {
                alias: float(np.mean(rmse_mean[ch])) for alias, ch in plot_vars
            },
        }
        with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        all_metrics[idx] = metrics

        for alias, ch in plot_vars:
            print(
                f"  {alias:8s} RMSE obs-frames {metrics['rmse_obs_frames'][alias]:.4f}  "
                f"unobs-frames {metrics['rmse_unobs_frames'][alias]:.4f}"
            )

    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "sampler": "trigflow-posterior" if use_trigflow else "edm-posterior",
        "sampler_family": getattr(args, "sampler", "guided"),
        "sampler_args": {
            k: getattr(args, k)
            for k in vars(args)
            if k.startswith(("daps_", "diffpir_", "red_", "restart_", "hmc_", "bp_"))
        },
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
        "obs_downscale": args.obs_downscale,
        "obs_mode": args.obs_mode,
        "obs_noise_std": args.obs_noise_std,
        "guidance_scale": args.guidance_scale,
        "gamma": args.gamma,
        "meas_std": args.meas_std,
        "grad_clip": args.grad_clip,
        "guide_through_net": not args.no_net_grad,
        "glue_weight": args.glue_weight,
        "weighting": args.weighting,
        "corrector_steps": args.corrector_steps,
        "corrector_snr": args.corrector_snr,
        "corrector_max_sigma": args.corrector_max_sigma,
        "corrector_min_sigma": args.corrector_min_sigma,
        "corrector_steps2": args.corrector_steps2,
        "corrector_max_sigma2": args.corrector_max_sigma2,
        "guide_max_sigma": args.guide_max_sigma,
        "guide_min_sigma": args.guide_min_sigma,
        "guide_reuse": args.guide_reuse,
        "guide_momentum": args.guide_momentum,
        "ens_weight": args.ens_weight,
        "restart_k": args.restart_k,
        "restart_sigma": args.restart_sigma,
        "restart_steps": args.restart_steps,
        "guide_norm": args.guide_norm,
        "noise_scale": args.noise_scale,
        "warm_start": args.warm_start,
        "sigma_start": args.sigma_start,
        "use_pp": not args.no_pp,
        "checkpoint": args.diffusion_checkpoint,
        "config": args.diffusion_config,
        "variables": list(dataset.variables),
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved posterior samples under {output_root}")


if __name__ == "__main__":
    main()
