"""Conditioning methods extended for repulsion-guided particle sampling."""

import torch
from functools import partial
from torch.nn import functional as F
import numpy as np

class ConditioningMethod:
    def __init__(self, operator , num_sampling = 1, scale=1, **kwargs):
        super().__init__()
        self.operator = operator
        self.num_sampling = num_sampling
        self.scale = scale
    
    ## Sparse Sensors
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, sigma,
                     n_points=1000, std=1e-2, gamma=1e-3, mu=1, **kwargs):

        b, _, h, w = x_0_hat.shape
        device = x_0_hat.device

        h_idx = torch.randint(0, h, size=(b, n_points), device=device)
        w_idx = torch.randint(0, w, size=(b, n_points), device=device)
        b_idx = torch.arange(b, device=device).unsqueeze(-1).expand(-1, n_points)

        sampled_true = measurement[b_idx, :, h_idx, w_idx]
        mean_exp = x_0_hat[b_idx, :, h_idx, w_idx]
        noise = torch.randn_like(sampled_true) * std
        sampled_true_noisy = sampled_true + noise
        
        err1 = (sampled_true_noisy - mean_exp)**2
        var = std**2 + gamma * (sigma/mu)**2
        log_p = -(err1 / var).sum() / 2

        # Optimize gradient computation - no need for higher order gradients
        grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=False, create_graph=False)[0]
        grad = torch.clamp(grad, -1, 1)

        scaled_score = grad * self.scale * sigma

        return scaled_score, err1.mean().item()

    ## Mask-based Sparse Sensors
    def mask_conditioning(self, x_prev, x_t, x_0_hat, measurement, sigma, mask,
                         std=1e-2, gamma=1e-3, mu=1, **kwargs):

        b, c, h, w = x_0_hat.shape
        device = x_0_hat.device

        # Mask can be either (b, 1, h, w) or (b, c, h, w) where channels are repeated
        # Extract spatial mask: if single channel, use it directly; otherwise take max across channels
        if mask.shape[1] == 1:
            # Single channel mask: (b, 1, h, w) -> (b, h, w)
            spatial_mask = mask.squeeze(1) > 0.5  # (b, h, w) boolean mask
        else:
            # Multi-channel mask (expanded): take max across channels to get spatial locations
            spatial_mask = mask.max(dim=1, keepdim=False)[0] > 0.5  # (b, h, w) boolean mask

        # Get active spatial indices
        active_indices = spatial_mask.nonzero(as_tuple=False)  # (N_active, 3) with [batch_idx, h_idx, w_idx]

        if active_indices.numel() == 0:
            # No active points, return zero gradient
            return torch.zeros_like(x_prev), 0.0

        # Extract indices
        b_idx = active_indices[:, 0]
        h_idx = active_indices[:, 1]
        w_idx = active_indices[:, 2]

        # Sample from measurement at active spatial locations (all channels)
        sampled_true = measurement[b_idx, :, h_idx, w_idx]  # (N_active, c)
        mean_exp = x_0_hat[b_idx, :, h_idx, w_idx]  # (N_active, c)

        err1 = (sampled_true - mean_exp)**2
        var = std**2 + gamma * (sigma/mu)**2
        log_p = -(err1 / var).sum() / 2

        # Optimize gradient computation - no need for higher order gradients
        grad = torch.autograd.grad(outputs=log_p, inputs=x_prev, retain_graph=False, create_graph=False)[0]

        grad = torch.clamp(grad, -1, 1)
        scaled_score = grad * self.scale * sigma

        return scaled_score, err1.mean().item()
    

class VideoSuperResolutionOperator:
    """Measurement operator for video super-resolution with partial frame coverage.

    ``A(u)``: select the observed frames of a ``(B, C, T, H, W)`` video and
    reduce them to a low-resolution spatial grid. With ``mode="subsample"``
    the low-res measurement is a strided subsampling of the full-resolution
    field (``u[..., ::f, ::f]``); interpolation modes (``"bicubic"``,
    ``"bilinear"``, ``"area"``) are also supported.
    """

    def __init__(
        self,
        target_shape=(128, 256),
        in_shape=(16, 32),
        obs_frames=None,
        mode="subsample",
    ):
        super().__init__()
        self.target_shape = tuple(int(v) for v in target_shape)
        self.in_shape = tuple(int(v) for v in in_shape)
        self.obs_frames = None if obs_frames is None else [int(f) for f in obs_frames]
        self.mode = mode
        if any(t % i != 0 for t, i in zip(self.target_shape, self.in_shape)):
            raise ValueError(
                f"target_shape {self.target_shape} must be divisible by in_shape {self.in_shape}"
            )
        self.strides = tuple(t // i for t, i in zip(self.target_shape, self.in_shape))

    def select_frames(self, video):
        if self.obs_frames is None:
            return video
        return video[:, :, self.obs_frames]

    def _spatial_down(self, video):
        if self.mode == "subsample":
            sh, sw = self.strides
            return video[..., ::sh, ::sw]
        b, c, t, h, w = video.shape
        flat = video.reshape(b * c, t, h, w).transpose(0, 1).reshape(t * b * c, 1, h, w)
        kwargs = {"align_corners": False} if self.mode in {"bicubic", "bilinear"} else {}
        flat = F.interpolate(flat, size=self.in_shape, mode=self.mode, **kwargs)
        h2, w2 = self.in_shape
        return flat.reshape(t, b * c, h2, w2).transpose(0, 1).reshape(b, c, t, h2, w2)

    def forward(self, video, **kwargs):
        return self._spatial_down(self.select_frames(video))

    def back_project(self, video, measurement, weight=1.0):
        """DDNM/CM4IR-style back-projection: correct the observed frames with
        the (interpolating pseudo-inverse) upsampled residual y - A(u)."""
        corr = self.transpose(measurement - self.forward(video))
        out = video.clone()
        if self.obs_frames is None:
            return out + weight * corr
        out[:, :, self.obs_frames] = out[:, :, self.obs_frames] + weight * corr
        return out

    def transpose(self, data, **kwargs):
        """Upsample low-res observed frames back to the full spatial grid."""
        b, c, t, h, w = data.shape
        flat = data.reshape(b * c, t, h, w).transpose(0, 1).reshape(t * b * c, 1, h, w)
        mode = "bicubic" if self.mode == "subsample" else self.mode
        kwargs2 = {"align_corners": False} if mode in {"bicubic", "bilinear"} else {}
        flat = F.interpolate(flat, size=self.target_shape, mode=mode, **kwargs2)
        h2, w2 = self.target_shape
        return flat.reshape(t, b * c, h2, w2).transpose(0, 1).reshape(b, c, t, h2, w2)


class LatentVideoConditioning:
    """DPS/SDA-style likelihood guidance for latent video diffusion models.

    The clean-data estimate ``x_0_hat`` lives in the (normalized) diffusion
    latent space; the measurement lives in (standardized) pixel/field space.
    The likelihood therefore goes through the autoencoder decoder:

        log p(y | x_t) ~= -||y - A(D(x_0_hat(x_t)))||^2 / (2 var(sigma))

    with the SDA-style annealed variance ``var = std^2 + gamma * (sigma/mu)^2``
    (Rozet & Louppe 2023), which automatically weakens the guidance at high
    noise. ``sigma`` is the EDM-equivalent noise level (``sigma_d * tan(t)``
    for TrigFlow), and the returned score is pre-scaled by ``scale * sigma`` so
    samplers can use it exactly like the original ConditioningMethod: with
    ``scale = 1`` this matches the exact guided probability-flow ODE.

    The gradient is taken w.r.t. ``x_prev`` when it carries gradients (full
    DPS, backprop through the denoiser network) and w.r.t. ``x_0_hat``
    otherwise (FlowChef/MPGD approximation: denoiser Jacobian ~ identity,
    much cheaper since only the decoder is differentiated).

    Optionally a PSLD-style "gluing" penalty ``glue_weight * ||x_0_hat -
    E(D(x_0_hat))||^2`` keeps the latent estimate on the encoder manifold
    (requires ``encode_fn``).
    """

    def __init__(
        self,
        operator,
        decode_fn,
        scale: float = 1.0,
        meas_std: float = 1e-2,
        gamma: float = 1e-2,
        grad_clip: float | None = 1.0,
        encode_fn=None,
        glue_weight: float = 0.0,
        weighting: str = "sda",
        sigma_data: float = 1.0,
        ens_weight: float = 0.0,
        guide_norm: str = "none",
    ):
        super().__init__()
        if weighting not in {"sda", "pigdm"}:
            raise ValueError("weighting must be 'sda' or 'pigdm'")
        if not 0.0 <= ens_weight <= 1.0:
            raise ValueError("ens_weight must be in [0, 1]")
        if guide_norm not in {"none", "dsg"}:
            raise ValueError("guide_norm must be 'none' or 'dsg'")
        self.operator = operator
        self.decode_fn = decode_fn
        self.scale = scale
        self.meas_std = meas_std
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.encode_fn = encode_fn
        self.glue_weight = glue_weight
        self.weighting = weighting
        self.sigma_data = sigma_data
        self.ens_weight = ens_weight
        self.guide_norm = guide_norm
        self.last_repulsion_stats = None
        if glue_weight > 0 and encode_fn is None:
            raise ValueError("glue_weight > 0 requires encode_fn")

    def _variance(self, sigma, mu=1.0):
        if self.weighting == "pigdm":
            # piGDM/OT-ODE: r_t^2 = sigma^2 sigma_d^2 / (sigma^2 + sigma_d^2),
            # the posterior variance of x0 given x_t for a unit-variance prior.
            sig2 = (sigma / mu) ** 2
            r2 = sig2 * self.sigma_data**2 / (sig2 + self.sigma_data**2)
            return self.meas_std**2 + self.gamma * r2
        return self.meas_std**2 + self.gamma * (sigma / mu) ** 2

    def data_loss(self, x_0_hat, measurement):
        """Unweighted data misfit ||y - A(D(x0hat))||^2 (for hard data consistency)."""
        decoded = self.decode_fn(x_0_hat)
        pred_obs = self.operator.forward(decoded)
        return ((measurement - pred_obs) ** 2).sum()

    def _repulsion_objective(
        self,
        decoded,
        bandwidth: float,
        spatial_pool: int,
        unobserved_only: bool,
    ):
        """RBF particle repulsion in decoded, standardized weather space.

        ``decoded`` is produced from the denoised latent estimate.  Pooling is
        used only for the pairwise distance calculation, keeping the K=4
        smoke test memory bounded while gradients still flow to every latent.
        """
        if decoded.shape[0] < 2:
            raise ValueError("repulsion requires at least two particles")

        features = decoded
        observed = sorted(set(getattr(self.operator, "obs_frames", ())))
        if unobserved_only and observed:
            keep = torch.ones(features.shape[2], dtype=torch.bool, device=features.device)
            keep[observed] = False
            if not torch.any(keep):
                raise ValueError("repulsion mask contains no unobserved source frames")
            features = features[:, :, keep]

        if spatial_pool > 1:
            features = F.avg_pool3d(
                features,
                kernel_size=(1, spatial_pool, spatial_pool),
                stride=(1, spatial_pool, spatial_pool),
            )
        flat = features.float().flatten(1)

        distances = []
        for left in range(flat.shape[0]):
            for right in range(left + 1, flat.shape[0]):
                distances.append(torch.mean((flat[left] - flat[right]) ** 2))
        pairwise_mse = torch.stack(distances)

        if bandwidth > 0:
            bandwidth_used = pairwise_mse.new_tensor(float(bandwidth))
        else:
            # A detached median implements the usual adaptive RBF bandwidth
            # without allowing the bandwidth itself to cancel the force.
            bandwidth_used = torch.sqrt(
                torch.median(pairwise_mse.detach()).clamp_min(1e-12)
            )
        kernels = torch.exp(
            -pairwise_mse / (2.0 * bandwidth_used.square().clamp_min(1e-12))
        )
        # Gradient ascent on the negative similarity pushes nearby particles
        # apart; the pretrained prior still supplies the base ODE velocity.
        objective = -kernels.mean()
        stats = {
            "pairwise_mse": float(pairwise_mse.detach().mean()),
            "bandwidth": float(bandwidth_used.detach()),
            "kernel": float(kernels.detach().mean()),
        }
        return objective, stats

    def conditioning(
        self,
        x_prev,
        x_t,
        x_0_hat,
        measurement,
        sigma,
        mu=1.0,
        repulsion_strength: float = 0.0,
        repulsion_bandwidth: float = 0.0,
        repulsion_spatial_pool: int = 8,
        repulsion_unobserved_only: bool = True,
        repulsion_grad_norm: str = "rms",
        **kwargs,
    ):
        grad_input = x_prev if (x_prev is not None and x_prev.requires_grad) else x_0_hat

        decoded = self.decode_fn(x_0_hat)
        pred_obs = self.operator.forward(decoded)
        err = (measurement - pred_obs) ** 2
        var = self._variance(sigma, mu)

        if self.ens_weight > 0 and pred_obs.shape[0] > 1:
            # Ensemble-mean guidance: couple the batch by additionally fitting
            # the BATCH-MEAN prediction to the data. This targets the actual
            # evaluation estimator (the ensemble mean over posterior draws):
            # log p = -[(1-w) sum_b err_b + w B err_mean] / (2 var).
            B = pred_obs.shape[0]
            err_mean = (measurement[:1] - pred_obs.mean(dim=0, keepdim=True)) ** 2
            log_p = -(
                (1.0 - self.ens_weight) * (err / var).sum()
                + self.ens_weight * B * (err_mean / var).sum()
            ) / 2
        else:
            log_p = -(err / var).sum() / 2

        if self.glue_weight > 0:
            glue = (x_0_hat - self.encode_fn(decoded)) ** 2
            log_p = log_p - self.glue_weight * glue.sum() / 2

        self.last_repulsion_stats = None
        if repulsion_strength > 0:
            if repulsion_spatial_pool < 1:
                raise ValueError("repulsion_spatial_pool must be at least 1")
            if repulsion_grad_norm not in {"none", "rms"}:
                raise ValueError("repulsion_grad_norm must be 'none' or 'rms'")
            err_value = err.mean().item()
            decoder_only = grad_input is x_0_hat
            if decoder_only:
                # K=4 on a 40-GB A100 cannot retain the full decoder graph for
                # both backward passes.  Finish and release the observation
                # graph, then recompute the decoder for the repulsive gradient.
                # This preserves both gradients and trades a few extra decoder
                # forwards for substantially lower peak memory.
                data_grad = torch.autograd.grad(
                    outputs=log_p,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                del decoded, pred_obs, err, log_p
                decoded_repulsion = self.decode_fn(x_0_hat)
                repulsion_objective, stats = self._repulsion_objective(
                    decoded_repulsion,
                    bandwidth=repulsion_bandwidth,
                    spatial_pool=repulsion_spatial_pool,
                    unobserved_only=repulsion_unobserved_only,
                )
                repulsion_grad = torch.autograd.grad(
                    outputs=repulsion_objective,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            else:
                # Full DPS differentiates through the denoiser, so both terms
                # must share its graph.  Use an 80-GB GPU for K=4 in this mode.
                repulsion_objective, stats = self._repulsion_objective(
                    decoded,
                    bandwidth=repulsion_bandwidth,
                    spatial_pool=repulsion_spatial_pool,
                    unobserved_only=repulsion_unobserved_only,
                )
                data_grad = torch.autograd.grad(
                    outputs=log_p,
                    inputs=grad_input,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                repulsion_grad = torch.autograd.grad(
                    outputs=repulsion_objective,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            if repulsion_grad_norm == "rms":
                dims = tuple(range(1, repulsion_grad.ndim))
                rms = torch.sqrt(torch.mean(repulsion_grad.square(), dim=dims, keepdim=True))
                repulsion_grad = repulsion_grad / rms.clamp_min(1e-12)
            if self.guide_norm == "dsg":
                flat = data_grad.flatten(1)
                norm = flat.norm(dim=1).clamp_min(1e-12).view(
                    -1, *([1] * (data_grad.ndim - 1))
                )
                data_grad = data_grad / norm * (data_grad[0].numel() ** 0.5)
            if self.grad_clip is not None:
                data_grad = torch.clamp(data_grad, -self.grad_clip, self.grad_clip)

            scaled_score = (
                self.scale * data_grad + repulsion_strength * repulsion_grad
            ) * sigma
            stats["strength"] = float(repulsion_strength)
            stats["gradient_rms"] = float(
                torch.sqrt(torch.mean(repulsion_grad.detach().square()))
            )
            self.last_repulsion_stats = stats
            return scaled_score, err_value

        # Preserve the original path exactly when repulsion is disabled.
        grad = torch.autograd.grad(
            outputs=log_p, inputs=grad_input, retain_graph=False, create_graph=False
        )[0]
        if self.guide_norm == "dsg":
            # DSG-style spherical normalization (arXiv:2402.03201, simplified):
            # unit-normalize the gradient per batch element and rescale by
            # sqrt(n), removing the loss-magnitude (step-size) sensitivity.
            flat = grad.flatten(1)
            norm = flat.norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (grad.ndim - 1)))
            grad = grad / norm * (grad[0].numel() ** 0.5)
        if self.grad_clip is not None:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)

        scaled_score = grad * self.scale * sigma
        return scaled_score, err.mean().item()

    def bp_latent(self, x_0_hat, measurement, weight=1.0):
        """Back-projection + re-encode hard consistency (DDNM/CM4IR-flavored).

        Decode the Tweedie latent, correct the observed frames with the
        pseudo-inverse residual, and re-encode onto the latent manifold.
        Requires encode_fn. Cheap: one decoder + one encoder forward.
        """
        if self.encode_fn is None:
            raise ValueError("bp_latent requires encode_fn")
        with torch.no_grad():
            decoded = self.decode_fn(x_0_hat)
            corrected = self.operator.back_project(decoded, measurement, weight)
            return self.encode_fn(corrected)


class IGRAVideoOperator:
    """Sparse radiosonde (IGRA) point observations of selected frames of a video.

    Observations live at station ``[lat, lon]`` locations (lat in [-90, 90],
    lon in [-180, 180]) and are interpolated from the standardized field with
    bilinear ``grid_sample`` (vendored from igra_gen's IGRAOperator: the field
    is rolled by half a longitude circle because ERA5 longitudes start at 0).
    """

    def __init__(self, obs_frames, channel_map=None, mode="bilinear"):
        super().__init__()
        self.obs_frames = [int(f) for f in obs_frames]
        # channel_map[i] = video channel index for measurement channel i
        # (the IGRA pickle typically carries a 13-variable subset of the 69
        # ERA5 variables). None = identity.
        self.channel_map = None if channel_map is None else [int(c) for c in channel_map]
        self.mode = mode

    def H(self, x: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """Interpolate a (H, W) field at (N, 2) [lat, lon] query points."""
        x = torch.roll(x, shifts=x.shape[1] // 2, dims=1)
        if query_points.shape[0] == 0:
            return torch.zeros(0, device=x.device, dtype=torch.float32)
        norm_lat = 2 * (query_points[:, 0] + 90) / 180 - 1
        norm_lon = 2 * (query_points[:, 1] + 180) / 360 - 1
        grid = (
            torch.stack((norm_lon, norm_lat), dim=-1)
            .to(dtype=torch.float32)
            .view(1, -1, 1, 2)
        )
        x = x.unsqueeze(0).unsqueeze(0)
        vals = F.grid_sample(
            x.float(), grid, mode=self.mode, align_corners=True,
            padding_mode="reflection",
        )
        return vals.squeeze(0).squeeze(0).squeeze(-1)

    def error_function(self, video, query_locations, true_values):
        """Mean squared misfit at the stations over the observed frames.

        - video: (B, C, T, H, W) decoded standardized fields.
        - query_locations / true_values: nested lists indexed
          [frame_position][channel] -> array (N, 2) / (N,), shared across the
          batch. ``frame_position`` runs over ``self.obs_frames``.
        """
        device = video.device
        error = video.new_zeros(())
        counter = 1.0
        for b in range(video.shape[0]):
            for fi, t in enumerate(self.obs_frames):
                q_frame = query_locations[fi]
                v_frame = true_values[fi]
                for c in range(len(v_frame)):
                    vals = torch.as_tensor(
                        np.asarray(v_frame[c]), device=device, dtype=torch.float32
                    )
                    if vals.numel() == 0:
                        continue
                    queries = torch.as_tensor(
                        np.asarray(q_frame[c]), device=device, dtype=torch.float32
                    )
                    video_ch = c if self.channel_map is None else self.channel_map[c]
                    interp = self.H(video[b, video_ch, t], queries)
                    error = error + torch.mean((interp - vals) ** 2)
                    counter += 1
        return error / counter


class DenseFrameConditionOperator:
    """Dense full-grid observation of selected frames of a video.

    Used to chain forecast windows autoregressively: the previous window's
    generated trailing frame(s) act as the (dense, all-channel) initial
    condition of the next window. ``error_function`` matches the
    :class:`IGRAVideoOperator` interface so it plugs into
    :class:`IGRAVideoConditioning` unchanged; the measurement is
    ``[ic_fields, None]`` with ``ic_fields`` a (F, C, H, W) tensor of
    standardized fields, one per entry of ``obs_frames``.
    """

    def __init__(self, obs_frames):
        super().__init__()
        self.obs_frames = [int(f) for f in obs_frames]

    def error_function(self, video, ic_fields, _unused=None):
        # ic_fields: (F, C, H, W) shared across the batch, or (B, F, C, H, W)
        # with one IC per ensemble member (batch-dim chaining).
        ic = ic_fields.to(device=video.device, dtype=video.dtype)
        error = video.new_zeros(())
        for fi, t in enumerate(self.obs_frames):
            tgt = ic[:, fi] if ic.ndim == 5 else ic[fi]
            error = error + ((video[:, :, t] - tgt) ** 2).mean()
        return error / len(self.obs_frames)


class IGRAVideoConditioning:
    """DPS/SDA-style sparse-observation and particle-repulsion guidance.

    Same structure as :class:`LatentVideoConditioning`, but the data misfit is
    the station-interpolated error of :class:`IGRAVideoOperator` (already a
    mean over points/channels/frames, matching igra_gen's convention).  The
    optional repulsive score is evaluated in decoded standardized weather
    space and, by default, only on source frames without observations.
    """

    def __init__(
        self,
        operator: IGRAVideoOperator,
        decode_fn,
        scale: float = 1.0,
        meas_std: float = 1e-2,
        gamma: float = 1e-2,
        grad_clip: float | None = 1.0,
        guide_norm: str = "none",
        hurricane_descriptor_config: dict | None = None,
    ):
        super().__init__()
        if guide_norm not in {"none", "dsg"}:
            raise ValueError("guide_norm must be 'none' or 'dsg'")
        self.operator = operator
        self.decode_fn = decode_fn
        self.scale = scale
        self.meas_std = meas_std
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.guide_norm = guide_norm
        self.last_repulsion_stats = None
        self._repulsion_mask_cache = {}
        self.hurricane_descriptor_config = hurricane_descriptor_config
        self._hurricane_tensor_cache = {}

    def data_loss(self, x_0_hat, measurement):
        """Differentiable station misfit (for RED-diff / hard consistency)."""
        decoded = self.decode_fn(x_0_hat)
        query_locations, true_values = measurement
        return self.operator.error_function(decoded, query_locations, true_values)

    def _observation_complement_mask(
        self,
        decoded,
        query_locations,
        observation_buffer: int,
    ):
        """Return a channel/time/space mask that is zero at observed pixels.

        IGRA values are sampled with bilinear interpolation after rolling the
        ERA5 longitude axis by half a globe.  Each station therefore depends on
        at most four decoded grid cells.  Those cells, plus an optional grid-cell
        buffer, are excluded only for the observed variable and time.  Every
        other variable, pixel, and frame remains eligible for repulsion.
        """
        if observation_buffer < 0:
            raise ValueError("repulsion_observation_buffer must be non-negative")

        _, channels, timesteps, height, width = decoded.shape
        cache_key = (
            id(query_locations), channels, timesteps, height, width,
            str(decoded.device), int(observation_buffer),
        )
        cached = self._repulsion_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = torch.ones(
            (1, channels, timesteps, height, width),
            device=decoded.device,
            dtype=torch.float32,
        )
        for frame_position, source_frame in enumerate(self.operator.obs_frames):
            if frame_position >= len(query_locations):
                break
            if source_frame < 0 or source_frame >= timesteps:
                continue
            locations_by_variable = query_locations[frame_position]
            for measurement_channel, locations in enumerate(locations_by_variable):
                points = np.asarray(locations, dtype=np.float32)
                if points.size == 0:
                    continue
                points = points.reshape(-1, 2)
                video_channel = (
                    measurement_channel
                    if self.operator.channel_map is None
                    else self.operator.channel_map[measurement_channel]
                )
                if video_channel < 0 or video_channel >= channels:
                    continue

                latitude = torch.as_tensor(
                    points[:, 0], device=decoded.device, dtype=torch.float32
                ).clamp(-90.0, 90.0)
                longitude = torch.as_tensor(
                    points[:, 1], device=decoded.device, dtype=torch.float32
                ).clamp(-180.0, 180.0)
                # Match IGRAVideoOperator.H: normalized [-180, 180] longitude
                # on the half-globe-rolled field with align_corners=True.
                row = (latitude + 90.0) / 180.0 * (height - 1)
                col_rolled = (longitude + 180.0) / 360.0 * (width - 1)
                row_candidates = (torch.floor(row).long(), torch.ceil(row).long())
                col_candidates = (
                    torch.floor(col_rolled).long(), torch.ceil(col_rolled).long()
                )

                for base_row in row_candidates:
                    for base_col_rolled in col_candidates:
                        base_col = torch.remainder(base_col_rolled - width // 2, width)
                        for delta_row in range(-observation_buffer, observation_buffer + 1):
                            marked_row = (base_row + delta_row).clamp(0, height - 1)
                            for delta_col in range(
                                -observation_buffer, observation_buffer + 1
                            ):
                                marked_col = torch.remainder(base_col + delta_col, width)
                                mask[
                                    0,
                                    video_channel,
                                    source_frame,
                                    marked_row,
                                    marked_col,
                                ] = 0.0

        if not torch.any(mask > 0):
            raise ValueError("observation-complement repulsion mask is empty")
        self._repulsion_mask_cache[cache_key] = mask
        return mask

    @staticmethod
    def _rbf_from_flat(flat, bandwidth: float, flat_weights=None):
        """Negative mean RBF similarity for a (particles, features) tensor."""
        distances = []
        for left in range(flat.shape[0]):
            for right in range(left + 1, flat.shape[0]):
                squared_difference = (flat[left] - flat[right]) ** 2
                if flat_weights is None:
                    distances.append(torch.mean(squared_difference))
                else:
                    distances.append(
                        torch.sum(squared_difference * flat_weights)
                        / flat_weights.sum().clamp_min(1e-12)
                    )
        pairwise_mse = torch.stack(distances)
        if bandwidth > 0:
            bandwidth_used = pairwise_mse.new_tensor(float(bandwidth))
        else:
            bandwidth_used = torch.sqrt(
                torch.median(pairwise_mse.detach()).clamp_min(1e-12)
            )
        kernels = torch.exp(
            -pairwise_mse / (2.0 * bandwidth_used.square().clamp_min(1e-12))
        )
        return -kernels.mean(), {
            "pairwise_mse": float(pairwise_mse.detach().mean()),
            "bandwidth": float(bandwidth_used.detach()),
            "kernel": float(kernels.detach().mean()),
        }

    def _hurricane_tensors(self, decoded):
        """Materialize static hurricane geometry/normalization on the active device."""
        config = self.hurricane_descriptor_config
        if config is None:
            raise ValueError(
                "hurricane repulsion requires hurricane_descriptor_config"
            )
        key = (
            str(decoded.device),
            str(decoded.dtype),
            decoded.shape[-2],
            decoded.shape[-1],
        )
        cached = self._hurricane_tensor_cache.get(key)
        if cached is not None:
            return cached

        latitude = torch.as_tensor(
            np.asarray(config["latitude"]).squeeze(),
            device=decoded.device,
            dtype=torch.float32,
        )
        longitude = torch.as_tensor(
            np.asarray(config["longitude"]).squeeze(),
            device=decoded.device,
            dtype=torch.float32,
        )
        longitude = torch.remainder(longitude + 180.0, 360.0) - 180.0
        if latitude.numel() != decoded.shape[-2] or longitude.numel() != decoded.shape[-1]:
            raise ValueError(
                "hurricane latitude/longitude dimensions do not match decoded fields"
            )
        lat_min, lat_max = config["region_lat"]
        lon_min, lon_max = config["region_lon"]
        iy = torch.nonzero(
            (latitude >= lat_min) & (latitude <= lat_max), as_tuple=False
        ).squeeze(-1)
        ix = torch.nonzero(
            (longitude >= lon_min) & (longitude <= lon_max), as_tuple=False
        ).squeeze(-1)
        if iy.numel() == 0 or ix.numel() == 0:
            raise ValueError("hurricane descriptor region is outside the model grid")
        lat_grid = latitude[iy][:, None].expand(iy.numel(), ix.numel()).reshape(-1)
        lon_grid = longitude[ix][None, :].expand(iy.numel(), ix.numel()).reshape(-1)
        cached = {
            "iy": iy,
            "ix": ix,
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
        }
        self._hurricane_tensor_cache[key] = cached
        return cached

    def _hurricane_descriptors(self, decoded):
        """Differentiable Laura center/intensity descriptors on forecast frames."""
        config = self.hurricane_descriptor_config
        tensors = self._hurricane_tensors(decoded)
        iy, ix = tensors["iy"], tensors["ix"]
        latitude, longitude = tensors["lat_grid"], tensors["lon_grid"]
        observed = sorted(set(getattr(self.operator, "obs_frames", ())))
        forecast_frames = [
            frame for frame in range(decoded.shape[2]) if frame not in observed
        ]
        horizon = int(config.get("horizon", 0))
        if horizon > 0:
            forecast_frames = forecast_frames[:horizon]
        if not forecast_frames:
            raise ValueError("hurricane repulsion contains no forecast frames")

        first_forecast = forecast_frames[0]
        earlier_observed = [frame for frame in observed if frame < first_forecast]
        anchor_frame = max(earlier_observed) if earlier_observed else None
        mslp_channel = int(config["mslp_channel"])
        u10_channel = int(config["u10_channel"])
        v10_channel = int(config["v10_channel"])
        means = config["weather_means"]
        stds = config["weather_stds"]
        pressure_temperature = float(config["pressure_temperature_hpa"])
        wind_temperature = float(config["wind_temperature_ms"])
        track_radius = float(config["track_radius_deg"])

        def regional_fields(frame):
            pressure = (
                decoded[:, mslp_channel, frame][:, iy][:, :, ix].float()
                * float(stds[0])
                + float(means[0])
            ) * 0.01
            u10 = (
                decoded[:, u10_channel, frame][:, iy][:, :, ix].float()
                * float(stds[1])
                + float(means[1])
            )
            v10 = (
                decoded[:, v10_channel, frame][:, iy][:, :, ix].float()
                * float(stds[2])
                + float(means[2])
            )
            return pressure.flatten(1), torch.sqrt(
                u10.flatten(1).square() + v10.flatten(1).square() + 1e-8
            )

        def diagnose(frame, previous_center=None):
            pressure, wind = regional_fields(frame)
            pressure_logits = -(
                pressure - pressure.detach().amin(dim=1, keepdim=True)
            ) / pressure_temperature
            if previous_center is not None:
                previous_lat, previous_lon = previous_center
                cos_lat = torch.cos(torch.deg2rad(previous_lat)).unsqueeze(1)
                distance2 = (
                    (latitude.unsqueeze(0) - previous_lat.unsqueeze(1)).square()
                    + (
                        (longitude.unsqueeze(0) - previous_lon.unsqueeze(1))
                        * cos_lat
                    ).square()
                )
                pressure_logits = pressure_logits - distance2 / (
                    2.0 * track_radius**2
                )
            pressure_weight = torch.softmax(pressure_logits, dim=1)
            center_lat = torch.sum(pressure_weight * latitude.unsqueeze(0), dim=1)
            center_lon = torch.sum(pressure_weight * longitude.unsqueeze(0), dim=1)
            central_pressure = torch.sum(pressure_weight * pressure, dim=1)

            cos_lat = torch.cos(torch.deg2rad(center_lat)).unsqueeze(1)
            distance2 = (
                (latitude.unsqueeze(0) - center_lat.unsqueeze(1)).square()
                + (
                    (longitude.unsqueeze(0) - center_lon.unsqueeze(1)) * cos_lat
                ).square()
            )
            wind_logits = (
                wind - wind.detach().amax(dim=1, keepdim=True)
            ) / wind_temperature - distance2 / (2.0 * track_radius**2)
            wind_weight = torch.softmax(wind_logits, dim=1)
            maximum_wind = torch.sum(wind_weight * wind, dim=1)
            return center_lat, center_lon, central_pressure, maximum_wind

        previous_center = None
        if anchor_frame is not None:
            anchor = diagnose(anchor_frame)
            previous_center = (anchor[0], anchor[1])

        raw = []
        for frame in forecast_frames:
            descriptor = diagnose(frame, previous_center=previous_center)
            raw.append(torch.stack(descriptor, dim=-1))
            previous_center = (descriptor[0], descriptor[1])
        raw = torch.stack(raw, dim=1)
        scaled = torch.stack(
            (
                raw[..., 0] / float(config["position_scale_deg"]),
                raw[..., 1] / float(config["position_scale_deg"]),
                raw[..., 2] / float(config["pressure_scale_hpa"]),
                raw[..., 3] / float(config["wind_scale_ms"]),
            ),
            dim=-1,
        )
        return scaled, raw, forecast_frames

    def _hurricane_repulsion_objective(self, decoded, bandwidth: float):
        scaled, raw, forecast_frames = self._hurricane_descriptors(decoded)
        objective, stats = self._rbf_from_flat(scaled.float().flatten(1), bandwidth)
        track_distances = []
        pressure_distances = []
        wind_distances = []
        for left in range(raw.shape[0]):
            for right in range(left + 1, raw.shape[0]):
                mean_lat = 0.5 * (raw[left, :, 0] + raw[right, :, 0])
                dx = (raw[left, :, 1] - raw[right, :, 1]) * torch.cos(
                    torch.deg2rad(mean_lat)
                )
                dy = raw[left, :, 0] - raw[right, :, 0]
                track_distances.append(111.0 * torch.sqrt(dx.square() + dy.square()))
                pressure_distances.append(
                    torch.abs(raw[left, :, 2] - raw[right, :, 2])
                )
                wind_distances.append(torch.abs(raw[left, :, 3] - raw[right, :, 3]))
        return objective, {
            **stats,
            "mask_mode": "hurricane",
            "active_fraction": float(len(forecast_frames) / decoded.shape[2]),
            "observation_buffer": 0,
            "split_beta": 0.0,
            "descriptor_frames": forecast_frames,
            "descriptor_track_distance_km": float(
                torch.cat(track_distances).detach().mean()
            ),
            "descriptor_pressure_difference_hpa": float(
                torch.cat(pressure_distances).detach().mean()
            ),
            "descriptor_wind_difference_ms": float(
                torch.cat(wind_distances).detach().mean()
            ),
        }

    def _rbf_repulsion_component(
        self,
        features,
        bandwidth: float,
        spatial_pool: int,
        weights=None,
    ):
        """Return one independently normalized negative-RBF component."""
        if weights is not None and not torch.any(weights > 0):
            raise ValueError("repulsion component mask is empty")
        if spatial_pool > 1:
            pool_kwargs = {
                "kernel_size": (1, spatial_pool, spatial_pool),
                "stride": (1, spatial_pool, spatial_pool),
            }
            if weights is None:
                features = F.avg_pool3d(features, **pool_kwargs)
            else:
                pooled_weights = F.avg_pool3d(weights, **pool_kwargs)
                features = F.avg_pool3d(features * weights, **pool_kwargs)
                features = features / pooled_weights.clamp_min(1e-12)
                weights = pooled_weights
        flat = features.float().flatten(1)
        flat_weights = None if weights is None else weights.float().flatten()
        objective, stats = self._rbf_from_flat(
            flat, bandwidth, flat_weights=flat_weights
        )
        return objective, {
            **stats,
            "active_fraction": (
                1.0 if weights is None else float(weights.mean().detach())
            ),
        }

    def _repulsion_objective(
        self,
        decoded,
        bandwidth: float,
        spatial_pool: int,
        mask_mode: str,
        measurement=None,
        observation_buffer: int = 0,
        split_beta: float = 1.0,
    ):
        """Negative RBF similarity between decoded posterior particles.

        ``split`` keeps the original forecast-frame objective and adds a second,
        independently normalized objective over the unobserved pixels and
        variables of observed frames::

            R = R_forecast + split_beta * R_observed_complement

        This prevents the sparse observed-frame contribution from being diluted
        by one pooled full-video objective.
        """
        if decoded.shape[0] < 2:
            raise ValueError("repulsion requires at least two particles")
        if mask_mode not in {"frames", "spatial", "split", "hurricane", "all"}:
            raise ValueError(
                "repulsion_mask_mode must be 'frames', 'spatial', 'split', "
                "'hurricane', or 'all'"
            )
        if split_beta < 0:
            raise ValueError("repulsion_split_beta must be non-negative")
        if mask_mode == "hurricane":
            return self._hurricane_repulsion_objective(decoded, bandwidth)

        observed = sorted(set(getattr(self.operator, "obs_frames", ())))
        observed = [frame for frame in observed if 0 <= frame < decoded.shape[2]]
        keep_forecast = torch.ones(
            decoded.shape[2], dtype=torch.bool, device=decoded.device
        )
        if observed:
            keep_forecast[observed] = False

        if mask_mode in {"frames", "split"}:
            if not torch.any(keep_forecast):
                raise ValueError("repulsion mask contains no forecast source frames")
            forecast_objective, forecast_stats = self._rbf_repulsion_component(
                decoded[:, :, keep_forecast],
                bandwidth=bandwidth,
                spatial_pool=spatial_pool,
            )
            forecast_stats["active_fraction"] = float(keep_forecast.float().mean())
            if mask_mode == "frames":
                return forecast_objective, {
                    **forecast_stats,
                    "mask_mode": mask_mode,
                    "observation_buffer": int(observation_buffer),
                    "split_beta": 0.0,
                }

        if mask_mode in {"spatial", "split"}:
            if measurement is None:
                raise ValueError(
                    f"{mask_mode} repulsion masking requires the observations"
                )
            query_locations, _ = measurement
            complement_weights = self._observation_complement_mask(
                decoded,
                query_locations=query_locations,
                observation_buffer=observation_buffer,
            )
            if mask_mode == "spatial":
                objective, stats = self._rbf_repulsion_component(
                    decoded,
                    bandwidth=bandwidth,
                    spatial_pool=spatial_pool,
                    weights=complement_weights,
                )
                return objective, {
                    **stats,
                    "mask_mode": mask_mode,
                    "observation_buffer": int(observation_buffer),
                    "split_beta": 0.0,
                }

            if not observed:
                raise ValueError("split repulsion requires at least one observed frame")
            observed_selector = torch.zeros(
                decoded.shape[2], device=decoded.device, dtype=torch.float32
            )
            observed_selector[observed] = 1.0
            observed_weights = complement_weights * observed_selector.view(
                1, 1, -1, 1, 1
            )
            observed_objective, observed_stats = self._rbf_repulsion_component(
                decoded,
                bandwidth=bandwidth,
                spatial_pool=spatial_pool,
                weights=observed_weights,
            )
            objective = forecast_objective + split_beta * observed_objective
            normalizer = 1.0 + split_beta
            stats = {
                "pairwise_mse": (
                    forecast_stats["pairwise_mse"]
                    + split_beta * observed_stats["pairwise_mse"]
                ) / normalizer,
                "bandwidth": (
                    forecast_stats["bandwidth"]
                    + split_beta * observed_stats["bandwidth"]
                ) / normalizer,
                "kernel": (
                    forecast_stats["kernel"]
                    + split_beta * observed_stats["kernel"]
                ) / normalizer,
                "mask_mode": mask_mode,
                "active_fraction": observed_stats["active_fraction"],
                "forecast_pairwise_mse": forecast_stats["pairwise_mse"],
                "observed_pairwise_mse": observed_stats["pairwise_mse"],
                "forecast_bandwidth": forecast_stats["bandwidth"],
                "observed_bandwidth": observed_stats["bandwidth"],
                "forecast_kernel": forecast_stats["kernel"],
                "observed_kernel": observed_stats["kernel"],
                "forecast_active_fraction": forecast_stats["active_fraction"],
                "observed_active_fraction": observed_stats["active_fraction"],
                "split_beta": float(split_beta),
                "observation_buffer": int(observation_buffer),
            }
            return objective, stats

        objective, stats = self._rbf_repulsion_component(
            decoded,
            bandwidth=bandwidth,
            spatial_pool=spatial_pool,
        )
        return objective, {
            **stats,
            "mask_mode": mask_mode,
            "observation_buffer": int(observation_buffer),
            "split_beta": 0.0,
        }

    def conditioning(
        self,
        x_prev,
        x_t,
        x_0_hat,
        measurement,
        sigma,
        mu=1.0,
        repulsion_strength: float = 0.0,
        repulsion_bandwidth: float = 0.0,
        repulsion_spatial_pool: int = 8,
        repulsion_unobserved_only: bool = True,
        repulsion_mask_mode: str | None = None,
        repulsion_observation_buffer: int = 0,
        repulsion_split_beta: float = 1.0,
        repulsion_grad_norm: str = "rms",
        **kwargs,
    ):
        grad_input = x_prev if (x_prev is not None and x_prev.requires_grad) else x_0_hat

        decoded = self.decode_fn(x_0_hat)
        query_locations, true_values = measurement
        err = self.operator.error_function(decoded, query_locations, true_values)
        var = self.meas_std**2 + self.gamma * (sigma / mu) ** 2
        log_p = -(err / var) / 2

        self.last_repulsion_stats = None
        if repulsion_strength > 0:
            if repulsion_spatial_pool < 1:
                raise ValueError("repulsion_spatial_pool must be at least 1")
            if repulsion_grad_norm not in {"none", "rms"}:
                raise ValueError("repulsion_grad_norm must be 'none' or 'rms'")
            mask_mode = repulsion_mask_mode
            if mask_mode is None:
                mask_mode = "frames" if repulsion_unobserved_only else "all"

            err_value = err.item()
            decoder_only = grad_input is x_0_hat
            if decoder_only:
                # Release the observation graph before rebuilding the decoder
                # graph for the repulsive score.  This is the K=8/80-GB path.
                data_grad = torch.autograd.grad(
                    outputs=log_p,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                del decoded, err, log_p
                decoded_repulsion = self.decode_fn(x_0_hat)
                repulsion_objective, stats = self._repulsion_objective(
                    decoded_repulsion,
                    bandwidth=repulsion_bandwidth,
                    spatial_pool=repulsion_spatial_pool,
                    mask_mode=mask_mode,
                    measurement=measurement,
                    observation_buffer=repulsion_observation_buffer,
                    split_beta=repulsion_split_beta,
                )
                repulsion_grad = torch.autograd.grad(
                    outputs=repulsion_objective,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]
            else:
                # Full DPS shares the denoiser graph between the likelihood and
                # repulsion terms.  This path is substantially more memory hungry.
                repulsion_objective, stats = self._repulsion_objective(
                    decoded,
                    bandwidth=repulsion_bandwidth,
                    spatial_pool=repulsion_spatial_pool,
                    mask_mode=mask_mode,
                    measurement=measurement,
                    observation_buffer=repulsion_observation_buffer,
                    split_beta=repulsion_split_beta,
                )
                data_grad = torch.autograd.grad(
                    outputs=log_p,
                    inputs=grad_input,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                repulsion_grad = torch.autograd.grad(
                    outputs=repulsion_objective,
                    inputs=grad_input,
                    retain_graph=False,
                    create_graph=False,
                )[0]

            if repulsion_grad_norm == "rms":
                dims = tuple(range(1, repulsion_grad.ndim))
                rms = torch.sqrt(
                    torch.mean(repulsion_grad.square(), dim=dims, keepdim=True)
                )
                repulsion_grad = repulsion_grad / rms.clamp_min(1e-12)
            if self.guide_norm == "dsg":
                flat = data_grad.flatten(1)
                norm = flat.norm(dim=1).clamp_min(1e-12).view(
                    -1, *([1] * (data_grad.ndim - 1))
                )
                data_grad = data_grad / norm * (data_grad[0].numel() ** 0.5)
            if self.grad_clip is not None:
                data_grad = torch.clamp(
                    data_grad, -self.grad_clip, self.grad_clip
                )

            scaled_score = (
                self.scale * data_grad + repulsion_strength * repulsion_grad
            ) * sigma
            stats["strength"] = float(repulsion_strength)
            stats["gradient_rms"] = float(
                torch.sqrt(torch.mean(repulsion_grad.detach().square()))
            )
            self.last_repulsion_stats = stats
            return scaled_score, err_value

        # Preserve the independent sampler path when repulsion is disabled.
        grad = torch.autograd.grad(
            outputs=log_p, inputs=grad_input, retain_graph=False, create_graph=False
        )[0]
        if self.guide_norm == "dsg":
            flat = grad.flatten(1)
            norm = flat.norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (grad.ndim - 1)))
            grad = grad / norm * (grad[0].numel() ** 0.5)
        if self.grad_clip is not None:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)

        scaled_score = grad * self.scale * sigma
        return scaled_score, err.item()


class SuperResolutionOperator:
    def __init__(self, in_shape=(32,32), target_shape=(256, 256), mode="bicubic"):
        super().__init__()
        self.scale = target_shape[0]//in_shape[0]
        self.up_sample = partial(F.interpolate, size=target_shape, mode=mode, align_corners=False)
        self.down_sample = partial(F.interpolate, size=in_shape, mode=mode, align_corners=False)
        # self.down_sample = Resizer(in_shape, 1/self.scale).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)  

    def transpose(self, data, **kwargs):
        return self.up_sample(data) 

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

class MaskingOperator:
    def __init__(self, mask, **kwargs):
        super().__init__()
        self.mask = mask

    def forward(self, data, **kwargs):
        return data * self.mask

    def transpose(self, data, **kwargs):
        return data * self.mask
