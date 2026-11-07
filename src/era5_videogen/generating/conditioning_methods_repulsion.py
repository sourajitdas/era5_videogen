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

    def data_loss(self, x_0_hat, measurement):
        """Differentiable station misfit (for RED-diff / hard consistency)."""
        decoded = self.decode_fn(x_0_hat)
        query_locations, true_values = measurement
        return self.operator.error_function(decoded, query_locations, true_values)

    def _repulsion_objective(
        self,
        decoded,
        bandwidth: float,
        spatial_pool: int,
        unobserved_only: bool,
    ):
        """Negative mean RBF similarity between decoded posterior particles."""
        if decoded.shape[0] < 2:
            raise ValueError("repulsion requires at least two particles")

        features = decoded
        observed = sorted(set(getattr(self.operator, "obs_frames", ())))
        if unobserved_only and observed:
            keep = torch.ones(
                features.shape[2], dtype=torch.bool, device=features.device
            )
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
        pairwise_mse = torch.stack(
            [
                torch.mean((flat[left] - flat[right]) ** 2)
                for left in range(flat.shape[0])
                for right in range(left + 1, flat.shape[0])
            ]
        )
        if bandwidth > 0:
            bandwidth_used = pairwise_mse.new_tensor(float(bandwidth))
        else:
            bandwidth_used = torch.sqrt(
                torch.median(pairwise_mse.detach()).clamp_min(1e-12)
            )
        kernels = torch.exp(
            -pairwise_mse / (2.0 * bandwidth_used.square().clamp_min(1e-12))
        )
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
                    unobserved_only=repulsion_unobserved_only,
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
