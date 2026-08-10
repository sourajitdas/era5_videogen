import numpy as np
import torch

from era5_videogen.generating.conditioning_methods import *


_AMP_DTYPE = None


def _amp_dtype():
    """bfloat16 where supported (Ampere+); float16 fallback for V100 (sm70),
    which has no native bf16 -- keeps the samplers runnable on any GPU."""
    global _AMP_DTYPE
    if _AMP_DTYPE is None:
        # is_bf16_supported() reports emulated support on sm70; key off the
        # compute capability instead (native bf16 needs Ampere, sm80+). NOTE:
        # V100s additionally lack a cuDNN conv3d engine for the AE decoder in
        # this torch build, so sampling jobs should request a100 regardless.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            _AMP_DTYPE = torch.float16
            print("amp: bf16 unsupported on this GPU; falling back to fp16", flush=True)
        else:
            _AMP_DTYPE = torch.bfloat16
    return _AMP_DTYPE

class DiffusionSampler:
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def edm_sampler(
        self,
        latents,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """Proposed EDM sampler (Algorithm 2)."""
        if num_steps < 2:
            raise ValueError("EDM sampling requires num_steps >= 2")

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next.detach()

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, condition).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.net(x_next, t_next, condition).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def _model_output(self, x, t, condition=None, auxiliary=None):
        out = self.net(x, t, condition, auxiliary)
        if isinstance(out, tuple):
            return out[0]
        return out

    @torch.no_grad()
    def trigflow_sampler(
        self,
        latents,
        condition=None,
        num_steps=18,
        sigma_data=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        use_pp=True,
        auxiliary=None,
    ):
        """Swift-style DPM sampler for TrigFlow velocity-prediction models.

        This mirrors ``swift.generating.diffusion.DiffusionSampler.dpm_solver``:
        an EDM-spaced sigma schedule is mapped to TrigFlow angles with
        ``t = atan(sigma / sigma_data)``, then a DPM/DPM++ rotation update is
        applied to the spherical path.
        """
        if num_steps < 2:
            raise ValueError("TrigFlow DPM sampling requires num_steps >= 2")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        if sigma_min <= 0 or sigma_max <= 0 or sigma_max <= sigma_min:
            raise ValueError("Expected 0 < sigma_min < sigma_max")

        device = latents.device
        dtype = torch.float32
        ramp = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        t_steps = torch.atan(sigmas / sigma_data)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_t = latents.to(dtype) * sigma_data
        t_prev = None
        pred_prev = None

        for step_idx in range(num_steps):
            s = t_steps[step_idx]
            t = t_steps[step_idx + 1]
            delta = s - t
            cos_delta = torch.cos(delta)
            sin_delta = torch.sin(delta)

            F_s = self._model_output(
                x_t / sigma_data,
                s.expand(x_t.shape[0]),
                condition,
                auxiliary,
            ).to(dtype)

            if use_pp:
                pred = torch.cos(s) * x_t - torch.sin(s) * sigma_data * F_s
                denom = torch.sin(s)
            else:
                pred = torch.sin(s) * x_t + torch.cos(s) * sigma_data * F_s
                denom = torch.cos(s)

            first_order_step = cos_delta * x_t - sin_delta * sigma_data * F_s
            if step_idx == 0 or step_idx == num_steps - 1:
                x_next = first_order_step
            else:
                logtan = lambda u: torch.log(torch.tan(torch.clamp(u, 1e-4, 1.569)))
                r_s = (logtan(s) - logtan(t_prev)) / (logtan(s) - logtan(t))
                correction = (sin_delta / (2 * r_s * torch.clamp(denom, min=1e-3))) * (
                    pred_prev - pred
                )
                x_next = first_order_step + (correction if use_pp else -correction)

            t_prev = s
            pred_prev = pred
            x_t = x_next

        return x_t

class PosteriorDiffusionSampler:
    def __init__(self, net, conditioning=None):
        super().__init__()
        self.net = net
        if conditioning is None:
            # Legacy default: pixel-space 2D super-resolution conditioning.
            sr = SuperResolutionOperator(in_shape=(32, 32), target_shape=(256, 256))
            conditioning = ConditioningMethod(sr)
        self.cm = conditioning

    def _model_output(self, x, t, condition=None, auxiliary=None):
        out = self.net(x, t, condition, auxiliary)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _langevin_corrector(
        self,
        x_t,
        s,
        sigma_data,
        measurement,
        condition,
        auxiliary,
        steps,
        snr,
        guide_through_net,
        noise_scale=1.0,
        randn_like=torch.randn_like,
    ):
        """SDA-style Langevin corrector at angle ``s`` (posterior score)."""
        device = x_t.device
        for _ in range(steps):
            x_cur = x_t.detach()
            if guide_through_net:
                x_cur.requires_grad_(True)
            with torch.set_grad_enabled(guide_through_net):
                with torch.autocast(device.type, enabled=(device.type == "cuda"), dtype=_amp_dtype()):
                    F_s = self._model_output(
                        x_cur / sigma_data, s.expand(x_t.shape[0]), condition, auxiliary
                    )
                F_s = F_s.to(torch.float32)
            with torch.set_grad_enabled(True):
                x0_src = x_cur if guide_through_net else x_cur.detach()
                x0_hat = torch.cos(s) * x0_src - torch.sin(s) * sigma_data * F_s
                if not guide_through_net:
                    x0_hat = x0_hat.detach().requires_grad_(True)
                likelihood_score, _ = self.cm.conditioning(
                    x_prev=x_cur if guide_through_net else None,
                    x_t=x_t,
                    x_0_hat=x0_hat,
                    measurement=measurement,
                    sigma=sigma_data * torch.tan(s),
                )
            with torch.no_grad():
                x_cur = x_cur.detach()
                # Marginal prior score: (cos(s) x0_hat - x) / (sin^2(s) sigma_d^2).
                prior_score = (torch.cos(s) * x0_hat.detach() - x_cur) / (
                    torch.sin(s) ** 2 * sigma_data**2
                )
                # likelihood_score is pre-scaled by scale*sigma; undo sigma to get a score.
                lik_score = likelihood_score / (sigma_data * torch.tan(s)).clamp(min=1e-8)
                score = prior_score + lik_score
                tau = (snr * torch.sin(s) * sigma_data) ** 2
                # noise_scale < 1 biases the corrector toward the posterior
                # mean (variance-scaled reverse diffusion, arXiv:2503.20297).
                x_t = x_cur + tau * score + noise_scale * torch.sqrt(2 * tau) * randn_like(x_cur)
        return x_t

    def pos_trigflow_sampler(
        self,
        latents,
        measurement,
        condition=None,
        num_steps=50,
        sigma_data=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        use_pp=True,
        auxiliary=None,
        guide_through_net=True,
        log_every=2,
        x_warm=None,
        sigma_start=None,
        corrector_steps=0,
        corrector_snr=0.1,
        guide_max_sigma=float("inf"),
        guide_min_sigma=0.0,
        guide_reuse=1,
        guide_momentum=0.0,
        corrector_max_sigma=float("inf"),
        corrector_min_sigma=0.0,
        corrector_steps2=0,
        corrector_max_sigma2=0.0,
        noise_scale=1.0,
        randn_like=torch.randn_like,
    ):
        """TrigFlow DPM/DPM++ sampler with DPS-style posterior guidance.

        Mirrors :meth:`DiffusionSampler.trigflow_sampler` (EDM-spaced sigmas
        mapped to angles ``t = atan(sigma/sigma_data)``, rotation update), and
        injects the measurement-likelihood score at every step.

        Math. With x_t = cos(t) x0 + sin(t) z, z ~ N(0, sigma_d^2 I), the
        Tweedie estimate is x0_hat = cos(t) x_t - sin(t) sigma_d F(x_t/sigma_d, t)
        and the marginal score satisfies E[z|x_t] = -sin(t) sigma_d^2 score.
        Replacing score -> score + grad log p(y|x_t) in the probability-flow
        ODE shifts the velocity by -sigma_d^2 tan(t) * grad, i.e. the guided
        network output is

            F_guided = F - sigma * grad log p(y|x_t),   sigma = sigma_d tan(t).

        The conditioning method returns ``likelihood_score = scale * sigma *
        grad``, so the first-order rotation step becomes

            x_next = cos(d) x_t - sin(d) sigma_d (F - likelihood_score)

        which for ``scale = 1`` integrates the exact guided PF-ODE.

        ``guide_through_net=False`` differentiates the likelihood only through
        the decoder (FlowChef/MPGD approximation), running the denoiser
        gradient-free — much cheaper in memory and usually as good.

        Optional extensions (each off by default):
        - ``x_warm`` + ``sigma_start``: SDEdit-style warm start. Sampling
          begins at ``sigma_start`` from a forward-diffused ``x_warm`` (e.g.
          the encoded, upsampled observations) instead of pure noise.
        - ``corrector_steps > 0``: SDA-style Langevin corrector steps on the
          posterior score after each solver step.
        """
        if num_steps < 2:
            raise ValueError("TrigFlow DPM sampling requires num_steps >= 2")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        if sigma_start is not None:
            sigma_max = min(sigma_max, float(sigma_start))
        if sigma_min <= 0 or sigma_max <= 0 or sigma_max <= sigma_min:
            raise ValueError("Expected 0 < sigma_min < sigma_max")

        device = latents.device
        dtype = torch.float32
        ramp = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        t_steps = torch.atan(sigmas / sigma_data)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        if x_warm is not None:
            # Forward-diffuse the warm state to the (possibly reduced) start
            # level. noise_scale < 1 under-noises the restart (2503.20297),
            # biasing the cycle toward the warm (posterior-mean-ish) state.
            t0 = t_steps[0]
            x_t = torch.cos(t0) * x_warm.to(dtype) + noise_scale * torch.sin(
                t0
            ) * sigma_data * randn_like(x_warm.to(dtype))
        else:
            x_t = latents.to(dtype) * sigma_data
        t_prev = None
        pred_prev = None
        reuse_score = None  # cached likelihood score for lazy/reused guidance

        for step_idx in range(num_steps):
            s = t_steps[step_idx]
            t = t_steps[step_idx + 1]
            delta = s - t
            cos_delta = torch.cos(delta)
            sin_delta = torch.sin(delta)
            sigma_s = sigma_data * torch.tan(s)
            sigma_s_f = float(sigma_s)

            # Guidance is only worth its (expensive) gradient inside a noise band.
            # Outside it we run a plain forward-only ODE step (no backward).
            guide_now = guide_min_sigma <= sigma_s_f <= guide_max_sigma
            # Lazy guidance: recompute the gradient every `guide_reuse` steps and
            # reuse the cached score in between (the likelihood score varies
            # slowly across adjacent noise levels).
            recompute = guide_now and (
                guide_reuse <= 1 or step_idx % guide_reuse == 0 or reuse_score is None
            )
            net_grad = guide_through_net and recompute

            x_cur = x_t.detach()
            if net_grad:
                x_cur.requires_grad_(True)

            with torch.set_grad_enabled(net_grad):
                with torch.autocast(device.type, enabled=(device.type == "cuda"), dtype=_amp_dtype()):
                    F_s = self._model_output(
                        x_cur / sigma_data,
                        s.expand(x_t.shape[0]),
                        condition,
                        auxiliary,
                    )
                F_s = F_s.to(dtype)

            # Tweedie / denoised estimate; differentiable wrt x_cur (full DPS)
            # or used directly as the gradient target (decoder-only guidance).
            if not guide_now:
                likelihood_score = torch.zeros_like(x_cur)
                err = 0.0
            elif not recompute:
                likelihood_score = reuse_score
                err = -1.0  # reused; not recomputed this step
            else:
                with torch.set_grad_enabled(True):
                    x0_src = x_cur if net_grad else x_cur.detach()
                    x0_hat = torch.cos(s) * x0_src - torch.sin(s) * sigma_data * F_s
                    if not net_grad:
                        x0_hat = x0_hat.detach().requires_grad_(True)

                    likelihood_score, err = self.cm.conditioning(
                        x_prev=x_cur if net_grad else None,
                        x_t=x_t,
                        x_0_hat=x0_hat,
                        measurement=measurement,
                        sigma=sigma_s,
                    )
                likelihood_score = likelihood_score.detach()
                # Momentum guidance: EMA over likelihood scores across steps.
                # Unlike lazy reuse (frozen gradient), this averages gradient
                # noise while still recomputing every step.
                if guide_momentum > 0 and reuse_score is not None:
                    likelihood_score = likelihood_score + guide_momentum * reuse_score
                reuse_score = likelihood_score.detach()

            with torch.no_grad():
                x_cur = x_cur.detach()
                F_det = F_s.detach()

                if use_pp:
                    pred = torch.cos(s) * x_cur - torch.sin(s) * sigma_data * F_det
                    denom = torch.sin(s)
                else:
                    pred = torch.sin(s) * x_cur + torch.cos(s) * sigma_data * F_det
                    denom = torch.cos(s)

                first_order_step = cos_delta * x_cur - sin_delta * sigma_data * F_det
                if step_idx == 0 or step_idx == num_steps - 1 or pred_prev is None:
                    x_next = first_order_step
                else:
                    logtan = lambda u: torch.log(torch.tan(torch.clamp(u, 1e-4, 1.569)))
                    r_s = (logtan(s) - logtan(t_prev)) / (logtan(s) - logtan(t))
                    correction = (sin_delta / (2 * r_s * torch.clamp(denom, min=1e-3))) * (
                        pred_prev - pred
                    )
                    x_next = first_order_step + (correction if use_pp else -correction)

                # Guided velocity: F -> F - likelihood_score.
                x_next = x_next + sin_delta * sigma_data * likelihood_score.detach()

            t_prev = s
            pred_prev = pred
            x0_tweedie = (torch.cos(s) * x_cur - torch.sin(s) * sigma_data * F_det).detach()

            # SDA-style Langevin corrector at the new level (restricted to a
            # noise band so compute lands where correction actually helps).
            # Graded correction: a second band (corrector_steps2 within
            # sigma <= corrector_max_sigma2) ADDS extra steps at the lowest
            # noise levels, where measurement fidelity is locked in.
            sigma_t_f = float(sigma_data * torch.tan(t))
            n_corr = 0
            if step_idx < num_steps - 1:
                if corrector_steps > 0 and corrector_min_sigma <= sigma_t_f <= corrector_max_sigma:
                    n_corr += corrector_steps
                if corrector_steps2 > 0 and sigma_t_f <= corrector_max_sigma2:
                    n_corr += corrector_steps2
            if n_corr > 0:
                x_next = self._langevin_corrector(
                    x_next,
                    t,
                    sigma_data,
                    measurement,
                    condition,
                    auxiliary,
                    steps=n_corr,
                    snr=corrector_snr,
                    guide_through_net=guide_through_net,
                    noise_scale=noise_scale,
                    randn_like=randn_like,
                )
                pred_prev = None

            if log_every and (step_idx == 0 or (step_idx + 1) % log_every == 0):
                extra = ""
                print(f"step {step_idx:3d}  sigma {sigma_s.item():9.4f}  meas_err {err:.6f}{extra}")

            x_t = x_next

        return x_t

    def _unified_pos_edm_sampler(
        self,
        latents,
        measurement,
        condition=None,
        mask=None,
        randn_like=torch.randn_like,
        num_steps=60,
        sigma_min=0.001,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """Unified EDM posterior sampler that handles both masked and unmasked conditioning."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next.detach().requires_grad_()

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = (x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)).float()

            # Euler step.
            with torch.autocast(latents.device.type, enabled=True, dtype=_amp_dtype()):
                denoised = self.net(x_hat, t_hat, condition)

            d_cur = (x_hat - denoised) / t_hat

            # Compute conditioning score based on whether mask is provided
            if mask is not None:
                likelihood_score, err = self.cm.mask_conditioning(
                    x_prev=x_cur, x_t=x_next, x_0_hat=denoised,
                    measurement=measurement, sigma=t_hat, mask=mask
                )
            else:
                likelihood_score, err = self.cm.conditioning(
                    x_prev=x_cur, x_t=x_next, x_0_hat=denoised,
                    measurement=measurement, sigma=t_hat
                )
            x_next = x_hat + (t_next - t_hat) * (d_cur - likelihood_score)
            if i == 0 or (i + 1) % 2 == 0:
                print(f"{t_hat.item():.4f}, {err:.6f}")

            # Apply 2nd order correction.
            if i < num_steps - 1:
                with torch.no_grad():
                    with torch.autocast(latents.device.type, enabled=True, dtype=_amp_dtype()):
                        denoised = self.net(x_next.float(), t_next.float(), condition)
                    d_prime = (x_next - denoised) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime - likelihood_score)

        return x_next

    def pos_edm_sampler(
        self,
        latents,
        measurement,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=60,
        sigma_min=0.001,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """EDM sampler with standard conditioning."""
        return self._unified_pos_edm_sampler(
            latents=latents, measurement=measurement, condition=condition,
            mask=None, randn_like=randn_like, num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise
        )

    def pos_edm_sampler_with_mask(
        self,
        latents,
        measurement,
        mask,
        condition=None,
        randn_like=torch.randn_like,
        num_steps=60,
        sigma_min=0.001,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """EDM sampler with mask-based conditioning."""
        return self._unified_pos_edm_sampler(
            latents=latents, measurement=measurement, condition=condition,
            mask=mask, randn_like=randn_like, num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise
        )
