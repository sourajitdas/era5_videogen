from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Optional

import torch


# ----------------------------------------------------------------------------


def _noise_shape(x: torch.Tensor) -> tuple[int, ...]:
    return (x.shape[0],) + (1,) * (x.ndim - 1)


class DiffusionLoss(torch.nn.Module, ABC):
    """All diffusion loss implementations should inherit this."""

    @abstractmethod
    def forward(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        auxilary: Optional[Any] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this.")


# ----------------------------------------------------------------------------


class TrigFlowLoss(DiffusionLoss):
    """TrigFlow velocity-prediction loss (sCM/TrigFlow, arXiv:2410.11081, Eq. 4).

    x_t = cos(t) x + sin(t) z, with z ~ N(0, sigma_data^2 I), and the model
    predicts the velocity target v_t = cos(t) z - sin(t) x. The network is
    conditioned on the raw angle t and sees the unit-variance input x_t/sigma_data.

    Noise levels are emphasised either by a FIXED weighting or by a learned
    ADAPTIVE (logvar) weighting:

      * ``"edm"`` (default): w(t) = 1. This is uniform in velocity space and is
        *exactly* the EDM fixed weighting. The EDM denoiser weight
        (sigma^2 + sigma_d^2)/(sigma*sigma_d)^2 maps to a constant here because
        the x0-error and velocity-error differ by exactly sin^2(t):
        ||dx0||^2 = sin^2(t) * ||sigma_d F - v_t||^2, and w_EDM * sin^2(t) = 1/sigma_d^2.
      * ``"min_snr"`` / ``"soft_min_snr"`` (experimental): emphasise mid noise via
        the TrigFlow SNR(t) = cot^2(t) = sigma_d^2/sigma^2, converted to velocity
        space by the same sin^2(t) factor.
      * ``"adaptive"`` (EDM2 / sCM uncertainty weighting): the network additionally
        predicts a per-noise-level log-variance u(t) via its ``logvar`` head and the
        loss becomes ``||sigma_d F - v_t||^2 / exp(u(t)) + u(t)`` with a uniform prior
        weight. The model learns to down-weight the irreducible low-noise band, which
        rebalances gradients across noise levels and makes the reported number O(1).
        **Requires a model built with ``logvar=True``.** See EDM2 (arXiv:2312.02696)
        and sCM/TrigFlow (arXiv:2410.11081).

    Where mass lands across noise levels is the primary lever and is controlled by
    the sigma proposal (``noise_dist`` / ``P_mean`` / ``P_std``).

    TrigFlow uses the raw model or a pass-through wrapper (``PassPrecond``).
    Do not use EDM preconditioning here.
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        noise_dist: str = "lognormal",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        P_mean: float = -0.4,
        P_std: float = 1.0,
        weighting: str = "edm",
        snr_gamma: float = 5.0,
    ):
        super().__init__()
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive")
        if noise_dist not in {"loguniform", "lognormal"}:
            raise ValueError("noise_dist must be one of {'loguniform', 'lognormal'}")
        if sigma_min <= 0 or sigma_max <= 0 or sigma_max <= sigma_min:
            raise ValueError("Expected 0 < sigma_min < sigma_max")
        if weighting not in {"edm", "min_snr", "soft_min_snr", "adaptive"}:
            raise ValueError(
                "weighting must be one of "
                "{'edm', 'min_snr', 'soft_min_snr', 'adaptive'}"
            )

        self.sigma_data = float(sigma_data)
        self.noise_dist = noise_dist
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.weighting = weighting
        self.snr_gamma = float(snr_gamma)

    def _sample_tau(self, x: torch.Tensor) -> torch.Tensor:
        shape = _noise_shape(x)
        if self.noise_dist == "lognormal":
            rnd_normal = torch.randn(shape, device=x.device)
            return (rnd_normal * self.P_std + self.P_mean).exp()

        u = torch.rand(shape, device=x.device)
        sigma_min = torch.as_tensor(self.sigma_min, device=x.device)
        sigma_max = torch.as_tensor(self.sigma_max, device=x.device)
        log_tau = sigma_min.log() + u * (sigma_max.log() - sigma_min.log())
        return log_tau.exp()

    def _weight(
        self, cos_t: torch.Tensor, sin_t: torch.Tensor
    ) -> torch.Tensor | float:
        if self.weighting == "edm":
            return 1.0
        # TrigFlow SNR and the velocity-space conversion factor sin^2(t).
        sin2 = sin_t**2
        snr = (cos_t / sin_t.clamp_min(1e-8)) ** 2
        if self.weighting == "min_snr":
            return torch.minimum(snr, torch.full_like(snr, self.snr_gamma)) * sin2
        # soft_min_snr
        return (snr * self.snr_gamma / (snr + self.snr_gamma)) * sin2

    def _unwrap(self, net: torch.nn.Module) -> torch.nn.Module:
        net = getattr(net, "module", net)
        return getattr(net, "_orig_mod", net)

    def _has_edm_preconditioning(self, net: torch.nn.Module) -> bool:
        net = self._unwrap(net)
        preconditioning_type = getattr(net, "preconditioning_type", None)
        if preconditioning_type is not None:
            return preconditioning_type == "edm"
        return hasattr(net, "round_sigma") and hasattr(net, "model")

    def forward(self, net, x, condition=None, auxilary=None):
        if self._has_edm_preconditioning(net):
            raise ValueError(
                "TrigFlowLoss does not support EDMPrecond. Use the raw model or PassPrecond."
            )

        tau = self._sample_tau(x)
        t = torch.atan(tau / self.sigma_data)
        z = torch.randn_like(x) * self.sigma_data

        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        x_t = cos_t * x + sin_t * z
        v_t = cos_t * z - sin_t * x

        if self.weighting == "adaptive":
            # EDM2 / sCM uncertainty (logvar) weighting. The network predicts a
            # per-noise-level log-variance u(t) (its logvar head); the velocity MSE
            # is normalised by exp(u(t)) with a +u(t) penalty (uniform prior weight).
            # This self-balances the gradient across noise levels so the irreducible
            # low-noise band stops dominating. Requires a model with logvar=True.
            out = net(
                x_t / self.sigma_data,
                t.flatten(),
                condition,
                auxilary,
                return_logvar=True,
            )
            if not isinstance(out, tuple):
                raise ValueError(
                    "TrigFlowLoss(weighting='adaptive') requires a model with a "
                    "logvar head; set model.logvar: true."
                )
            F_x, logvar = out
            # fp32 for the exp/penalty under bf16 autocast (numerical safety).
            logvar = logvar.reshape(_noise_shape(x)).float()
            sq_err = (self.sigma_data * F_x - v_t) ** 2
            loss = sq_err / logvar.exp() + logvar
            return loss.mean()

        F_x = net(x_t / self.sigma_data, t.flatten(), condition, auxilary)

        weight = self._weight(cos_t, sin_t)
        loss = weight * (self.sigma_data * F_x - v_t) ** 2
        return loss.mean()


# Flow Matching Loss
