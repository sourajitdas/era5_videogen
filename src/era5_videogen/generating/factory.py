from typing import Callable

import torch

from era5_videogen.generating.diffusion import (
    DiffusionSampler,
    PosteriorDiffusionSampler,
)
from era5_videogen.generating.solver import ODESolver


def sampler_factory(mode: str, net: torch.nn.Module, net_cfg=None) -> Callable[..., torch.Tensor]:
    """Factory to return a sampler function based on the mode.

    Args:
        mode: Sampling mode ("edm", "cfm", "edm_pos_sample", "edm_pos_sample_mask")
        net: Neural network model
        net_cfg: Optional network configuration

    Returns:
        Sampler function
    """
    if mode == "edm":
        sampler_obj = DiffusionSampler(net)

        def sampler(X, generator: torch.Generator, condition=None, device=None, in_shape=None, *args, **kwargs) -> torch.Tensor:
            if in_shape is None:
                in_shape = X.shape
            if device is None:
                device = X.device
            return sampler_obj.edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                *args, **kwargs
            )
        return sampler

    elif mode == "cfm":
        sampler_obj = ODESolver(net)

        def cfm_sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return sampler_obj.sample(x_init=X, step_size=0.1, *args, **kwargs)
        return cfm_sampler

    elif mode == "edm_pos_sample":
        sampler_obj = PosteriorDiffusionSampler(net)

        def pos_sampler(X, generator, condition=None, device=None, in_shape=None, *args, **kwargs) -> torch.Tensor:
            if in_shape is None:
                in_shape = X.shape
            if device is None:
                device = X.device
            return sampler_obj.pos_edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                measurement=X,
                *args, **kwargs
            )
        return pos_sampler

    elif mode == "edm_pos_sample_mask":
        sampler_obj = PosteriorDiffusionSampler(net)

        def pos_mask_sampler(X, generator, mask, condition=None, device=None, in_shape=None, *args, **kwargs) -> torch.Tensor:
            if in_shape is None:
                in_shape = X.shape
            if device is None:
                device = X.device
            return sampler_obj.pos_edm_sampler_with_mask(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                measurement=X,
                mask=mask,
                *args, **kwargs
            )
        return pos_mask_sampler

    else:
        # Default: just return network output
        def sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return net(X)
        return sampler
