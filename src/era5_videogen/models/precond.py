from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def _expand_to_input_dims(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, *([1] * (x.ndim - 1)))


def _2d_resolution(x) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return tuple(int(v) for v in x)


def _model_target(model: DictConfig) -> str:
    return str(model.get("_target_", ""))


def _is_video3d(model: DictConfig) -> bool:
    target = _model_target(model)
    return target.endswith(
        (
            "models.swin3d.Swin3D",
            "models.swin3d.SwinV2",
            "models.dit3d.DiT3D",
            "models.songunet3d.SongUNet3D",
            "models.songunet3d.SongUNet",
        )
    )


def _resolve_condition_mode(model: DictConfig, mode: str) -> str:
    if mode == "auto":
        return "spatial2d" if _is_video3d(model) else "concat"
    if mode not in {"concat", "spatial2d"}:
        raise ValueError("condition_mode must be one of {'auto', 'concat', 'spatial2d'}.")
    return mode


def _prepare_auxilary(
    auxilary: Optional[torch.Tensor],
    auxilary_dim: int,
    x: torch.Tensor,
) -> Optional[torch.Tensor]:
    if auxilary_dim == 0:
        return None
    if auxilary is None:
        return torch.zeros([1, auxilary_dim], device=x.device)
    return auxilary.reshape(-1, auxilary_dim)


class EDMPrecond(torch.nn.Module):
    preconditioning_type = "edm"

    def __init__(
        self,
        model: DictConfig,
        img_resolution: tuple[int, int],
        img_channels: int,
        condition_channels: int = 0,
        auxilary_dim: int = 0,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1,
        num_frames: Optional[int] = None,
        condition_mode: str = "auto",
    ):
        super().__init__()
        self.img_resolution = _2d_resolution(img_resolution)
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.auxilary_dim = auxilary_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.num_frames = num_frames
        self.condition_mode = _resolve_condition_mode(model, condition_mode)

        model_kwargs = dict(
            img_resolution=self.img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            auxilary_dim=auxilary_dim,
            _convert_="object",
        )
        if _is_video3d(model):
            if num_frames is None:
                raise ValueError("num_frames must be provided for 3D video models.")
            model_kwargs["num_frames"] = num_frames
            model_kwargs["condition_channels"] = condition_channels
        elif self.condition_mode == "concat":
            model_kwargs["in_channels"] = img_channels + condition_channels

        self.model: torch.nn.Module = instantiate(model, **model_kwargs)

    def forward(self, x, sigma, condition=None, auxilary=None, **model_kwargs):
        sigma = _expand_to_input_dims(sigma, x)
        auxilary = _prepare_auxilary(auxilary, self.auxilary_dim, x)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x
        if condition is not None and self.condition_channels > 0 and self.condition_mode == "concat":
            arg = torch.cat([arg, condition], dim=1)

        if condition is not None and self.condition_channels > 0 and self.condition_mode == "spatial2d":
            model_kwargs["condition"] = condition

        F_x = self.model(arg, c_noise.flatten(), auxilary=auxilary, **model_kwargs)
        return c_skip * x + c_out * F_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class PassPrecond(torch.nn.Module):
    preconditioning_type = "pass"

    def __init__(
        self,
        model: DictConfig,
        img_resolution: tuple[int, int],
        img_channels: int,
        condition_channels: int = 0,
        auxilary_dim: int = 0,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1,
        num_frames: Optional[int] = None,
        condition_mode: str = "auto",
    ):
        super().__init__()
        self.img_resolution = _2d_resolution(img_resolution)
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.auxilary_dim = auxilary_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.num_frames = num_frames
        self.condition_mode = _resolve_condition_mode(model, condition_mode)

        model_kwargs = dict(
            img_resolution=self.img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            auxilary_dim=auxilary_dim,
            _convert_="object",
        )
        if _is_video3d(model):
            if num_frames is None:
                raise ValueError("num_frames must be provided for 3D video models.")
            model_kwargs["num_frames"] = num_frames
            model_kwargs["condition_channels"] = condition_channels
        elif self.condition_mode == "concat":
            model_kwargs["in_channels"] = img_channels + condition_channels

        self.model: torch.nn.Module = instantiate(model, **model_kwargs)

    def forward(
        self,
        x,
        t,
        condition=None,
        auxilary=None,
        auxiliary=None,
        **model_kwargs,
    ):
        if auxilary is None and auxiliary is not None:
            auxilary = auxiliary
        auxilary = _prepare_auxilary(auxilary, self.auxilary_dim, x)

        arg = x
        if condition is not None and self.condition_channels > 0 and self.condition_mode == "concat":
            arg = torch.cat([arg, condition], dim=1)

        if condition is not None and self.condition_channels > 0 and self.condition_mode == "spatial2d":
            model_kwargs["condition"] = condition

        return self.model(arg, t.flatten(), auxilary=auxilary, **model_kwargs)
