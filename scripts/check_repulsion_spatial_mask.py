#!/usr/bin/env python3
"""Small CPU checks for spatial, split, and hurricane repulsion objectives."""

from __future__ import annotations

import numpy as np
import torch

from era5_videogen.generating.conditioning_methods_repulsion import (
    IGRAVideoConditioning,
    IGRAVideoOperator,
)


def main() -> None:
    operator = IGRAVideoOperator(obs_frames=[0], channel_map=[1])
    conditioning = IGRAVideoConditioning(operator, decode_fn=lambda value: value)
    decoded = torch.zeros((2, 3, 2, 8, 16), dtype=torch.float32)
    query_locations = [[np.asarray([[0.0, 0.0]], dtype=np.float32)]]
    measurement = [query_locations, [[np.asarray([0.0], dtype=np.float32)]]]

    mask = conditioning._observation_complement_mask(
        decoded, query_locations, observation_buffer=0
    )
    excluded = torch.nonzero(mask[0, 1, 0] == 0, as_tuple=False)
    assert excluded.shape[0] == 4, excluded
    assert torch.all(mask[0, 0, 0] == 1)
    assert torch.all(mask[0, 1, 1] == 1)

    observed_only_difference = decoded.clone()
    for row, column in excluded.tolist():
        observed_only_difference[1, 1, 0, row, column] = 1.0
    _, all_stats = conditioning._repulsion_objective(
        observed_only_difference,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="all",
    )
    _, spatial_stats = conditioning._repulsion_objective(
        observed_only_difference,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="spatial",
        measurement=measurement,
    )
    assert all_stats["pairwise_mse"] > 0
    assert spatial_stats["pairwise_mse"] == 0

    unobserved_difference = observed_only_difference.clone()
    unobserved_difference[1, 1, 0, 0, 5] = 1.0
    _, spatial_stats = conditioning._repulsion_objective(
        unobserved_difference,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="spatial",
        measurement=measurement,
    )
    assert spatial_stats["pairwise_mse"] > 0

    # The split objective must preserve the forecast-frame term and add a
    # separately normalized observed-frame-complement term with weight beta.
    split_input = decoded.clone()
    split_input[1, 1, 1, 0, 5] = 1.0  # forecast-frame difference
    split_input[1, 1, 0, 0, 5] = 1.0  # unobserved pixel in observed frame
    frame_objective, _ = conditioning._repulsion_objective(
        split_input,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="frames",
        measurement=measurement,
    )
    split_zero_objective, _ = conditioning._repulsion_objective(
        split_input,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="split",
        measurement=measurement,
        split_beta=0.0,
    )
    split_objective, split_stats = conditioning._repulsion_objective(
        split_input,
        bandwidth=1.0,
        spatial_pool=1,
        mask_mode="split",
        measurement=measurement,
        split_beta=1.0,
    )
    assert torch.allclose(frame_objective, split_zero_objective)
    assert split_objective < split_zero_objective
    assert split_stats["forecast_pairwise_mse"] > 0
    assert split_stats["observed_pairwise_mse"] > 0

    hurricane_config = {
        "latitude": np.linspace(-90.0, 90.0, 8, dtype=np.float32),
        "longitude": np.arange(16, dtype=np.float32) * 22.5,
        "region_lat": [-20.0, 50.0],
        "region_lon": [-110.0, -20.0],
        "horizon": 1,
        "mslp_channel": 0,
        "u10_channel": 1,
        "v10_channel": 2,
        "weather_means": [100000.0, 0.0, 0.0],
        "weather_stds": [1000.0, 10.0, 10.0],
        "pressure_temperature_hpa": 2.0,
        "wind_temperature_ms": 2.0,
        "track_radius_deg": 30.0,
        "position_scale_deg": 2.0,
        "pressure_scale_hpa": 10.0,
        "wind_scale_ms": 5.0,
    }
    hurricane_conditioning = IGRAVideoConditioning(
        operator,
        decode_fn=lambda value: value,
        hurricane_descriptor_config=hurricane_config,
    )
    hurricane_input = torch.zeros(
        (2, 3, 2, 8, 16), dtype=torch.float32, requires_grad=True
    )
    with torch.no_grad():
        hurricane_input[0, 0, 1, 4, 12] = -3.0
        hurricane_input[1, 0, 1, 5, 14] = -3.0
        hurricane_input[0, 1, 1, 4, 12] = 2.0
        hurricane_input[1, 2, 1, 5, 14] = 2.0
    hurricane_objective, hurricane_stats = (
        hurricane_conditioning._repulsion_objective(
            hurricane_input,
            bandwidth=1.0,
            spatial_pool=1,
            mask_mode="hurricane",
            measurement=measurement,
        )
    )
    hurricane_gradient = torch.autograd.grad(
        hurricane_objective, hurricane_input
    )[0]
    assert hurricane_stats["descriptor_track_distance_km"] > 0
    assert torch.isfinite(hurricane_gradient).all()
    assert hurricane_gradient.abs().sum() > 0
    print(
        "PASS: spatial, split, and hurricane-descriptor repulsion checks passed."
    )


if __name__ == "__main__":
    main()
