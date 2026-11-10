#!/usr/bin/env python3
"""Small CPU check for the sparse-observation repulsion mask."""

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
    print(
        "PASS: observed bilinear footprints are excluded and split repulsion "
        "keeps separately normalized forecast/complement objectives."
    )


if __name__ == "__main__":
    main()
