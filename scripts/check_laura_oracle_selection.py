#!/usr/bin/env python3
"""Small dependency-free check for Laura observation-manifest filtering."""

from __future__ import annotations

import numpy as np

from era5_videogen.observation_selection import filter_igra_measurement, site_key


def main() -> None:
    queries = np.asarray(
        [[20.0, -80.0], [21.0, -81.0], [45.0, -110.0]], dtype=np.float32
    )
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    measurement = [[[queries]], [[values]]]
    manifest = {
        "schema_version": 1,
        "source_index": 941,
        "observed_frames": [0],
        "site_round_decimals": 3,
        "selection_region": {
            "latitude": [15.0, 35.0],
            "longitude": [-100.0, -65.0],
        },
        "keep_outside_region": True,
        "selected_site_keys": [site_key(20.0, -80.0, 3)],
    }
    filtered, stats = filter_igra_measurement(
        measurement, manifest, source_index=941, obs_frames=[0]
    )
    kept_queries = np.asarray(filtered[0][0][0])
    kept_values = np.asarray(filtered[1][0][0])
    assert kept_queries.shape == (2, 2)
    assert np.allclose(kept_values, [1.0, 3.0])
    assert stats["selected_records_in_region"] == 1
    assert stats["dropped_records_in_region"] == 1
    assert stats["retained_records_outside_region"] == 1
    assert stats["retained_records_per_frame"] == [2]
    assert site_key(20.0, 280.0, 3) == site_key(20.0, -80.0, 3)
    print("PASS: oracle observation-selection manifest filtering")


if __name__ == "__main__":
    main()
