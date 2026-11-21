# Oracle-informed observation selection with repulsion-guided sampling

This feature branch extends the repulsion-guided posterior sampler with a
retrospective observation-selection experiment for Hurricane Laura. It is
intended as a research diagnostic for extreme-event sampling and observing
system design.

## Method at a glance

1. Generate a small pilot posterior ensemble using the first two observed
   days.
2. At every candidate radiosonde site, calculate how well each pilot member
   matches the observed variables and times.
3. Use the known future ERA5 trajectory as an **oracle target** to score the
   pilot members by forecast-track and landfall error.
4. Greedily retain sites that favor accurate pilot members while penalizing
   excessive loss of trajectory spread.
5. Freeze the selected-site mask and generate a new repulsion-guided posterior
   ensemble using only the selected observations.

Future ERA5 is used only in the retrospective mask-design step. It is not
given to the final posterior sampler. Consequently, this is not an operational
forecasting method; it identifies observation patterns that a future
non-oracle selection method could attempt to approximate.

For a selected site set \(S\), the pilot members receive soft weights from
their observation misfits. The selector minimizes a track-error term plus a
penalty when weighted pairwise trajectory spread falls below the requested
fraction of the original spread.

## Main implementation files

- `src/era5_videogen/observation_selection.py`: loads a selection manifest and
  filters the real in-situ measurement records.
- `src/era5_videogen/generate_posterior_igra_repulsion.py`: accepts
  `--observation_selection_manifest` and records mask provenance.
- `scripts/select_laura_oracle_observations.py`: constructs the oracle-selected
  site mask.
- `scripts/check_laura_oracle_selection.py`: lightweight filtering checks.
- `scripts/plot_laura_oracle_selection.py`: selected/dropped-site and track
  comparison plots.
- `scripts/run_laura_oracle_site_selection_nersc.sh`: pilot selection and K=8
  oracle-mask demonstration.
- `scripts/run_laura_oracle_ensemble_batch_nersc.sh`: additional independent
  K=8 posterior batches.
- `scripts/aggregate_laura_oracle_ensemble.py`: 100-member diagnostics,
  convergence, and empirical trajectory-density coloring.
- `scripts/run_laura_oracle_ensemble_aggregate_nersc.sh`: NERSC aggregation
  wrapper.

The aggregate plot colors tracks using a Gaussian-kernel density estimate in
trajectory space. Red tracks are isolated/low-support (extreme), while blue
tracks lie in densely sampled/high-support regions. This is a relative
empirical support diagnostic, not a normalized per-member probability from the
flow sampler.

## NERSC example

The scripts assume that the Laura source-index-941 bundle and the base
repulsion sampler are already available from the parent `super-resolution`
branch.

Create the oracle mask and the first K=8 run:

```bash
sbatch scripts/run_laura_oracle_site_selection_nersc.sh
```

Generate twelve additional K=8 batches (seeds 18--29):

```bash
sbatch --array=1-12%4 scripts/run_laura_oracle_ensemble_batch_nersc.sh
```

After the array completes, aggregate the first 100 members:

```bash
sbatch \
  --export=ALL,TARGET_MEMBERS=100,TRACK_DENSITY_BANDWIDTH_KM=0 \
  scripts/run_laura_oracle_ensemble_aggregate_nersc.sh
```

`TRACK_DENSITY_BANDWIDTH_KM=0` selects the median pairwise track distance as
the automatic Gaussian-kernel bandwidth.

## Interpretation cautions

- The oracle selector uses the realized future track and therefore measures
  retrospective information value.
- The generated posterior members remain equally weighted Monte Carlo draws.
- Repulsion is coupled within each K=8 batch; there is no repulsion interaction
  between members from different batches in the 100-member aggregate.
- A matched random-drop mask and a full-observation ensemble are appropriate
  controls when assessing whether a selected mask improves forecast skill.
