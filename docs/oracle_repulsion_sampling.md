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

## Portable Python workflow (no Slurm required)

The workflow can be run directly from a Bash shell on any Linux workstation or
cluster node with the Python environment, model files, Laura bundle, and a
CUDA GPU with sufficient memory. The sampling configuration below uses eight
joint particles and was designed for an 80 GB GPU. The selection, plotting,
and aggregation commands do not require Slurm.

Run all commands from the repository root. First activate the project
environment and define machine-specific paths:

```bash
conda activate era5_env

export REPO_DIR="$PWD"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export BUNDLE="/path/to/laura_w941_forecast_bundle_v1"
export AE="/path/to/autoencoder/ae_model.pt"
export CKPT="${BUNDLE}/models/dj_checkpoint-019901.pt"
export OBS="${BUNDLE}/data/observations/global_joint_2020_year_superob.pkl"
export OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
export RUNTIME_CONFIG="${OUTPUT_ROOT}/laura_runtime_config.yaml"

export BASELINE_NAME="laura_w941_forecast_obs2d_repulsion_k8_alpha0p005_hurricane_h9"
export BASELINE_RUN="${OUTPUT_ROOT}/${BASELINE_NAME}"
export ORACLE_TAG="oracle_sites192_alpha0p005_hurricane_h9"
export ORACLE_DIR="${OUTPUT_ROOT}/${ORACLE_TAG}"
export ORACLE_NAME="laura_w941_forecast_obs2d_${ORACLE_TAG}"
export ORACLE_RUN="${OUTPUT_ROOT}/${ORACLE_NAME}"
export MANIFEST="${ORACLE_DIR}/oracle_observation_selection.json"

mkdir -p "${OUTPUT_ROOT}" "${ORACLE_DIR}"
```

### 1. Prepare and validate the Laura runtime configuration

```bash
python scripts/prepare_laura_forecast_repulsion.py \
  --bundle "${BUNDLE}" \
  --autoencoder "${AE}" \
  --output-config "${RUNTIME_CONFIG}" \
  --source-index 941
```

Define the sampler arguments shared by the pilot, oracle-mask, and additional
ensemble runs:

```bash
export OBS_VARS="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,geopotential_500,geopotential_850,u_component_of_wind_500,u_component_of_wind_850,v_component_of_wind_500,v_component_of_wind_850,temperature_500,temperature_850,specific_humidity_500,specific_humidity_850"

COMMON_SAMPLER_ARGS=(
  --diffusion_config "${RUNTIME_CONFIG}"
  --diffusion_checkpoint "${CKPT}"
  --ae_model_path "${AE}"
  --igra_pkl "${OBS}"
  --obs_variables "${OBS_VARS}"
  --output_dir "${OUTPUT_ROOT}"
  --sample_index 0
  --num_posterior_samples 8
  --obs_frames 0-7
  --num_steps 50
  --guidance_scale 0.4
  --guide_norm dsg
  --gamma 2e-6
  --meas_std 5e-4
  --corrector_steps 1
  --corrector_snr 0.3
  --corrector_max_sigma 3
  --noise_scale 0.0
  --repulsion_strength 0.005
  --repulsion_bandwidth 0
  --repulsion_min_sigma 0.05
  --repulsion_max_sigma 5.0
  --repulsion_schedule sine
  --repulsion_spatial_pool 8
  --repulsion_mask_mode hurricane
  --hurricane_descriptor_horizon 9
  --hurricane_region_lat 15 35
  --hurricane_region_lon -100 -65
  --no_net_grad
  --log_every 2
)
```

### 2. Generate the K=8 pilot ensemble

The pilot ensemble is used by the retrospective site-selection objective:

```bash
python -m era5_videogen.generate_posterior_igra_repulsion \
  "${COMMON_SAMPLER_ARGS[@]}" \
  --name "${BASELINE_NAME}" \
  --seed 17
```

### 3. Select the oracle-informed observation mask

This reproduces the 192-site experiment, with a 90% target spread ratio and a
spread-penalty weight of 10:

```bash
python scripts/select_laura_oracle_observations.py \
  --bundle "${BUNDLE}" \
  --baseline-run "${BASELINE_RUN}" \
  --observations-pkl "${OBS}" \
  --output-manifest "${MANIFEST}" \
  --source-index 941 \
  --obs-days 2 \
  --selection-mode oracle \
  --budget 192 \
  --selection-region 15 35 -100 -65 \
  --site-round-decimals 3 \
  --min-site-records 4 \
  --min-site-separation-deg 0.5 \
  --target-spread-ratio 0.9 \
  --spread-penalty 10 \
  --weight-temperature 1.0 \
  --landfall-weight 0.5 \
  --keep-outside-region
```

### 4. Generate a posterior ensemble using the selected observations

Future ERA5 is not provided to this command; the sampler receives only the
fixed observation-selection manifest:

```bash
python -m era5_videogen.generate_posterior_igra_repulsion \
  "${COMMON_SAMPLER_ARGS[@]}" \
  --observation_selection_manifest "${MANIFEST}" \
  --name "${ORACLE_NAME}" \
  --seed 17
```

### 5. Plot and score the K=8 comparison

```bash
python scripts/plot_laura_oracle_selection.py \
  --bundle "${BUNDLE}" \
  --baseline-run "${BASELINE_RUN}" \
  --selected-run "${ORACLE_RUN}" \
  --selection-manifest "${MANIFEST}" \
  --output-dir "${ORACLE_DIR}" \
  --source-index 941 \
  --obs-days 2
```

### 6. Optional: generate and aggregate 100 posterior members

The following loop runs twelve additional K=8 batches sequentially, using
seeds 18--29. A non-Slurm environment may instead run these commands in its own
job scheduler, provided that each GPU has sufficient memory.

```bash
for SEED in $(seq 18 29); do
  SEED_TAG=$(printf '%04d' "${SEED}")
  RUN_NAME="laura_w941_forecast_obs2d_oracle_sites192_k8_seed${SEED_TAG}_alpha0p005_hurricane_h9"
  python -m era5_videogen.generate_posterior_igra_repulsion \
    "${COMMON_SAMPLER_ARGS[@]}" \
    --observation_selection_manifest "${MANIFEST}" \
    --name "${RUN_NAME}" \
    --seed "${SEED}"
done
```

Build the repeated `--batch-run` arguments and aggregate the first 100 members:

```bash
BATCH_ARGS=()
for SEED in $(seq 18 29); do
  SEED_TAG=$(printf '%04d' "${SEED}")
  BATCH_ARGS+=(
    --batch-run
    "${OUTPUT_ROOT}/laura_w941_forecast_obs2d_oracle_sites192_k8_seed${SEED_TAG}_alpha0p005_hurricane_h9"
  )
done

ANALYSIS_DIR="${OUTPUT_ROOT}/oracle_sites192_alpha0p005_hurricane_h9_ensemble100"

python scripts/aggregate_laura_oracle_ensemble.py \
  --bundle "${BUNDLE}" \
  --reference-run "${ORACLE_RUN}" \
  "${BATCH_ARGS[@]}" \
  --selection-manifest "${MANIFEST}" \
  --source-index 941 \
  --obs-days 2 \
  --target-members 100 \
  --track-density-bandwidth-km 0 \
  --output-dir "${ANALYSIS_DIR}"
```

`--track-density-bandwidth-km 0` selects the median pairwise trajectory
distance as the automatic Gaussian-kernel bandwidth.

## NERSC/Slurm wrappers

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
