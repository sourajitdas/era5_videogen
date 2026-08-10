# Multimodal spatiotemporal atmospheric data assimilation with latent Video flow matching

Code for [Chakraborty & Maulik (2026), arXiv:2608.05103](https://arxiv.org/abs/2608.05103).

A latent generative **video** prior of ERA5 used as a general-purpose data
assimilation engine: one frozen prior, never conditioned on observations,
performs super-resolution, station data fusion, and observation-to-forecast.
Only the measurement operator supplied at sampling time changes between tasks.

The prior models 8-day windows (32 six-hourly frames, 69 ERA5 variables,
128x256 grid). A 3D autoencoder compresses each window to a (128, 8, 32, 64)
latent; a 525M-parameter DiT3D flow-matching model (TrigFlow) is trained on the
latents. Assimilation is posterior sampling with guidance differentiated
through the frozen decoder. A single 40 GB GPU runs every sampling task.

## Quick start

Download the trained checkpoints and demo data (one test window,
2020-01-24 to 2020-02-01, with matching station observations) from
[Box](https://app.box.com/folder/407292865654) and place the `data/` and
`checkpoints/` folders in the repo root. Then these run out of the box:

```bash
scripts/run_sr.sh          # super-resolution from 8x-coarsened observations
scripts/run_igra.sh        # full-state reconstruction from radiosonde soundings
scripts/run_forecast.sh    # 6-day forecast driven only by station observations
```

Each writes `samples.npy`, `ground_truth.npy`, per-variable RMSE, and figures
under `outputs/`. All scripts take window indices (`scripts/run_sr.sh 93 244`),
and forecasts can be chained autoregressively:

```bash
scripts/run_forecast.sh 93 --chain 3
```

## Repo structure

```
src/era5_videogen/     the prior: training, sampling, measurement operators
  train.py             trains the TrigFlow latent video prior
  generate_posterior.py        super-resolution and gridded observations
  generate_posterior_igra.py   sparse station observations (IGRA / ISD / ICOADS)
  generate_forecast_chain.py   autoregressive multi-window forecasts
  generating/          the guided sampler and the measurement operators
src/ae/                the autoencoder: training and latent encoding
checkpoints/           trained autoencoder + prior
data/                  bundled demo window and observations
scripts/               one script per task, plus the training pipeline
```

## The three tasks

| Task | Observations | Sampler | NFE |
|---|---|---|---|
| Super-resolution | every 4th frame, 8x coarsened | 30 steps + low-sigma corrector, lambda=0 | 47 |
| Station assimilation | IGRA soundings, 8 of 32 frames | 50 steps + corrector, DSG guidance, lambda=0 | 76 |
| Observations to forecast | joint network, first 8 frames only | as above, guidance scale 0.4 | 76 |

The sampler is a guided TrigFlow ODE (EDM schedule, DPM-Solver++(2M), DPS
guidance with SDA annealing, optional low-sigma Langevin corrector). Key flags:
`--num_steps`, `--corrector_steps`, `--corrector_max_sigma`, `--guide_norm dsg`,
`--noise_scale`, `--guide_momentum`.

## Training from scratch

```bash
export ERA5_DATA_ROOT=/path/to/era5              # see Data layout
scripts/train_ae.sh                              # 1. autoencoder
scripts/encode_latents.sh <ae_checkpoint.pt> /path/to/latents   # 2. encode + stats
scripts/train_prior.sh data.dataset.root=/path/to/latents       # 3. prior
```

Released models: autoencoder trained on 16 A100s for ~8-9 days, flow matching on 8 A100s for ~5 days (EMA at 19,901 kimg).

## Data 

`ERA5_DATA_ROOT` holds one HDF5 file per 6-hourly timestep (`input/<variable>`
2D datasets on the 128x256 grid) in `train/`, `val/`, `test/` subdirectories,
plus `normalize_mean.npz`, `normalize_std.npz`, `lat.npy`, `lon.npy`. The
bundled `data/era5/` is the reference. Station observations are frame-indexed
pickles of `[queries, values]` per channel.

## Cite

```bibtex
@article{chakraborty2026multimodal,
  title={Multimodal Spatiotemporal Atmospheric Data Assimilation with Latent Flow-matching},
  author={Chakraborty, Dibyajyoti and Maulik, Romit},
  journal={arXiv preprint arXiv:2608.05103},
  year={2026}
}
```
