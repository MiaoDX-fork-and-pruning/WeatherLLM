# AI Weather / High-Resolution Precipitation Research Landscape

> Proposed research note for WeatherLLM / PhyDiff-Net  
> Status: discussion draft  
> Scope: high-resolution precipitation forecasting, NWP post-processing/downscaling, probabilistic forecasting, extreme precipitation, and AI weather foundation models.

## Executive summary

The current reference set — Pangu-Weather, GraphCast, GenCast, and Aurora — is important but not sufficient for positioning this project.

Those four papers mostly represent **global medium-range learned weather forecasting**. They establish that neural weather models can replace or complement parts of numerical weather prediction (NWP), but they do not directly answer the problem that WeatherLLM is currently best equipped to study:

> **Given a future atmospheric forecast from an operational/global model and recent high-resolution precipitation observations, can a learned probabilistic model produce materially better high-resolution precipitation forecasts — especially for heavy and extreme rainfall — than raw NWP, statistical post-processing, and deterministic neural downscaling?**

This is a different scientific setting from GraphCast/Pangu-style independent forecasting and from radar-only precipitation nowcasting.

The most promising project positioning is therefore **forecast-conditioned probabilistic high-resolution precipitation post-processing/downscaling**, with GMCP as high-resolution target supervision and ECMWF/ERA5 atmospheric fields as dynamical context.

A second, useful setting should be retained as an **analysis-conditioned upper bound**. It should not be presented as an operational forecast result.

## 1. Taxonomy: do not mix these problem settings

| Setting | Typical input | Output | Horizon | Representative work | Relevance to WeatherLLM |
|---|---|---|---|---|---|
| Independent global weather forecasting | analysis/reanalysis atmospheric state | future global atmosphere | days to 10–15 d | GraphCast, Pangu, FourCastNet, FuXi, AIFS, GenCast, Aurora | Important context, but not the closest baseline |
| Precipitation nowcasting | recent radar / rain fields | local high-res rain | minutes to ~3–6 h | DGMR, NowcastNet, MetNet family | Useful architectures/metrics; different conditioning regime |
| Forecast-conditioned precipitation post-processing | NWP forecast + static/local predictors (+ observations) | corrected/local precipitation | hours to days | statistical MOS/EMOS/BMA; ML downscaling | **Closest scientific family** |
| Statistical / ML downscaling | coarse predictor fields | high-res field | diagnostic or forecast | CNN/U-Net/SR/diffusion downscaling literature | **Directly relevant** |
| Regional learned forecasting | regional initial state + boundary/global context | regional future state | hours to days | regional AI models / learned nesting | Potential extension |
| Hybrid physics–ML | governing dynamics + learned component | future state | varies | NeuralGCM; physically constrained nowcasting | Relevant only if physics is actually enforced |

### Critical distinction

Using meteorological variables as model inputs is **not** by itself “physics-informed”. A strong physics claim requires at least one of: an explicit conservation/evolution equation in the model; a differentiable physical solver/operator; a measurable physically meaningful constraint; or an ablation demonstrating that the physical constraint itself is responsible for improvement.

## 2. Global AI weather forecasting: current coverage and blind spots

### Tier 0 global references

**GraphCast** — learned global weather simulator, 0.25° grid, 6-hour steps, up to 10 days. Core value for this project: learned atmospheric dynamics and evaluation methodology. It is not a direct precipitation-specialized downscaling baseline.

- https://deepmind.google/research/publications/22598/
- https://github.com/google-deepmind/graphcast

**Pangu-Weather** — global 3D neural weather model at 0.25° with hierarchical temporal aggregation. Important for architecture and learned global dynamics, but primarily deterministic and not focused on calibrated high-resolution precipitation post-processing.

- https://www.nature.com/articles/s41586-023-06185-3
- https://github.com/198808xc/Pangu-Weather

**GenCast** — probabilistic global weather model, 0.25°, 12-hour steps, ensembles to 15 days. Its most relevant lesson here is probabilistic verification: CRPS, risk, ensembles, and extreme-event probabilities.

- https://www.nature.com/articles/s41586-024-08252-9
- https://github.com/google-deepmind/graphcast/tree/main/gencast

**Aurora** — Earth-system foundation model pretrained on heterogeneous geophysical data and adapted to multiple tasks. Relevant for transfer/foundation-model ideas, but foundation-model breadth does not establish the specific value of GMCP-conditioned precipitation post-processing.

- https://www.nature.com/articles/s41586-025-09005-y
- https://www.microsoft.com/en-us/research/publication/aurora-a-foundation-model-for-the-earth-system/

### Tier 1 global references missing from the current core set

**FourCastNet / AFNO** — influential global learned weather forecasting with Fourier/AFNO modeling.
- https://arxiv.org/abs/2202.11214
- https://github.com/NVlabs/FourCastNet

**FuXi** — 15-day cascaded global ML forecasting at 0.25°/6 h, with ensemble experiments and CRPS.
- https://www.nature.com/articles/s41612-023-00512-1
- https://github.com/tpys/FuXi

**ECMWF AIFS / AIFS ENS** — operationally critical because this project uses ECMWF products. AIFS became operational in February 2025 and AIFS ENS in July 2025. Future experiments should specify whether conditioning comes from IFS, AIFS, or both.
- https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ai-forecasts-become-operational
- https://www.ecmwf.int/en/forecasts/datasets/aifs-machine-learning-data

**NeuralGCM** — a genuine hybrid physics–ML system: differentiable large-scale dynamics plus learned components trained end-to-end.
- https://www.nature.com/articles/s41586-024-07744-y
- https://github.com/google-research/neuralgcm

**Stormer** — transformer-based global weather model with randomized dynamics forecasting, useful for scaling and ablation methodology.
- https://arxiv.org/abs/2312.03876
- https://github.com/tung-nd/stormer

Tier-2 context includes ClimaX, FengWu, Prithvi Weather/Climate, learned data assimilation, regional foundation models, and other recent Earth-system foundation models. These are useful context but should not dominate baseline compute.

## 3. Precipitation-specific work: nowcasting is relevant but not equivalent

**DGMR** uses recent radar precipitation to generate stochastic future radar fields. Its key lesson is that deterministic pixel losses often produce blurry averages, while generative models can preserve spatial structure and represent uncertainty.
- https://www.nature.com/articles/s41586-021-03854-z

**NowcastNet** predicts high-resolution precipitation from recent radar and combines physical evolution with conditional generation, with lead times up to about 3 h. It is directly relevant to claims about “physics + generative precipitation”, but it is a nowcasting setting rather than multi-day NWP-conditioned forecasting.
- https://www.nature.com/articles/s41586-023-06184-4

**MetNet / MetNet-2 / MetNet-3** are important high-resolution regional weather/precipitation systems demonstrating the value of large receptive fields and multi-source conditioning.
- https://arxiv.org/abs/2003.12140
- https://arxiv.org/abs/2111.07470
- https://arxiv.org/abs/2306.06079

Other useful nowcasting references include SEVIR, PredRNN/PredRNN-V2, Earthformer, and pySTEPS. They are useful for spatiotemporal architectures, event verification, sharpness, and spectral diagnostics, but should not be treated as evidence that the 6 h–multi-day forecast-conditioned problem is solved.

## 4. Closest research family: forecast-conditioned post-processing/downscaling

This is the largest blind spot in the current reference set.

Classical meteorology has a long tradition of **Model Output Statistics (MOS)** and ensemble post-processing. Modern ML extends the same setting: condition on a numerical forecast, then learn the conditional distribution of the verifying local/high-resolution weather.

Relevant baseline families include:
- raw/interpolated NWP precipitation;
- bias correction and quantile mapping;
- logistic/censored regression for occurrence and amount;
- Bayesian Model Averaging (BMA);
- Ensemble Model Output Statistics (EMOS);
- tree-based ML;
- CNN/U-Net/Transformer downscalers;
- generative super-resolution / diffusion / flow models.

A defensible baseline ladder is therefore:

1. raw/interpolated NWP precipitation;
2. simple bias correction / quantile mapping;
3. calibrated statistical probabilistic baseline (EMOS-like);
4. deterministic neural downscaler;
5. probabilistic neural/generative downscaler;
6. ablations of the proposed method;
7. global AI weather models only when variable, initialization, valid time, grid and verification are genuinely comparable.

## 5. Recommended target setting

### Primary task: operational forecast-conditioned prediction

At forecast initialization time `t0`, use:
- ECMWF IFS or AIFS atmospheric forecasts for future lead times;
- recent GMCP precipitation history available at `t0`;
- static geographical predictors such as topography and land/sea mask;
- optionally season/time encodings.

Predict:
- hourly or accumulated precipitation distribution on the 0.1° GMCP grid;
- lead times grouped into meaningful ranges such as 0–6 h, 6–24 h, 1–3 d, and 3–5 d where data permit;
- ensemble samples, quantiles, or exceedance probabilities `P(R > τ)`.

### Secondary task: analysis-conditioned upper bound

Use ERA5 analysis/reanalysis fields to estimate how much precipitation predictability is available when atmospheric-state error is reduced.

This is an **upper-bound experiment**, not an operational result. A model conditioned on future analysis can access atmospheric states that were not known to an operational forecaster at initialization.

## 6. Probabilistic precipitation should be first-class

Precipitation is zero-inflated, heavy-tailed, spatially intermittent, displacement-sensitive, and strongly heteroscedastic. RMSE/MAE alone are insufficient and can reward over-smoothed predictions.

Recommended verification:

**Distributional**: CRPS, Brier Score for threshold exceedance, reliability diagrams, rank/PIT histograms, sharpness vs calibration.

**Event/extreme**: CSI/ETS, POD, FAR, precision/F1, threshold-exceedance Brier scores, precision-recall curves for rare thresholds.

**Spatial**: Fractions Skill Score (FSS), neighborhood CSI, spectral/power-density diagnostics, and object/displacement-aware verification where practical.

**Intensity/tail**: conditional bias by rain-rate bin, tail quantile error, exceedance frequency, and peak-intensity distributions.

All thresholds must be defined in **physical units** after a validated inverse transformation.

## 7. Extreme precipitation should define the claim, not only the loss

A strong extreme-rain result should establish whether the method improves detection, amount calibration, spatial localization, probability calibration, and performance across meteorological regimes. Results should be stratified by lead time, region, season, threshold and regime (e.g. convection, orography, monsoon, tropical cyclones, fronts).

A single global CSI headline is not sufficient.

## 8. What “physics-informed” should mean for PhyDiff-Net

Use a claim hierarchy:

**Physics-constrained**: explicit physical evolution/constraint, dimensional consistency, measurable residual, and ablation showing benefit.

**Physics-inspired**: advection-aware operator, moisture/water-budget regularization, divergence/vorticity/moisture-flux features, or physically structured architecture, with empirical evidence.

Avoid claiming “physics-informed” solely because ERA5 meteorological variables are inputs; that is meteorological conditioning.

## 9. Dataset / benchmark map

| Dataset/system | Role | Main caveat |
|---|---|---|
| GMCP | 0.1°, hourly high-resolution precipitation target | document provenance, uncertainty, independent validation |
| ERA5 | reanalysis / analysis-conditioned atmospheric state | not an operational forecast |
| ECMWF IFS | operational NWP forecast | archive exact cycle, init time and lead |
| ECMWF AIFS | operational ML forecast | now a relevant conditioning/baseline source |
| WeatherBench2 / WeatherBench-X | standardized global forecast evaluation | not a precipitation-specialized HR benchmark |
| GPM IMERG | satellite precipitation | retrieval uncertainty |
| ERA5-Land | higher-resolution land reanalysis | model/reanalysis-derived precipitation |
| MRMS / radar | high-resolution regional precipitation | excellent for nowcasting, not global |
| station/gauge networks | point precipitation | representativeness/interpolation issues |
| CHIRPS | gauge+satellite precipitation | mainly daily / land-focused |

GMCP can be a major research asset if provenance, independent validation, leakage prevention, and extreme-event fidelity are rigorously established.

## 10. Required data/evaluation audit before SOTA claims

### Normalization
- Fit statistics on the **training split only**.
- Reuse identical statistics for validation/test.
- Store preprocessing statistics with checkpoints.
- Inverse-transform predictions and labels with the same training statistics.

### Thresholds
- Define rainfall thresholds in physical units (`mm/h`, `mm/6h`, etc.).
- Apply thresholds after inverse transformation unless a mathematically verified normalized equivalent is used.
- Do not mix accumulation durations.

### Temporal splits
- Prefer year-based held-out splits.
- Audit overlapping windows at split boundaries.
- Prevent target-time leakage through atmospheric fields unavailable at initialization.

### Conditioning source
Every result must state whether it uses ERA5 analysis/reanalysis, IFS forecast, or AIFS forecast, along with version/cycle, initialization time and lead time.

### Verification space
Report the native 0.1° GMCP space separately from regridded experiments, with exact accumulation period and units.

### Reproducibility
Archive experiment config, git commit, split manifest, normalization statistics, random seed, checkpoint and evaluation command.

## 11. Tiered reading list

### Tier 0 — required before finalizing the claim
1. GraphCast
2. Pangu-Weather
3. GenCast
4. Aurora
5. DGMR
6. NowcastNet
7. MetNet-2 / MetNet-3
8. WeatherBench2 / WeatherBench-X
9. ECMWF AIFS / AIFS ENS documentation
10. A modern review of ML precipitation/statistical downscaling

### Tier 1 — architecture / probabilistic / hybrid context
FourCastNet, FuXi, NeuralGCM, Stormer, ClimaX, FengWu, Prithvi Weather/Climate, EMOS/BMA calibration literature, and diffusion/generative downscaling.

### Tier 2 — exploratory
Regional foundation models, learned data assimilation, flow matching for weather, object-based precipitation verification, and extreme-value tail modeling.

## 12. Bottom line

The project should not primarily try to prove that “another AI weather model can beat GraphCast/Pangu”.

The most differentiated and testable question enabled by the available data is:

> **Can future NWP atmospheric guidance plus high-resolution precipitation history be transformed into calibrated 0.1° probabilistic precipitation forecasts that materially improve extreme-rain skill over raw NWP, statistical calibration, and deterministic neural downscaling?**

That question is narrow enough to evaluate rigorously, operationally relevant, and broad enough to justify architectural innovation if the experiments show genuine skill.
