# Recommended Scientific Positioning and Experiment Plan

> Proposed discussion document for the next WeatherLLM / PhyDiff-Net iteration.

## 1. Recommended project statement

### Primary scientific claim to test

**Forecast-conditioned probabilistic precipitation downscaling/post-processing**

Given atmospheric forecasts available at initialization time and recent high-resolution precipitation history, learn the conditional distribution of future precipitation on the 0.1° GMCP grid, with explicit emphasis on calibrated heavy/extreme precipitation.

This framing is preferred over “independent global weather forecasting” because it matches the strongest project asset: high-resolution precipitation supervision.

## 2. Three settings that must remain separate

### A. Operational forecast-conditioned
`ECMWF forecast at t0 -> future high-resolution GMCP precipitation`

This should be the primary deployment/scientific setting.

### B. Analysis-conditioned
`ERA5 analysis/reanalysis -> future high-resolution GMCP precipitation`

Use this as an upper-bound / predictability experiment. Do not describe it as operational unless every conditioning field was actually available at initialization.

### C. Independent forecasting
`current state -> model predicts future atmospheric state + precipitation`

This is GraphCast/Pangu/AIFS territory and is a substantially larger scope. Recommendation: do not make this the first paper objective unless post-processing experiments show the problem is saturated.

## 3. Scientific hypotheses

### H1 — atmospheric forecast conditioning adds real precipitation skill
A model conditioned on future NWP atmospheric guidance will outperform GMCP-history-only baselines, raw/interpolated NWP precipitation, and simple statistical correction on both average and extreme precipitation.

### H2 — probabilistic generative modeling improves heavy-tail quality
Compared with deterministic MSE/MAE regression, a probabilistic model will reduce CRPS/Brier Score, improve exceedance reliability, preserve sharper spatial/intensity distributions, and improve rare-event recall without unacceptable FAR.

This hypothesis justifies diffusion/flow/GAN-style generation; architecture novelty alone is not the scientific claim.

### H3 — recent high-resolution precipitation history adds value beyond NWP atmosphere
At matched atmospheric guidance, adding recent GMCP history should improve short-lead localization and intensity, especially in the 0–24 h range. The effect should be measured as a function of lead time.

### H4 — gains are regime-dependent
Improvements may concentrate in convection, orographic precipitation, monsoon rainbands, tropical cyclones, and fronts. Report stratified skill rather than only global means.

### H5 — the analysis-conditioned vs forecast-conditioned gap is informative
The performance gap estimates how much precipitation error is attributable to atmospheric forecast uncertainty. This is a useful diagnostic even if the final model does not dominate every baseline.

## 4. Minimum baseline matrix

| Baseline | Purpose |
|---|---|
| persistence / climatology | lower bound and leakage sanity check |
| raw NWP precipitation regridded to 0.1° | operational reference |
| bias correction / quantile mapping | classical post-processing |
| EMOS-like/calibrated probabilistic baseline | probability calibration reference |
| deterministic U-Net/Transformer downscaler | architecture/capacity baseline |
| deterministic proposed backbone | isolate stochastic-generation value |
| probabilistic diffusion/generative model | proposed family |
| no-GMCP-history ablation | test H3 |
| no-atmosphere ablation | test H1 |
| no-physics-constraint ablation | test actual physics contribution |
| no-extreme weighting/branch ablation | test extreme-specific mechanism |

GraphCast/Pangu/GenCast should be reported only when target variable, initialization, valid time, grid and verification data are fairly matched.

## 5. Forecast-horizon protocol

Do not aggregate all lead times into one number. Suggested bins:
- 0–6 h: overlaps with nowcasting behavior;
- 6–24 h: strong local-history + NWP interaction;
- 24–72 h: short/medium-range forecast-conditioned behavior;
- 3–5 d: atmospheric forecast uncertainty increasingly dominant;
- beyond 5 d: exploratory unless skill is robust.

If current labels are 6-hour accumulations, first validate exact 6-hour windows before moving to hourly targets.

## 6. Evaluation contract

Freeze an evaluation contract before model comparison.

### Deterministic continuous
MAE, RMSE, correlation, bias.

### Threshold/event
CSI, POD, FAR, precision, F1, ETS where appropriate, at a fixed threshold table in physical units.

### Probabilistic
CRPS, threshold Brier Score, reliability, rank/PIT diagnostics, ensemble spread vs error.

### Spatial
FSS at multiple neighborhood scales, neighborhood CSI, and power-spectrum/PSD or equivalent sharpness diagnostics.

### Stratification
At minimum: lead time, season, region, rain-rate bin, and extreme-event subset.

## 7. Data protocol

### Split
Use non-overlapping year-based splits. Avoid choosing the final held-out split primarily to match historical headline benchmark years.

### Normalization
Fit preprocessing statistics on training data only and reuse unchanged for validation/test.

### Target definition
Every experiment must encode units, accumulation window, valid-time convention, missing-value rule, transforms/clipping, and inverse transform.

### Atmospheric variables
Prefer interpretable groups: humidity/moisture, vertical velocity, winds, temperature, geopotential, surface pressure, CAPE/instability variables where reliable, and integrated vapor/moisture-flux features. Ablate groups rather than assuming all channels help.

## 8. Architecture recommendation

Do **not** start by increasing the 467M-parameter model. First establish a trustworthy benchmark with a compact architecture.

### Stage 1 — deterministic conditional baseline
Use a multiscale U-Net/Transformer with atmospheric-forecast encoder, high-resolution GMCP-history encoder, static-feature encoder, and fused 0.1° decoder. Goal: prove the conditioning/data pipeline.

### Stage 2 — probabilistic head
Add a quantile/distributional head, latent-variable generator, diffusion decoder, or flow-matching decoder. Goal: prove probabilistic value via proper scores and calibration.

### Stage 3 — physical inductive bias
Only after the baseline is stable, add an explicit advection constraint, moisture/water-budget constraint, or physically structured evolution operator. Goal: make a measurable physics claim.

### Stage 4 — scale
Increase model capacity/pretraining only when ablations establish the value of each mechanism.

## 9. Immediate reproducibility / implementation audit checklist

The following should be verified before interpreting current extreme-rain results:

- [ ] normalization fitted on training data only;
- [ ] validation/test reuse training normalization statistics;
- [ ] inverse transform uses exactly the same training statistics;
- [ ] extreme thresholds are defined in physical units;
- [ ] thresholds correspond to the correct accumulation duration;
- [ ] ERA5 analysis and ECMWF forecast fields are not mixed in one claim;
- [ ] no future fields unavailable at initialization leak into inputs;
- [ ] train/validation/test windows do not overlap across boundaries;
- [ ] interpolation direction and native target grid are documented;
- [ ] all headline metrics are computed after a validated inverse transform;
- [ ] missing/dry pixels are treated consistently;
- [ ] checkpoints include preprocessing metadata.

Any failed item should block a SOTA claim until fixed.

## 10. Decision gates

### Gate A — is the problem learnable?
A compact deterministic ERA5/NWP+GMCP model must beat raw NWP and statistical correction. If not, architecture novelty is premature.

### Gate B — is high-resolution history useful?
`atmosphere + GMCP history` must beat `atmosphere only`. If not, revise the value proposition of GMCP temporal conditioning.

### Gate C — is probabilistic generation useful?
A generative model must beat deterministic baselines on proper probabilistic scores and tail behavior. If not, do not claim diffusion is necessary.

### Gate D — is the physics component real?
The physics-constrained model must beat the same model without the constraint and show reduced physical residual or improved regime skill. If not, use “physics-inspired” rather than “physics-constrained”.

### Gate E — is the result operationally meaningful?
Forecast-conditioned results should remain strong with archived real forecasts rather than future ERA5 analysis.

## 11. Proposed first-paper story, if evidence supports it

1. GMCP enables long-term high-resolution precipitation supervision.
2. Existing global AI weather models are strong at atmospheric dynamics but local precipitation extremes remain difficult.
3. Formulate precipitation forecasting as forecast-conditioned probabilistic downscaling/post-processing.
4. Learn fine-scale precipitation distributions from atmospheric forecast guidance + recent GMCP history.
5. Demonstrate improved calibrated threshold exceedance and spatial extreme-rain skill over raw NWP, statistical calibration, deterministic neural downscaling and relevant generative baselines.
6. Use ablations to isolate high-resolution history, stochastic generation and any physical constraint.
7. Present analysis-conditioned experiments as an upper-bound diagnostic, clearly separate from operational results.

The strongest result is not “lower global RMSE than GenCast”. It is **better calibrated and spatially useful extreme-precipitation probabilities at 0.1° under a fair operational conditioning protocol**.
