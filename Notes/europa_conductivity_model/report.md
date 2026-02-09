# Europa Surface-Conductivity Modeling Report

Date: 2026-02-07
Scope: Practical static-snapshot physical model for spatially variable effective ocean/sheet conductivity in the nonuniform workflow GUI, aimed at demonstrating gradiometer discrimination power.

## 1. Problem Framing
Your solver currently uses an effective sheet conductivity/admittance on the sphere. The key modeling need is to replace purely synthetic heterogeneity with physically motivated spatial structure tied to known Europa processes:
- induced-ocean electrical response (global constraint)
- composition heterogeneity (NaCl vs sulfate/carbonate influence)
- localized exchange features (chaos/plume source regions)
- transport by ocean circulation (large-scale anisotropy encoded in a single snapshot)

For this phase, the conductivity map is intentionally a fixed snapshot (no time evolution) to highlight what a gradiometer can resolve from spatial structure alone.

## 2. Evidence-Constrained Priors (What We Can Defend)
1. Europa must contain a conductive layer to explain induced magnetic signatures (Khurana et al., 1998; Kivelson et al., 2000).
2. Conductivity lower bounds from induction+gravity coupling are at least O(10^-2 to 10^-1 S/m) (Zimmer et al., 2000; Hand & Chyba, 2007).
3. Surface composition is spatially heterogeneous; irradiated NaCl signatures are associated with geologically young chaos terrain (Trumbo et al., 2019; follow-on UV confirmation work).
4. Plume-like activity has observational support (Roth et al., 2014; Sparks et al., 2017; Jia et al., 2018), implying potentially localized ocean-surface exchange pathways.
5. Ocean circulation on Europa is expected to produce structured lateral transport and nonuniform heat/solute patterns (Soderlund et al., 2014; Soderlund, 2019 review; Gissinger & Petitdemange, 2019).
6. Hydrothermal and water-rock interaction can alter sulfur/chloride balance and redox chemistry, supporting regional chemistry contrasts over time (e.g., Nakamura & Tajika, 2021; Daswani et al., 2021).

## 3. Recommended Model Form
Use a positive-definite multiplicative decomposition for effective sheet conductivity:

sigma_eff(theta, phi) = sigma0 * exp(X_chem + X_exchange + X_flow + X_bg)

Where:
- sigma0: global baseline (calibrated to induction amplitude/phase)
- X_chem: composition-driven field (slowly varying; hemispheric + regional)
- X_exchange: localized anomalies (chaos/plume/hotspot regions)
- X_flow: transported anisotropy (jet/cell-like patterns in the chosen snapshot)
- X_bg: residual stochastic field (small amplitude, band-limited)

Why this form:
- always positive conductivity
- additive controls in log-space are easy to regularize
- separates interpretable physical processes

## 4. Spatial Components
## 4.1 Composition field X_chem
- Start with low-degree SH terms (l <= 6) to represent broad hemispheric differences.
- Add optional regional masks for known chaos terrains.
- Prior: smooth, slowly varying, persistent across runs.

## 4.2 Exchange field X_exchange
Represent each active exchange zone i by a spherical Gaussian:
A_i * exp[-(Delta_i(theta,phi)^2)/(2*w_i^2)]

- center: vent/chaos candidate location
- w_i: 2 to 15 degrees (user-set or sampled)
- A_i: snapshot anomaly amplitude

## 4.3 Flow field X_flow
Use a low-order streamfunction proxy to impose elongated structures:
- build psi(theta,phi) from low-degree SH coefficients
- set X_flow proportional to directional derivative of tracer-like field along flow
- or simpler: use anisotropic kernels aligned with latitude/equator for first version

This captures current-related nonuniformity without full ocean GCM coupling.

## 5. Snapshot-Only Assumption
No temporal evolution is required for this use case. The model is a single physically plausible conductivity realization used to test spatial detectability and inversion sensitivity for gradiometer measurements.

## 6. Mapping to Existing GUI Inputs
Current Step 1 uses mean conductivity + single mode RMS perturbation. Replace with a mode selector:
- `conductivity_model = synthetic_sh | europa_physical_v1`

For `europa_physical_v1`, expose:
- `sigma0`
- `chem_contrast` (0..1)
- `n_exchange_sites`
- `exchange_amp`
- `exchange_width_deg`
- `flow_anisotropy`
- `random_seed`

Keep `extra inducance scale` separate from conductivity physics.

## 7. Calibration Strategy
1. Calibrate sigma0 to reproduce induction amplitude/phase envelope from legacy Galileo constraints.
2. Tune heterogeneity amplitudes so modeled magnetic perturbations remain consistent with plausible residual structure, not grossly violating global induction fits.
3. Use composition priors (chloride-linked terrains) to constrain where large anomalies are allowed.
4. Run sensitivity sets: no-plume, plume-only, flow-only, combined.

## 8. Minimal v1 Implementation Plan
1. Add `build_europa_conductivity_field(config, grid)` in `workflow/workflow_nonuniform_gui.py` or a new module.
2. Build log-conductivity field from:
- low-degree SH background
- N Gaussian exchange patches
- optional equatorial anisotropy term
3. Exponentiate and renormalize area-weighted mean to target sigma0.
4. Save component maps in state for diagnostics:
- `sigma_eff`, `x_chem`, `x_exchange`, `x_flow`
5. Add Step 1b plots for component decomposition.

## 9. Risks and Limits
- Surface spectra are an indirect proxy for ocean composition.
- Plume recurrence/location remains uncertain.
- Lateral conductivity contrasts in the ocean are not directly measured yet.
- Overfitting heterogeneity to sparse magnetic data is easy; regularization is required.

## 10. Recommendation
Adopt `europa_physical_v1` as a prior-driven synthetic-physical hybrid snapshot model now, then tighten priors using Europa Clipper magnetic/plasma/imagery products as they become available.
