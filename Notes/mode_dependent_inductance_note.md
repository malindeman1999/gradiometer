# Mode-Dependent Inductance for the Workflow GUI

## Why change from a single inductance scale?

Current Step 1 uses one reactance for all modes:

- `X_s = omega * mu0 * R / 2 * inductance_scale`

That makes every spherical-harmonic degree respond with the same inductive phase lag.  
But in the spectral solver, self-field coupling is already degree-dependent, so a degree-dependent inductance model is more physically consistent.

## Core idea

Replace one scalar reactance with a per-degree reactance:

- `X_l = omega * L_l`
- `L_l = mu0 * R * c_l`

where `c_l` is a chosen dimensionless degree-dependent coefficient.

Then use an l-dependent admittance in spectral space:

- `Y_lm_eff = Y_lm_ohmic / (1 + i * omega * tau_l)`
- `tau_l = L_l * Y_ref`

`Y_ref` is a reference sheet admittance scale (for example the mean of `sigma_s`), used to make `tau_l` dimensional.

This acts like an RL low-pass per degree: high-l modes can be damped/phase-lagged differently than low-l modes.

## Physically motivated choices for `c_l`

You can pick one of these closures:

1. Geometry-like decay:
- `c_l = 1 / (2*l + 1)`

Why this is physically plausible:
- In spherical harmonic induction, many geometric coupling factors carry `(2*l+1)` in the denominator.
- Higher `l` modes represent smaller spatial scales; their global inductive leverage is weaker than low `l` large-scale loops.
- This gives a conservative, monotone reduction of inductive loading with degree without being too aggressive.

2. Match solver self-coupling trend (from existing operators):
- In code, `F_l ~ 1/(l(l+1))`, `S_l ~ 1/((2l+1) l(l+1))`
- So feedback path scales like `S_l*F_l ~ 1/((2l+1)[l(l+1)]^2)`
- A matching closure is:
  `c_l ~ 1 / ((2*l + 1) * (l*(l+1))^2)`

Why this is physically plausible:
- This is the most internally consistent with your current spectral MQS solver, because it mirrors how inductive feedback already weakens with degree through `S_l` and Faraday response through `F_l`.
- High-`l` currents have shorter spatial loops and stronger derivative penalties (`l(l+1)` factors), so their effective inductive back-reaction is expected to fall off rapidly.
- If your goal is "do not fight the existing operator physics," this is the most coherent choice.

3. Calibrated family:
- `c_l = c0 / (2*l + 1)^p`, with `p` fit to benchmark or data.

Why this is physically plausible:
- Real oceans/shells are not ideal single-parameter media; unresolved 3-D structure, finite thickness, and forcing spectra can shift degree dependence away from simple analytic forms.
- A one-parameter exponent `p` keeps the model constrained (smooth monotone scaling) while allowing empirical correction.
- It is a standard reduced-order strategy: preserve known asymptotic trend (`2*l+1` denominator) and fit only the strength of the decay.

Option 1 is the simplest physically reasonable start. Option 2 is most aligned with current internal operator scaling.

## What each scaling implies physically

- Slower decay with `l` (small `p`, or pure `1/(2l+1)`) means you assume fine spatial structure still carries meaningful inductive lag.
- Faster decay with `l` (self-coupling-like scaling) means you assume induction is dominated by large-scale modes and small-scale modes are mostly resistive.
- If your emitted field diagnostics show excessive high-`l` phase lag, your `c_l` likely decays too slowly.
- If high-`l` response is too resistive/underpowered versus reference, your `c_l` likely decays too quickly.

## Limits of physical interpretability

These `c_l` laws are still closures, not exact Maxwell solutions.

- They are physically plausible because they respect spherical geometry trends and operator structure.
- They are not physically complete because full mode/frequency coupling in layered media is richer than any one-factor `c_l`.
- Treat them as controlled approximations to bridge between a single global scale and a full EM boundary-value solve.

## How this fits your current workflow

Current flow:
- build `sigma_s(theta,phi)` on grid
- convert to `Y_s` (complex) on grid
- SH transform to `Y_s(l,m)`
- build mixing matrix with Gaunt coefficients

Proposed mode-dependent variant:

1. Keep Step 1 ohmic base:
- `Y_grid_ohmic = sigma_s` (or existing complex form with global term off)

2. SH transform:
- `Y_lm_ohmic = SH[Y_grid_ohmic]`

3. Apply l-filter:
- for each degree `l`, multiply all `m` by
  `H_l = 1 / (1 + i * omega * tau_l)`
- `Y_lm_eff(l,m) = H_l * Y_lm_ohmic(l,m)`

4. Use `Y_lm_eff` in existing `step3_solve_currents`.

This avoids trying to define `1/sigma_lm` directly (not physically clean for nonuniform fields), and keeps the change linear and stable.

## Minimal implementation sketch

In Step 1 after `sigma_proj = sh_forward(...)`:

- compute `tau_l` vector for `l=0..lmax` (set `tau_0=0`)
- build complex `H_l` and broadcast over `m`
- set `Y_s = sigma_proj_complex * H_l`

with a user-selectable mode:
- `inductance_mode = global | l_dependent`

and parameters:
- `inductance_scale`
- `l_profile = geom | self_coupling | powerlaw`
- optional `power_p`.

## Validation plan

1. Recovery test:
- if `tau_l = const`, recover current global behavior.

2. Consistency test:
- compare first-order vs self-consistent in weak-response regime (existing validation style).

3. Stability test:
- check condition number of `A = I - S*M` vs l-profile and scale.

4. Sensitivity maps:
- compare phase and amplitude of emitted `B_rad` vs degree for each profile.

## Practical recommendation

Start with:

- `c_l = 1/(2*l+1)`
- `tau_l = inductance_scale * mu0 * R * c_l * Y_ref`
- `Y_ref = mean(sigma_s)`

Then compare against:

- current global model (`inductance_scale` scalar)
- self-consistent-only model (`inductance_scale = 0`).

That gives a controlled path to determine whether mode-dependent inductance improves realism without destabilizing the solve.
