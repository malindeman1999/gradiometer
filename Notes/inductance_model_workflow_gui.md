# Inductance Model in `workflow_nonuniform_gui.py`

## What Physics the GUI Is Solving

The GUI solves a magnetoquasistatic (MQS), thin-shell, spherical-harmonic induction model on a sphere of radius R (Europa default R=1.56e6 m).

Core steps:

1. Ambient radial magnetic field B_r_ext(l,m) is prescribed in SH space (default single Jovian frequency).
2. Faraday mapping gives toroidal electric field
   E_lm = -(i*omega*R)/(l*(l+1)) * B_r_lm
3. Surface current uses sheet admittance \(Y_s\):
   K = Y_s * E
4. For non-uniform \(Y_s(\theta,\phi)\), multiplication in real space becomes a **Gaunt convolution** in SH space (mode coupling).
5. Self-consistent mode solves
   (I - S*M) b_tot = b_ext
   where \(M\) is the admittance/Faraday/Gaunt mixing operator and
   S_l = mu0 / ((2*l+1)*l*(l+1))
   maps toroidal current to radial self-field.

Magnetic field emitted by toroidal surface current uses analytic MQS spherical formulas:
B_r_emit_lm = -mu0/((2*l+1)*l*(l+1)) * K_lm
B_pol_emit_lm = mu0*l/(2*l+1) * K_lm
(with exterior radial decay (R/r)^(l+2) off-surface).

## How Admittance Is Built in Step 1

The GUI synthesizes a real conductivity field sigma_s(theta,phi) and then uses
Z_s = R_s + i*X_s
R_s = 1/sigma_s
X_s = inductance_scale * omega * mu0 * R / 2
Y_s = 1/Z_s

Important practical detail:

- GUI default is `inductance_scale = 0.0`, so by default X_s=0 and Y_s ~= sigma_s (purely real sheet admittance).

## How Accurate Is It?

Short answer: internally consistent and numerically accurate for the stated MQS thin-shell model; physically approximate for real Europa.

### 1) Numerical accuracy inside this model

Local checks run in this repo:

- `pytest -q tests/test_inductance.py`: passed (2 tests).
- `python tests/validation/test01_uniform_equivalence_check.py`: passed; uniform and spectral solvers agree within configured tolerances (typically `abs_tol=1e-6`, `rel_tol=1e-2`).
- `python tests/validation/test05_realspace_product_consistency.py`: passed; spectral solver matches independent Gaunt-convolution reference for tested random seeds.

So for solved equations and implemented operators, current code is behaving consistently at about 1e-6 absolute / 1e-3 to 1e-2 relative levels in these validation scripts.

### 2) Physical/model accuracy limits

This is not a full Maxwell + layered interior forward model. Main approximations:

- MQS regime (displacement current neglected).
- Thin conductive shell represented by surface admittance.
- Single-shell geometry; no explicit multilayer interior coupling.
- Harmonic truncation at user-chosen `lmax`.
- Driver usually single-frequency and idealized low-order spatial mode.
- Optional impedance reactance term is user-scaled (`inductance_scale`), not derived from full layered electrodynamics.

Because of these assumptions, the model is best interpreted as a controlled reduced-order induction model. It is typically good for algorithmic comparisons, sensitivity studies, and workflow development, but should be cross-validated against higher-fidelity layered/finite-element or analytic spherical Bessel solutions before claiming geophysical absolute accuracy.

## Practical Guidance

- For strongest internal fidelity, use self-consistent solve (Step 5), sufficient `lmax`, and check condition number logs.
- Use iterative solve (Step 6) only when its series terms decrease with order.
- Treat `inductance_scale` as a calibration/sensitivity knob unless you have an external physical calibration target.
