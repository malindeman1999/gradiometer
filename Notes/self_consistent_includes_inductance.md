# Why Inductance Is Included in the Self-Consistent Solve

## Short answer

In this codebase, the self-consistent solve includes inductive physics through the **current-to-self-field feedback operator**, even though it does not explicitly integrate magnetic energy \(W=\int B^2/(2\mu)\,dV\).

## Where this appears in code

### 1) Optional Step-1 local reactance term (not required for self-consistent inductance)

In the GUI preprocessing, a local sheet reactance can be added:

- `workflow/workflow_nonuniform_gui.py:221`
  - `X_s = inductance_scale * omega * mu0 * R / 2`

This is a modeling closure in the sheet impedance \(Z_s = R_s + iX_s\). It is optional and controlled by `inductance_scale`.

### 2) Self-consistent operator solve (the key inductive feedback)

In Step 5, the solve constructs

- `workflow/workflow_nonuniform_gui.py:586`
  - `A = I - diag(S_diag) @ mixing_matrix`

and solves for total radial field coefficients with this feedback included.

The same structure appears in the iterative form via

- `workflow/workflow_nonuniform_gui.py:752`
  - `SM = diag(S_diag) @ mixing_matrix`

### 3) Physics inside `S_diag`

`S_diag` is not arbitrary; it is the spherical MQS self-field kernel for toroidal surface currents:

- `europa_model/solvers.py:259`
  - `S[l] = mu0 / ((2*l+1)*l*(l+1))`

This is exactly the mode-dependent coefficient that maps toroidal current mode amplitude to its own normal magnetic field on the sphere.

Supporting formula is documented in:

- `europa_model/inductance.py:39`
  - `B_n,self = -mu0 * J_lm / ((2l+1) l(l+1))`

## Why this is inductance without explicit energy integration

Inductance can be represented in equivalent ways:

1. Energy form: \(W = \tfrac12 L I^2\)
2. Flux/self-field form: current creates self-field/self-flux that reacts back on the current-driving field

The self-consistent solve uses representation (2):

- Faraday map from radial \(B\) to toroidal \(E\):
  - `europa_model/solvers.py:129`, `europa_model/solvers.py:250`
  - factor \(F_l = -i\omega R/[l(l+1)]\)
- Admittance/mixing map \(M\): \(b \mapsto k\)
- Self-field map \(S\): \(k \mapsto b_{self}\)
- Closed-loop solve: \((I - SM)b_{tot} = b_{ext}\)

That closed loop is inductive back-reaction in operator form.

## Practical implication for GUI use

- With Step 5 enabled, the model already includes mode-dependent inductive feedback via `S_diag`.
- `inductance_scale` in Step 1 adds an additional local RL-like closure in the admittance term.
- So Step 1 `inductance_scale` and Step 5 self-consistent feedback are conceptually different contributions; they should not be assumed equivalent.

## Note on field-energy calculations

A separate field-energy computation (like in `toroidal_inductance/`) is still useful as:

- an independent diagnostic,
- a way to tabulate \(L_l\) under a chosen current-mode normalization,
- a cross-check against operator-based behavior.

But the absence of explicit \(\int B^2\,dV\) inside Step 5 does not mean inductance is absent.
