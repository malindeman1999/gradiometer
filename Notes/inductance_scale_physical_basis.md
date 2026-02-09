# Physical Basis of `inductance_scale = 1` in the Workflow GUI

## Scope

This note is specifically about the Step 1 impedance model in:
- `workflow/workflow_nonuniform_gui.py`

and the meaning of setting:
- `inductance_scale = 1`

## Equation Used in the GUI

The GUI defines sheet impedance and admittance as

- `R_s = 1 / sigma_s`
- `X_s = inductance_scale * omega * mu0 * R / 2`
- `Z_s = R_s + i X_s`
- `Y_s = 1 / Z_s`

where:
- `sigma_s` is sheet conductivity (S),
- `omega` is angular frequency (rad/s),
- `R` is body radius,
- `mu0` is vacuum permeability.

So, `inductance_scale = 1` means:

- `X_s = omega * mu0 * R / 2`

equivalently an effective sheet inductance:

- `L_eff = mu0 * R / 2`

because `X = omega L`.

## Physical Basis

There are two layers of physics here:

1. General inductive reactance:
- For any inductive model element, reactive impedance is `X = omega L` (phasor-domain circuit relation from Faraday/Lenz law).

2. Thin-shell electromagnetic induction:
- MQS thin-sheet models reduce 3-D Maxwell induction to surface-conductance dynamics on a sphere/shell.
- In that framework, induction is geometry-dependent and generally mode-dependent (spherical harmonic degree dependence), not a universal single constant for all modes.

Therefore, `mu0*R/2` is best interpreted as an **order-of-magnitude geometric inductance closure** for the sheet, not a fundamental exact coefficient valid for all harmonic content.

## Is `inductance_scale = 1` Physically Correct?

Short answer: **it is physically plausible as a heuristic, but not generally exact.**

Why:

- It has correct dimensions and correct frequency/permeability/radius scaling for an inductive term.
- It captures that larger bodies and higher frequencies increase inductive reactance.
- But strict spherical-shell MQS induction is modal and solved by Maxwell boundary-value structure; a single `R/2` coefficient cannot exactly represent all degrees `l,m`.

In your workflow this is especially important because:

- Step 5 self-consistent solve already includes inductive feedback through the spectral self-field operator (`A = I - S*M`).
- Adding `inductance_scale=1` in Step 1 changes `Y_s` itself to be complex before that solve.
- So in self-consistent mode, this term acts as an extra modeled inductive loading choice, not a uniquely required physical constant.

## Practical Interpretation for This Codebase

- `inductance_scale = 0`: use purely Ohmic sheet admittance in Step 1; rely on solver self-field feedback for inductive behavior.
- `inductance_scale = 1`: include an additional local RL-like inductive component in `Y_s` with `L_eff = mu0*R/2`.
- `inductance_scale` should be treated as a calibration/sensitivity parameter unless you have an external validation target that supports that exact coefficient for your chosen forcing and harmonic regime.

## References

1. Sun, J. and Egbert, G. D. (2012), "A thin-sheet model for global electromagnetic induction," *Geophysical Journal International* 189(1), 343-356.  
   https://doi.org/10.1111/j.1365-246X.2012.05383.x  
   (Thin-sheet EM induction formulation on a sphere; surface-conductance framework.)

2. Kuvshinov, A. V., Avdeev, D. B., and Pankratov, O. V. (1999), "Modelling of electromagnetic fields in thin heterogeneous layers with application to field generation by volcanoes-theory and example," *Geophysical Journal International* 138(1), 125-136.  
   https://doi.org/10.1046/j.1365-246x.1999.00873.x  
   (Appendix discusses thin-sheet approximation history and MQS validity conditions; cites Maxwell and Price thin-sheet lineage.)

3. Khurana, K. K. et al. (1998), "Induced magnetic fields as evidence for subsurface oceans in Europa and Callisto," *Nature* 395, 777-780.  
   https://doi.org/10.1038/27394  
   (Planetary induction context: time-varying external field induces eddy currents in conductive subsurface layers.)

4. Zimmer, C., Khurana, K. K., and Kivelson, M. G. (2000), "Subsurface Oceans on Europa and Callisto: Constraints from Galileo Magnetometer Observations," *Icarus* 147(2), 329-347.  
   https://doi.org/10.1006/icar.2000.6456  
   (Simple spherical-shell induction modeling in the Europa/Callisto context; demonstrates why shell electrical structure controls induced response.)

5. Jackson, J. D., *Classical Electrodynamics* (3rd ed., Wiley, 1999/2021 adaptation listing).  
   https://www.wiley-vch.de/en/areas-interest/natural-sciences/physics-11ph/electricity-11pha/classical-electrodynamics-978-1-119-77076-3  
   (Faraday law / quasistatic electromagnetic foundations used by the phasor induction model.)
