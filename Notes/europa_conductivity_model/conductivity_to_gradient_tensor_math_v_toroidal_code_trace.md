# Conductivity to Magnetic-Gradient Map (V-Toroidal): Code Trace

This note parallels `conductivity_to_gradient_tensor_math_v_toroidal.md`, but for each numbered math step it points to the implementing code and confirms whether the implementation matches the stated equation.

Conventions for the three harmonic degrees used below:

- `L` (or `l_a`): output/current harmonic degree (mode index of `K_a`, and emitted-field mode index)
- `l0` (or `l_alpha`): conductivity/admittance harmonic degree
- `lp`/`l_b`: input magnetic-field harmonic degree (for `b_{lp mp}` and `e_{lp mp}`)

## 1. Conductivity to Admittance

Math step:

$$
A_n = \sigma_n, \qquad
A_\alpha = \int Y_\alpha^* A\,d\Omega \approx \sum_n w_n Y_\alpha^* A_n.
$$

Code:

1. Conductivity to admittance grid value (purely Ohmic path, `X_s=0`, so `Y=1/R=\sigma_s`):
   - `workflow/workflow_nonuniform_gui.py:261` (`_complex_sheet_admittance`)
   - `workflow/workflow_nonuniform_gui.py:268` (`X_s = 0.0`)
   - `workflow/workflow_nonuniform_gui.py:391` (call site)
2. SH projection to `A_alpha`:
   - `workflow/workflow_nonuniform_gui.py:393` (`Y_s = sh_forward(...)`)

Confirmation: implemented as written for this workflow configuration.

## 2. Faraday Map

Math step:

$$
e_{lm}=F_l b_{lm}, \qquad
F_l=-\frac{i\omega R}{l(l+1)},\; l\ge1,\; F_0=0.
$$

Code:

1. Direct map:
   - `europa_model/solvers.py:91` (`toroidal_e_from_radial_b`)
   - `europa_model/solvers.py:109` (`factor = -(1j * omega * radius) / ell`)
2. Diagonal operator used in matrix assembly:
   - `europa_model/solvers.py:226` (`_build_faraday_diag`)
   - `europa_model/solvers.py:231` (`F[l] = factor`, only for `l>=1`)

Confirmation: implemented exactly with a single `omega*R` placement in `F`.

## 3. V-Toroidal Kernel Weight

Math step:

$$
\widetilde V_{a\alpha b}
= \frac{\tfrac12[\ell(L)+\ell(l_b)-\ell(l_0)]}{\sqrt{\ell(L)\ell(l_b)}}\,G_{a\alpha b},
\quad \widetilde V=0 \text{ if } \ell(L)\ell(l_b)=0.
$$

Code:

1. Numerator/denominator and safe zero handling:
   - `europa_model/solver_variants/solver_variant_precomputed.py:35` (`_v_toroidal_factor_tensors`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:42` (`numer = 0.5 * (...)`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:43` (`denom = sqrt(...)`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:46` (`factor = numer / denom_safe`, masked to zero where invalid)

Confirmation: implemented exactly as the toroidal-normalized factor.

## 4. Mixing Matrix Assembly

Math step:

$$
M^{(V)}_{ab}=\sum_\alpha A_\alpha\,F_b\,\widetilde V_{a\alpha b}.
$$

Code (sparse precomputed path):

1. Sparse Gaunt index extraction (`L,l0,l_in`):
   - `europa_model/solver_variants/solver_variant_precomputed.py:105` (`_build_mixing_matrix_precomputed_sparse`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:123`
   - `europa_model/solver_variants/solver_variant_precomputed.py:128`
   - `europa_model/solver_variants/solver_variant_precomputed.py:130`
   - `europa_model/solver_variants/solver_variant_precomputed.py:132`
2. Multiply `Gaunt * factor * A_alpha * F_b`:
   - `europa_model/solver_variants/solver_variant_precomputed.py:162` (`y_vals`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:163` (`f_vals`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:168` (`kernel = vals * factor` for `v_toroidal`)
   - `europa_model/solver_variants/solver_variant_precomputed.py:172` (`contrib = kernel * y_vals * f_vals`)

Code (entry-point coupling threading):

1. Spectral solver accepts coupling mode and forwards it:
   - `europa_model/solvers.py:243`
   - `europa_model/solvers.py:248`
2. GUI solve selects coupling mode:
   - `workflow/workflow_nonuniform_gui.py:606`
   - `workflow/workflow_nonuniform_gui.py:607`
   - `workflow/workflow_nonuniform_gui.py:631`

Confirmation: implemented as the exact weighted replacement of Gaunt in `M`.

## 5. Self-Consistent Solve

Math step:

$$
\mathcal A = I - SM,\qquad
\mathcal A\,b_{\mathrm{tot}} = b_{\mathrm{ext}},\qquad
K = M\,b_{\mathrm{tot}}.
$$

Code:

1. Self-field diagonal `S_l`:
   - `europa_model/solvers.py:235`
   - `europa_model/solvers.py:239`
2. Matrix solve and current update (core solver):
   - `europa_model/solvers.py:388`
   - `europa_model/solvers.py:389`
   - `europa_model/solvers.py:390`
3. Same structure in precomputed variant:
   - `europa_model/solver_variants/solver_variant_precomputed.py:259`
   - `europa_model/solver_variants/solver_variant_precomputed.py:260`
   - `europa_model/solver_variants/solver_variant_precomputed.py:261`
4. Nonuniform workflow path:
   - `workflow/workflow_nonuniform_gui.py:715`
   - `workflow/workflow_nonuniform_gui.py:726`
   - `workflow/workflow_nonuniform_gui.py:730`

Confirmation: implemented as the same linear algebra described in the derivation.

## 6. Currents to Emitted Field

Math step:

$$
K \mapsto B \quad (\text{spectral operator}).
$$

Code:

1. Spectral mapping from toroidal currents to emitted field coefficients:
   - `europa_model/inductance.py:51` (`spectral_b_from_surface_currents`)
   - `europa_model/inductance.py:81`
   - `europa_model/inductance.py:86`
   - `europa_model/inductance.py:89`
   - `europa_model/inductance.py:94`

Confirmation: implemented (spectral emitted-field operator is explicit in code).

## 7. Field to Gradient Tensor / RSS Gradient

Math step in derivation note:

$$
G_{ip}(x)=\partial B_i/\partial x_p,\qquad
g=\left(G_{ip}\overline{G_{ip}}\right)^{1/2}.
$$

Code:

1. Gradient evaluation used by workflow rendering:
   - `europa_model/gradient_utils.py:294` (`rss_gradient_from_emit`)
   - `europa_model/gradient_utils.py:299` (calls toroidal field/gradient core)
   - `europa_model/gradient_utils.py:302` (builds gradient tensor stack)
   - `europa_model/gradient_utils.py:303` (RSS norm)
2. Plotting path:
   - `workflow/workflow_nonuniform_gui.py:780`
   - `workflow/workflow_nonuniform_gui.py:787`
3. Core gradient routine:
   - `europa_model/gradient_utils.py:179` (`_toroidal_field_and_gradients_spherical_core`)
   - `europa_model/gradient_utils.py:285` (`toroidal_field_and_gradients_spherical`)

Confirmation: implemented, but via the repository's toroidal gradient operator (not the explicit symbolic `H_{ipa}` tensor written in the derivation note).

## 8. End-to-End Equation-to-Code Match

Pipeline in code is:

$$
\sigma_n \to A_n \to A_\alpha \to M^{(V)} \to \mathcal A \to b_{\mathrm{tot}} \to K \to B \to G \to g.
$$

Main nonuniform workflow path:

1. `step1_build_grid_admittance` (conductivity/admittance + SH projection): `workflow/workflow_nonuniform_gui.py:338` and `workflow/workflow_nonuniform_gui.py:393`
2. `step3_solve_currents` (mixing + solve + emitted field): `workflow/workflow_nonuniform_gui.py:606`
3. `step4_render_gradient` / `step4_render_gradient_log100` (gradient magnitude rendering): `workflow/workflow_nonuniform_gui.py:780`, `workflow/workflow_nonuniform_gui.py:792`

## 9. Overall Confirmation

For the currently used sparse precomputed solver path, the code implements the V-toroidal math in the derivation:

1. Faraday factor appears once and in the correct place.
2. V-toroidal kernel factor matches the stated formula and zero-denominator guard.
3. Mixing matrix assembly uses `Gaunt * V-factor * A_alpha * F_b`.
4. Self-consistent solve uses `A = I - S M`, `b_tot = A^{-1} b_ext`, `K = M b_tot`.

Practical caveat:

1. Gradient rendering uses the repository's toroidal gradient operator in `gradient_utils.py` rather than an explicitly preassembled analytic `H_{ipa}` tensor object.
2. As in the derivation note, truncation (`lmax`) and numerical transforms/conditioning remain the non-exactness sources.
