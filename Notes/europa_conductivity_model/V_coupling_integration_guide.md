# Adding V-Coupling (Toroidal Normalization Only)

This note describes a controlled integration of V-coupling into the spectral solver, using only the toroidal-normalized kernel.

The goal is to keep the existing self-consistent solver structure and replace only the angular coupling kernel used to build the mixing matrix `M`.

## 1) What Stays the Same

The solver factorization is unchanged:

1. Faraday map: radial field coefficients `b_lm` to toroidal electric coefficients `e_lm`
2. Admittance coupling: `K(O) = Y_s(O) E_t(O)` in spectral form
3. Self-consistent closure: `A = I - diag(S) @ M`, then solve for `b_tot`, then `k = M @ b_tot`

Use exactly one Faraday factor:

$$
e_{lm} = F_l b_{lm}, \qquad
F_l = -\frac{i\omega R}{l(l+1)}.
$$

Do not move `omega*R` into the coupling kernel.

## 2) Kernel Definition (Toroidal-Normalized Only)

Mode labels:

- Output mode: `a = (L, M)`
- Admittance mode: `alpha = (l0, m0)`
- Input mode: `b = (lp, mp)`

Define:

$$
\ell(n) = n(n+1),
$$

$$
w(L,l0,lp) = \frac12\left[\ell(L) + \ell(lp) - \ell(l0)\right].
$$

Start from the scalar identity:

$$
\int Y_{\alpha} \, (\nabla_s Y_a^* \cdot \nabla_s Y_b) \, d\Omega
= w(L,l0,lp)\, G_{a\alpha b}.
$$

For toroidal-normalized basis

$$
\mathbf T_{lm} = \frac{\hat{\mathbf r} \times \nabla_s Y_{lm}}{\sqrt{\ell(l)}},
$$

the coupling kernel used in `M` is

$$
\tilde V_{a\alpha b}
= \frac{w(L,l0,lp)}{\sqrt{\ell(L)\,\ell(lp)}}\,G_{a\alpha b}.
$$

Implementation guards:

- If `L = 0` or `lp = 0`, set kernel contribution to zero (no physical toroidal `l=0` mode).
- Keep existing `l>=1` Faraday behavior as-is.

## 3) Code Integration Plan

### 3.1 Baseline (`europa_model/solvers.py`)

Modify `_build_mixing_matrix_spectral()` to accept:

- `coupling: Literal["gaunt", "v_toroidal"]` (default `"gaunt"`)

Add helpers:

```python
def _ell_int(l: int) -> int:
    return l * (l + 1)

def _v_weight(L: int, l0: int, lp: int) -> float:
    return 0.5 * (_ell_int(L) + _ell_int(lp) - _ell_int(l0))

def _v_toroidal_factor(L: int, l0: int, lp: int) -> float:
    denom = (_ell_int(L) * _ell_int(lp)) ** 0.5
    if denom == 0.0:
        return 0.0
    return _v_weight(L, l0, lp) / denom
```

In the nested accumulation loops, replace

```python
accum += Y_r * F_lprime * G_val
```

with

```python
if coupling == "gaunt":
    kernel = G_val
elif coupling == "v_toroidal":
    kernel = _v_toroidal_factor(L, l0, lprime) * G_val
else:
    raise ValueError(f"Unknown coupling mode: {coupling}")
accum += Y_r * F_lprime * kernel
```

### 3.2 Precomputed Sparse Path (`europa_model/solver_variants/solver_variant_precomputed.py`)

Apply the same toroidal factor in `_build_mixing_matrix_precomputed_sparse()` using sparse indices:

- `L = idx[0]`
- `l0 = idx[2]`
- `lp = idx[4]`

Vectorized factor:

$$
\text{factor} = \frac{\tfrac12[\ell(L)+\ell(lp)-\ell(l0)]}{\sqrt{\ell(L)\ell(lp)}},
$$

with `factor = 0` where denominator is zero.

Then use:

```python
contrib = vals * y_vals * f_vals * factor
```

This is required so GUI/workflow runs match the baseline option.

### 3.3 Solver Entry Points

Thread `coupling` through:

- `solve_spectral_first_order_sim()`
- `solve_spectral_self_consistent_sim()`
- `solve_spectral_self_consistent_sim_precomputed()`
- workflow builders that currently call Gaunt sparse mixing directly

Suggested variant names:

- `spectral_first_order_v_toroidal`
- `spectral_self_consistent_v_toroidal`
- `spectral_self_consistent_precomputed_v_toroidal`

## 4) Validation Plan

Run A/B with identical inputs:

- `gaunt`
- `v_toroidal`

Compare:

$$
\epsilon_M = \frac{\|M_V - M_G\|_F}{\|M_V\|_F},
\qquad
\epsilon_b = \frac{\|b_V - b_G\|}{\|b_V\|},
\qquad
\epsilon_k = \frac{\|k_V - k_G\|}{\|k_V\|}.
$$

Low-frequency sanity:

$$
\omega \to 0 \implies F_l \to 0 \implies k \to 0.
$$

Also add one regression test that checks sparse and dense implementations agree for the same coupling mode at low `lmax`.

## 5) Exactness and Limits

What is exact in this upgrade:

- The gradient identity
  $$
  \int Y_{\alpha}(\nabla_s Y_a^*\cdot\nabla_s Y_b)d\Omega
  = \tfrac12[\ell_a+\ell_b-\ell_\alpha]G_{a\alpha b}
  $$
  is exact.
- The toroidal normalization factor
  $$
  (\ell_a\ell_b)^{-1/2}
  $$
  is exact for the normalized basis definition used above.

What is still approximate at model level:

- The full vector product projection can include additional geometric terms depending on the exact state/basis conventions used elsewhere.
- The solver remains a spectral-truncation model (`lmax` finite).
- Any numerical quadrature/transforms and conditioning effects are still numerical approximations.

Bottom line:

- `v_toroidal` is a physically better angular kernel than plain Gaunt for toroidal coupling.
- It is exact for the gradient-weighted part under this basis definition.
- It is not automatically the full, final "all vector effects included" model unless all operator definitions are made fully consistent end-to-end.

## 6) Making Operators Consistent End-to-End

This section is a code-grounded consistency contract for this repository. The objective is that every operator in the chain uses the same basis, indexing, sign convention, and units:

$$
b \xrightarrow{F} e \xrightarrow{M} k \xrightarrow{S} b_{\mathrm{self}}
\xrightarrow{(I-SM)^{-1}} b_{\mathrm{tot}} \xrightarrow{M} k
\xrightarrow{\mathcal{B}} B.
$$

### 6.1 Use one phasor sign convention everywhere

`europa_model/solvers.py` uses:

$$
\nabla\times E = -i\omega B, \qquad
F_l = -\frac{i\omega R}{l(l+1)}.
$$

Keep this sign in:

- solver docstrings and notes,
- any derived self-field formulas,
- diagnostics that compare curl/time-derivative relations.

If a source document uses `+i\omega`, reconcile it explicitly before coding.

### 6.2 Keep one toroidal basis normalization

For this V-upgrade, enforce:

$$
\mathbf T_{lm} = \frac{\hat{\mathbf r}\times\nabla_s Y_{lm}}{\sqrt{l(l+1)}}.
$$

Then keep the same choice in:

- `M` build (dense and sparse),
- any transforms/projections that interpret `k_{lm}` as toroidal coefficients,
- any gradient/curl operators used for validation.

Important current-code note: `europa_model/transforms.py` builds a numerical VSH basis from raw gradients (`cross(r_hat, grad)`) without explicit `1/\sqrt{l(l+1)}` normalization. If this path is used to validate or compare toroidal coefficients, either normalize that basis or treat it as a separate approximate basis and do not mix it with analytic normalized formulas.

### 6.3 Enforce dense/sparse kernel parity

The same kernel must be used in both:

- `europa_model/solvers.py::_build_mixing_matrix_spectral`
- `europa_model/solver_variants/solver_variant_precomputed.py::_build_mixing_matrix_precomputed_sparse`

For `v_toroidal`, both paths must apply:

$$
\tilde V_{a\alpha b}
= \frac{\tfrac12[\ell(a)+\ell(b)-\ell(\alpha)]}{\sqrt{\ell(a)\ell(b)}}G_{a\alpha b},
\qquad \ell(n)=n(n+1),
$$

with zero contribution when the denominator is zero.

### 6.4 Lock the self-field sign and matrix form

Current code uses:

$$
A = I - S M.
$$

`_build_self_field_diag()` returns positive

$$
S_l = \frac{\mu_0}{(2l+1)l(l+1)},
$$

while `inductance.modal_radial_self_field()` includes a negative sign in the direct mapping. This can be consistent, but only if the global equation sign bookkeeping is fixed and documented once. Add one canonical sign derivation in the docs and align all helper comments to it.

### 6.5 Keep index/layout contracts fixed

All spectral operators should use one layout:

$$
\text{storage index} = (l, m+l_{\max}),
$$

and one flattened order:

$$
[(0,0), (1,-1), (1,0), (1,1), \dots].
$$

In this repo, `_flatten_lm` and `_unflatten_lm` define the contract; sparse row/column formulas in the precomputed path must remain algebraically equivalent to that order.

### 6.6 Separate exact solver operators from approximate observation stubs

`europa_model/observation.py` currently labels some gradient mappings as approximate/stub behavior. Do not use those routines to certify end-to-end operator exactness for V-coupling. For consistency validation, compare operators in spectral space first (`M`, `SM`, `A`, `k`) and only then compare observation outputs with clearly labeled approximation tolerance.

### 6.7 Recommended consistency tests (minimum set)

- Dense vs sparse `M` equality for both `gaunt` and `v_toroidal` at low `lmax`.
- Sign sanity: `\omega \to 0` implies `F\to 0` and `k\to 0`.
- Uniform-admittance reduction: spectral solver should match uniform-mode formulas when only `l=0,m=0` admittance is active.
- Self-field closure consistency: direct `A^{-1}` solve vs iterative series should agree within tolerance.

## 7) Implementation Checklist

- [ ] Add `coupling` option with values `gaunt` and `v_toroidal`
- [ ] Implement toroidal factor in `solvers.py` mixing builder
- [ ] Implement the same factor in precomputed sparse builder
- [ ] Thread `coupling` through solver and workflow entry points
- [ ] Add low-`lmax` dense-vs-sparse equivalence test for `v_toroidal`
- [ ] Add A/B comparison report for `M`, `b_tot`, and `k`
