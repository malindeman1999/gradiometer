# Exactness Review: V-Toroidal Coupling Derivation

This note summarizes a technical review of the conductivity-to-magnetic-gradient derivation using the toroidal-normalized V-coupling kernel. The focus is whether the mathematics is exact versus approximate, and where non-exactness enters.

---

## Executive Summary

Yes. The derivation is mathematically exact within the stated formulation.

- Analytic identities used (Gaunt integrals, surface-gradient identity, toroidal normalization) are exact.
- No heuristic term-dropping is introduced in the derivation itself.
- Non-exactness enters only through numerical implementation choices.

---

## 1. Key Identity: Surface-Gradient Gaunt Relation

The central identity is

$$
\int_\Omega Y_\alpha\,(\nabla_s Y_a^* \cdot \nabla_s Y_b)\,d\Omega
= \tfrac12\,[\ell_a + \ell_b - \ell_\alpha] \, G_{a\alpha b},
$$

with

$$
\ell(l) = l(l+1),
$$

and

$$
G_{a\alpha b} = \int_\Omega Y_a^*\,Y_\alpha\,Y_b\,d\Omega.
$$

### Exactness

- This identity is exact.
- It follows from the spherical-harmonic eigenvalue relation on the sphere, integration by parts on the sphere, and SH orthogonality.
- No asymptotic limit or dropped term is used.

---

## 2. Toroidal Basis Normalization

For the three degree symbols that appear throughout:
- $L$ (or $l_a$) denotes the output/current harmonic degree (mode index of $K_a$ and emitted-field mode index),
- $l_0$ (or $l_\alpha$) denotes the conductivity/admittance harmonic degree,
- $l_b$ denotes the input magnetic-field harmonic degree (for $b_b$ and $e_b$).

Define toroidal VSH as

$$
\mathbf T_{lm} = \frac{\hat{\mathbf r} \times \nabla_s Y_{lm}}{\sqrt{\ell(l)}}.
$$

This is a basis normalization choice, not an approximation.

With this basis, the toroidal-normalized V kernel is

$$
\widetilde V_{a\alpha b}
= \frac{\tfrac12\,[\ell_a + \ell_b - \ell_\alpha]}{\sqrt{\ell_a\,\ell_b}}\,G_{a\alpha b},
$$

with convention

$$
\widetilde V_{a\alpha b}=0 \quad \text{if} \quad \ell_a\ell_b=0.
$$

### Exactness

- The factor $(\ell_a\ell_b)^{-1/2}$ is exact for this normalization.
- This changes representation, not physics.

---

## 3. Mixing Operator and Frequency Dependence

Define

$$
C^{(V)}_{ab\alpha} = F_b\,\widetilde V_{a\alpha b},
$$

with

$$
F_b = -\frac{i\,\omega R}{l_b(l_b+1)} \quad (l_b\ge 1),
\qquad F_b=0 \text{ for } l_b=0.
$$

Then

$$
M^{(V)}_{ab} = \sum_\alpha A_\alpha\,F_b\,\widetilde V_{a\alpha b}.
$$

### Structural Consistency

- All $\omega R$ dependence appears once, in $F_b$.
- $M^{(V)}$ is linear in admittance coefficients $A_\alpha$.
- Replacing Gaunt-only coupling with V-toroidal coupling does not add new approximations.

---

## 4. Self-Consistent Solve

$$
\mathcal A^{(V)}_{ab} = \delta_{ab} - S_{ac} M^{(V)}_{cb},
$$

$$
\mathcal A^{(V)}\,b_{\mathrm{tot}} = b_{\mathrm{ext}},
$$

$$
K = M^{(V)}\,b_{\mathrm{tot}}.
$$

Given the model assumptions, this is an exact linear-algebra consequence.

---

## 5. Where Approximations Enter

Approximations are implementation-level:

- Finite spectral truncation at $l_{\max}$.
- Numerical SH transforms and quadrature.
- Conditioning/regularization in the linear solve.
- Any approximate observation-space operators.

These do not come from the analytic V-coupling derivation itself.

---

## Final Verdict

The derivation is mathematically exact within its stated formulation, with approximations introduced only by numerical realization.
