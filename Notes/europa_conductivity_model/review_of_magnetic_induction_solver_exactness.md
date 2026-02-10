# Review: Exactness of the Magnetic Induction Solver

This note summarizes whether the attached solver constitutes a *mathematically exact* solution for mapping **normal magnetic field → induced surface currents via conductivity/admittance → emitted magnetic field (and downstream gradients)**.

The short answer is:

> **Yes — it is mathematically exact within the stated thin‑shell, magnetoquasistatic (MQS), spherical‑harmonic model.**

It is *not* an exact solution of full Maxwell equations in a finite‑thickness conductor, but no additional approximations are introduced beyond the model assumptions themselves.

---

## 1. Model Being Solved (Explicit Assumptions)

The solver is an exact linear solution of the following model:

- Frequency‑domain (phasor) **magnetoquasistatic** induction
- Conducting body treated as a **zero‑thickness spherical shell**
- Surface response encoded by a (possibly non‑uniform) **surface admittance** \(Y_s(\theta,\phi)\)
- Fields expanded in **spherical harmonics** up to a chosen \(l_{\max}\)

Within these assumptions, the solution is closed and linear.

---

## 2. Faraday Mapping (Normal \(B\) → Toroidal \(E\))

The solver applies Faraday’s law *exactly* in spherical‑harmonic space:

\[
E^{\mathrm{tor}}_{\ell m}
= -\frac{i\,\omega R}{\ell(\ell+1)}\,B^{\mathrm{tot}}_{r,\ell m},
\quad \ell>0
\]

- Implemented as a **diagonal operator in SH space**
- \(\ell=0\) mode is correctly suppressed
- No numerical differentiation or real‑space approximation is used

This step is mathematically exact for SH fields on a sphere.

---

## 3. Surface Current Response

Surface currents satisfy the boundary condition

\[
\mathbf{K}(\theta,\phi) = Y_s(\theta,\phi)\,\mathbf{E}^{\mathrm{tor}}(\theta,\phi)
\]

### Uniform admittance
- Relation is purely diagonal in \((\ell,m)\)
- Implemented analytically and exactly

### Non‑uniform admittance
- Multiplication in real space becomes **mode coupling in SH space**
- Implemented via:
  - Gaunt / Wigner‑3j convolution, or
  - Toroidal‑normalized kernel \(\tilde V\)

Both are **exact representations of spherical‑harmonic products**, not approximations.

---

## 4. Emitted Field and Self‑Consistency

The solver enforces self‑consistency through a *linear matrix solve*:

\[
(I - S M)\,b_{\mathrm{tot}} = b_{\mathrm{ext}}
\]

where:
- \(M\) encodes **Faraday + admittance + mode coupling**
- \(S_\ell = \mu_0 /[(2\ell+1)\ell(\ell+1)]\) maps surface currents to emitted normal field

This correctly accounts for:

> emitted normal field → modifies Faraday → modifies current → modifies emitted field

No iterative heuristics or partial updates are used; the closure is exact for a linear MQS model.

---

## 5. What Is *Not* Included (By Design)

These are **model limitations**, not mathematical errors:

- No finite‑thickness conductor or skin‑depth physics
- No displacement current or radiation terms
- No \(l \to \infty\) limit (solution is exact *up to chosen* \(l_{\max}\))
- Magnetic **field gradients are not computed in this module**
  - The solver outputs self‑consistent surface fields and currents
  - Gradient tensors must be computed downstream

---

## 6. Bottom Line

- The solver is **mathematically exact within its stated MQS thin‑shell spherical‑harmonic model**
- All coupling between conductivity, Faraday induction, and emitted fields is treated consistently
- Any inaccuracy arises only from:
  - truncation in \(l\)
  - physical assumptions of the model itself

If gradient computation is performed later using the same SH normalization and conventions, the *entire pipeline* (normal field → currents → emitted field → gradients) can remain exact within this framework.

---

*This file is intended as a technical review summary, suitable for inclusion in documentation or code repositories.*

