# Conductivity to Magnetic-Gradient Map (Tensor Form, V-Toroidal Coupling)

This note mirrors the standard conductivity-to-gradient derivation, but with the angular coupling kernel replaced by the toroidal-normalized V-coupling described in `V_coupling_integration_guide.md`.

## 1. Objects and Indices

- Surface harmonic index: $a,b,c,\ldots$ (flattened $(l,m)$ index space)
- Admittance harmonic index: $\alpha,\beta,\ldots$ (flattened $(l,m)$)
- Surface node index: $n,m,\ldots$
- Cartesian component index: $i,j,k \in \{1,2,3\}$
- Gradient-direction index: $p,q,r \in \{1,2,3\}$

Key fields/operators:

- Conductivity at nodes: $\sigma_n$ (or harmonic coefficients $\sigma_\alpha$)
- Admittance at nodes: $A_n$
- Admittance in harmonic space: $A_\alpha$
- Mixing operator (from admittance): $M_{ab}(A)$
- Self-field operator: $S_{ab}$
- Ambient radial forcing coefficients: $b_{\mathrm{ext},a}$
- Total radial field coefficients: $b_{\mathrm{tot},a}$
- Toroidal electric-field coefficients: $e_b$
- Toroidal current coefficients: $K_a$
- Emitted-field gradient tensor at observation point $x$:
  $G_{ip}(x)=\partial B_i/\partial x_p$

## 2. Conductivity to Admittance

In this workflow setup, admittance is formed from conductivity in step 1 and projected to SH space:

$$
A_n = \sigma_n,
$$

$$
A_\alpha = \int_\Omega Y_\alpha^*(\hat{\mathbf r})\,A(\hat{\mathbf r})\,d\Omega
\approx \sum_n w_n\,Y_\alpha^*(\hat{\mathbf r}_n)\,A_n.
$$

## 3. Faraday Map (Unchanged)

The radial-to-toroidal map remains:

$$
e_b = F_b\,b_b,
\qquad
F_b =
\begin{cases}
-\dfrac{i\omega R}{l_b(l_b+1)}, & l_b\ge 1,\\[6pt]
0, & l_b=0.
\end{cases}
$$

All $\omega R$ dependence appears here once.

## 4. V-Toroidal Angular Coupling

### 4.1 Scalar Gaunt Kernel

$$
G_{a\alpha b}
= \int_\Omega Y_a^*(\hat{\mathbf r})\,Y_\alpha(\hat{\mathbf r})\,Y_b(\hat{\mathbf r})\,d\Omega.
$$

### 4.2 Gradient Weight Identity

Define $\ell(l)=l(l+1)$ and mode degrees $a\leftrightarrow(L,M)$, $\alpha\leftrightarrow(l_0,m_0)$, $b\leftrightarrow(l_b,m_b)$.
Here:
- $L$ is the output/current harmonic degree (the degree of $K_a$, and therefore of the emitted-field mode index),
- $l_0$ is the conductivity/admittance harmonic degree (the degree of $A_\alpha$),
- $l_b$ is the input magnetic-field harmonic degree (the degree of $b_b$ and $e_b$).

$$
\int_\Omega Y_\alpha\,(\nabla_s Y_a^*\cdot \nabla_s Y_b)\,d\Omega
= \frac12\!\left[\ell(L)+\ell(l_b)-\ell(l_0)\right]G_{a\alpha b}.
$$

So define

$$
w_{a\alpha b}
\equiv \frac12\!\left[\ell(L)+\ell(l_b)-\ell(l_0)\right].
$$

### 4.3 Toroidal Normalization

With

$$
\mathbf T_{lm}=\frac{\hat{\mathbf r}\times \nabla_s Y_{lm}}{\sqrt{\ell(l)}},
$$

the toroidal-normalized V kernel is

$$
\widetilde V_{a\alpha b}
= \frac{w_{a\alpha b}}{\sqrt{\ell(L)\,\ell(l_b)}}\,G_{a\alpha b},
$$

with convention $\widetilde V_{a\alpha b}=0$ if $\ell(L)\ell(l_b)=0$.

### 4.4 Mixing Tensor and Matrix

Define

$$
C^{(V)}_{ab\alpha} \equiv F_b\,\widetilde V_{a\alpha b}.
$$

Then

$$
M^{(V)}_{ab}=C^{(V)}_{ab\alpha}\,A_\alpha
=\sum_\alpha A_\alpha\,F_b\,\widetilde V_{a\alpha b}.
$$

Equivalent expanded form:

$$
M^{(V)}_{ab}
=\sum_\alpha A_\alpha\,F_b\,
\frac{\frac12\left[\ell(L)+\ell(l_b)-\ell(l_\alpha)\right]}
{\sqrt{\ell(L)\,\ell(l_b)}}\,
G_{a\alpha b}.
$$

Current coefficients are then

$$
K_a = M^{(V)}_{ab}\,b_{\mathrm{tot},b}.
$$

## 5. Self-Consistent Spectral Solve

The solve structure is unchanged; only $M\to M^{(V)}$:

$$
\mathcal A^{(V)}_{ab}=\delta_{ab}-S_{ac}M^{(V)}_{cb},
$$

$$
\mathcal A^{(V)}_{ab}\,b_{\mathrm{tot},b}=b_{\mathrm{ext},a},
\qquad
b_{\mathrm{tot},a}=\left((\mathcal A^{(V)})^{-1}\right)_{ab}\,b_{\mathrm{ext},b}.
$$

Then

$$
K_a=M^{(V)}_{ab}\,b_{\mathrm{tot},b}.
$$

## 6. Currents to Field Gradients

At observation points $x^{(u)}$:

$$
B_i(x^{(u)})=T_{ia}(x^{(u)})K_a,
$$

$$
G_{ip}(x^{(u)})=\frac{\partial B_i}{\partial x_p}
=H_{ipa}(x^{(u)})K_a.
$$

So

$$
G_{ip}^{(u)}
=H_{ipa}^{(u)}\,M^{(V)}_{ab}\,\left((\mathcal A^{(V)})^{-1}\right)_{bc}\,b_{\mathrm{ext},c}.
$$

## 7. Gradient Magnitude for Plots

$$
g^{(u)}=\left(G_{ip}^{(u)}\,\overline{G_{ip}^{(u)}}\right)^{1/2},
$$

with sum on $i,p$.

## 8. End-to-End Computation Sequence (V-Toroidal)

$$
\sigma_n \to A_n \to A_\alpha
\to \widetilde V_{a\alpha b}\ \text{(from }G_{a\alpha b}\text{ + toroidal weight)}
\to M^{(V)}_{ab}
\to \mathcal A^{(V)}
\to b_{\mathrm{tot}}
\to K
\to B
\to G
\to g.
$$

In implementation terms, this corresponds to replacing the Gaunt-only kernel in mixing assembly by the toroidal-normalized weighted kernel, in both dense and sparse precomputed paths.

## 9. Exactness Statement

Exact within this formulation:

- The identity
  $$
  \int Y_\alpha(\nabla_s Y_a^*\!\cdot\nabla_s Y_b)\,d\Omega
  = \tfrac12[\ell_a+\ell_b-\ell_\alpha]\,G_{a\alpha b}
  $$
  is exact.
- The toroidal normalization factor $(\ell_a\ell_b)^{-1/2}$ is exact for the chosen normalized toroidal basis.

Still approximate at model level:

- Finite truncation at $l_{\max}$.
- Numerical SH transforms/quadrature and conditioning.
- Any downstream observation-space approximations not using exact analytic operators.
