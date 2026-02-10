# Conductivity to Magnetic-Gradient Map (Tensor Form)

This note summarizes the forward model used in the workflow, written with tensor operators and Einstein summation.

## 1. Objects and Indices

- Surface harmonic index: $a,b,c,\ldots$ (flattened $(l,m)$ index space)
- Admittance harmonic index: $\alpha,\beta,\ldots$ (also flattened $(l,m)$)
- Surface node index: $n,m,\ldots$
- Cartesian component index: $i,j,k \in \{1,2,3\}$
- Gradient-direction index: $p,q,r \in \{1,2,3\}$

Key fields/operators:

- Conductivity at nodes: $\sigma_n$ (or harmonic coefficients $\sigma_a$)
- Admittance at nodes: $A_n$
- Admittance in harmonic space: $A_\alpha$
- Mixing operator (from admittance): $M_{ab}(A)$
- Self-field operator: $S_{ab}$
- Ambient radial forcing coefficients: $b_{\mathrm{ext},a}$
- Total radial field coefficients: $b_{\mathrm{tot},a}$
- Toroidal electric-field coefficients: $e_b$
- Toroidal current coefficients: $K_a$
- Emitted-field gradient tensor at observation point $x$:
  $G_{ip}(x) = \partial B_i/\partial x_p$

## 2. Conductivity to Admittance

In this workflow configuration, Step 1 uses no extra inductive term in admittance, so:

$$
A_n = \sigma_n.
$$

So admittance is real and equals conductivity; inductive behavior is handled by the self-consistent solve.

Admittance harmonic coefficients are inner products against spherical harmonics:

$$
A_\alpha = \int_{\Omega} Y_\alpha^*(\hat{\mathbf r})\,A(\hat{\mathbf r})\,d\Omega.
$$

Discrete weighted form (sampled nodes):

$$
A_\alpha \approx \sum_n w_n\,Y_\alpha^*(\hat{\mathbf r}_n)\,A_n.
$$

In this workflow implementation, the same mapping is evaluated using a weighted pseudoinverse SH transform. Because the workflow uses $N_{\text{node}}=(l_{\max}+1)^2=N_{\text{harm}}$, this operator is square; when well-conditioned, the pseudoinverse equals the ordinary inverse. The `pinv` form is used mainly for numerical robustness.

In index form (Einstein summation), mixing-matrix assembly is

$$
M_{ab} = C_{ab\alpha}\,A_\alpha,
$$

where $C_{ab\alpha}$ is the precomputed harmonic-coupling tensor (Gaunt-based in this workflow).

Definition of $C_{ab\alpha}$:

- $C_{ab\alpha}$ maps admittance harmonics to mixing entries.
- Domain/codomain: $\mathbb{C}^{N_{\text{harm}}}\to\mathbb{C}^{N_{\text{harm}}\times N_{\text{harm}}}$.
- It is assembled from Gaunt-coefficient contractions and frequency/geometry factors.

The Gaunt coefficient used in assembly is

$$
G_{a\alpha b}
= \int_{\Omega} Y_a^*(\hat{\mathbf r})\,Y_\alpha(\hat{\mathbf r})\,Y_b(\hat{\mathbf r})\,d\Omega.
$$

In implementation (`europa_model/solvers.py`), this is evaluated exactly via Wigner-$3j$ symbols.  
For indices $a\!\leftrightarrow\!(L,M)$, $\alpha\!\leftrightarrow\!(l_0,m_0)$, $b\!\leftrightarrow\!(l,m)$:

$$
G_{L M,\;l_0 m_0,\;l m}
= (-1)^M
\sqrt{\frac{(2L+1)(2l_0+1)(2l+1)}{4\pi}}
\begin{pmatrix}L & l_0 & l\\ 0 & 0 & 0\end{pmatrix}
\begin{pmatrix}L & l_0 & l\\ -M & m_0 & m\end{pmatrix},
$$

with selection rule

$$
-M + m_0 + m = 0.
$$

Define flattened index $b\leftrightarrow(l_b,m_b)$. The Faraday diagonal is

$$
F_b \equiv F_{l_b} =
\begin{cases}
-\dfrac{i\omega R}{l_b(l_b+1)}, & l_b\ge 1,\\[6pt]
0, & l_b=0,
\end{cases}
$$

where $\omega$ is forcing angular frequency and $R$ is body radius.

Then

$$
M_{ab}
= \sum_{\alpha} A_\alpha\,F_b\,G_{a\alpha b}
\equiv C_{ab\alpha}A_\alpha,
\qquad
C_{ab\alpha}=F_b\,G_{a\alpha b}.
$$

### 2.1 Why $F_b$ Scales as $1/[l_b(l_b+1)]$ (Maxwell-Faraday)

In phasor form (workflow sign convention),

$$
\nabla\times\mathbf E = i\omega\mathbf B.
$$

For one harmonic mode on radius $R$,

$$
B_r=b_{lm}Y_{lm}(\theta,\phi),\qquad
\mathbf E_t=e_{lm}\mathbf T_{lm},
$$

with $\mathbf T_{lm}$ toroidal vector spherical harmonic. On the sphere:

$$
\left[\nabla\times\mathbf E_t\right]_r
=-\frac{l(l+1)}{R}\,e_{lm}\,Y_{lm}.
$$

Matching both sides:

$$
-\frac{l(l+1)}{R}\,e_{lm}\,Y_{lm}=i\omega\,b_{lm}\,Y_{lm},
$$

so

$$
e_{lm}=-\frac{i\omega R}{l(l+1)}\,b_{lm}.
$$

Hence

$$
F_b=\frac{e_b}{b_b}=-\frac{i\omega R}{l_b(l_b+1)},
$$

with $F_b=0$ for $l_b=0$ (monopole does not support toroidal current).

Therefore, for mode $b\leftrightarrow(l_b,m_b)$:

$$
e_b = F_b\,b_b
= \begin{cases}
-\dfrac{i\omega R}{l_b(l_b+1)}\,b_b, & l_b\ge 1,\\[6pt]
0, & l_b=0.
\end{cases}
$$

Interpretation: $e_b$ is the complex scalar coefficient (amplitude+phase) multiplying toroidal basis function $\mathbf T_b$ in
$$
\mathbf E_t(\hat{\mathbf r})=\sum_b e_b\,\mathbf T_b(\hat{\mathbf r}).
$$

### 2.2 Why Gaunt Coefficients Couple Modes (from Maxwell + Product Structure)

Start from thin-sheet closure:

$$
\mathbf J_t(\hat{\mathbf r}) = A(\hat{\mathbf r})\,\mathbf E_t(\hat{\mathbf r}).
$$

Expand admittance and tangential field:

$$
A(\hat{\mathbf r})=\sum_\alpha A_\alpha\,Y_\alpha(\hat{\mathbf r}),
\qquad
\mathbf E_t(\hat{\mathbf r})=\sum_b e_b\,\mathbf T_b(\hat{\mathbf r}).
$$

Use normalized toroidal vector spherical harmonics:

$$
\mathbf T_{lm}(\hat{\mathbf r})
= \frac{\hat{\mathbf r}\times \nabla_s Y_{lm}(\hat{\mathbf r})}{\sqrt{l(l+1)}},
\qquad
\int_\Omega \mathbf T_a^*\cdot \mathbf T_b\,d\Omega=\delta_{ab}.
$$

With

$$
Y_{lm}(\theta,\phi)=N_{lm}P_l^m(\cos\theta)e^{im\phi},
$$

the derivatives used in $\mathbf T_{lm}$ are

$$
\partial_\phi Y_{lm}=im\,Y_{lm},
$$

$$
\partial_\theta Y_{lm}
=-N_{lm}\sin\theta\,\frac{dP_l^m(x)}{dx}\Big|_{x=\cos\theta}\,e^{im\phi}
=N_{lm}\frac{l\cos\theta\,P_l^m(\cos\theta)-(l+m)P_{l-1}^m(\cos\theta)}{\sin\theta}e^{im\phi}.
$$

Now project $\mathbf J_t$ onto output mode $a$:

$$
J_a
=\int_\Omega \mathbf T_a^*(\hat{\mathbf r})\cdot \mathbf J_t(\hat{\mathbf r})\,d\Omega
=\sum_{\alpha,b}A_\alpha e_b
\int_\Omega \mathbf T_a^*(\hat{\mathbf r})\cdot
\big(Y_\alpha(\hat{\mathbf r})\mathbf T_b(\hat{\mathbf r})\big)\,d\Omega.
$$

Define the exact vector coupling tensor:

$$
V_{a\alpha b}
\equiv
\int_\Omega \mathbf T_a^*(\hat{\mathbf r})\cdot
\big(Y_\alpha(\hat{\mathbf r})\mathbf T_b(\hat{\mathbf r})\big)\,d\Omega.
$$

Then the exact modal current coefficient is

$$
J_a=\sum_{\alpha,b}A_\alpha e_b\,V_{a\alpha b}.
$$

Using $e_b=F_b b_b$:

$$
J_a=\sum_{\alpha,b}A_\alpha F_b b_b\,V_{a\alpha b}.
$$

Workflow implementation note:

$$
V_{a\alpha b}\;\approx\;G_{a\alpha b}
=\int_\Omega Y_a^*(\hat{\mathbf r})Y_\alpha(\hat{\mathbf r})Y_b(\hat{\mathbf r})\,d\Omega,
$$

with $G$ evaluated by Wigner-$3j$ coefficients (given above).  
So in code,

$$
J_a \approx \sum_{\alpha,b}A_\alpha F_b b_b\,G_{a\alpha b},
$$

which is the structure assembled into `M_ab`.

### 2.3 Why $V_{a\alpha b}\approx G_{a\alpha b}$, and What is Dropped

Exact vector coupling is

$$
V_{a\alpha b}
=\int_\Omega \mathbf T_a^*(\hat{\mathbf r})\cdot
\big(Y_\alpha(\hat{\mathbf r})\mathbf T_b(\hat{\mathbf r})\big)\,d\Omega.
$$

Substitute $\mathbf T_{lm}=(\hat{\mathbf r}\times\nabla_s Y_{lm})/\sqrt{l(l+1)}$:

$$
V_{a\alpha b}
=\frac{1}{\sqrt{l_a(l_a+1)\,l_b(l_b+1)}}
\int_\Omega Y_\alpha\,
\big(\nabla_s Y_a^*\cdot \nabla_s Y_b\big)\,d\Omega.
$$

So the exact integrand contains derivative coupling $\nabla_s Y_a^*\cdot\nabla_s Y_b$.

These surface gradients are explicitly computable. On the unit sphere:


$$
\nabla_s Y_{lm}
= \hat{\theta}\,\frac{\partial Y_{lm}}{\partial \theta}
+ \hat{\phi}\,\frac{1}{\sin\theta}\,\frac{\partial Y_{lm}}{\partial \phi}
$$

so

$$
\nabla_s Y_a^* \cdot \nabla_s Y_b
= \frac{\partial Y_a^*}{\partial \theta}\frac{\partial Y_b}{\partial \theta}
+ \frac{1}{\sin^2\theta}\frac{\partial Y_a^*}{\partial \phi}\frac{\partial Y_b}{\partial \phi}
$$

Using

$$
\frac{\partial Y_{lm}}{\partial \phi}=imY_{lm},
\qquad
\frac{\partial Y_a^*}{\partial \phi}=-i m_a Y_a^*,
\qquad
\frac{\partial Y_b}{\partial \phi}=i m_b Y_b,
$$

gives

$$
\nabla_s Y_a^* \cdot \nabla_s Y_b
= \frac{\partial Y_a^*}{\partial \theta}\frac{\partial Y_b}{\partial \theta}
+ \frac{m_a m_b}{\sin^2\theta}Y_a^*Y_b
$$

Therefore the exact vector coupling can be evaluated as

$$
V_{a\alpha b}
= \frac{1}{\sqrt{l_a(l_a+1)\,l_b(l_b+1)}}
\int_{\Omega} Y_\alpha(\hat{\mathbf r})
\left(
\frac{\partial Y_a^*}{\partial \theta}\frac{\partial Y_b}{\partial \theta}
+ \frac{m_a m_b}{\sin^2\theta}Y_a^*Y_b
\right)\,d\Omega
$$

Using

$$
Y_{lm}(\theta,\phi)=N_{lm}P_l^m(\cos\theta)e^{im\phi},
\qquad
\partial_\theta Y_{lm}
=N_{lm}\frac{l\cos\theta\,P_l^m(\cos\theta)-(l+m)P_{l-1}^m(\cos\theta)}{\sin\theta}e^{im\phi},
$$

define

$$
Q_{lm}(\theta)\equiv
\frac{l\cos\theta\,P_l^m(\cos\theta)-(l+m)P_{l-1}^m(\cos\theta)}{\sin\theta}.
$$

Then

$$
V_{a\alpha b}
= \frac{1}{\sqrt{l_a(l_a+1)\,l_b(l_b+1)}}
\int_{\Omega}
Y_\alpha(\hat{\mathbf r})\,N_aN_b\,e^{i(m_b-m_a)\phi}
\left[
Q_a(\theta)Q_b(\theta)
+ \frac{m_a m_b}{\sin^2\theta}P_{l_a}^{m_a}(\cos\theta)P_{l_b}^{m_b}(\cos\theta)
\right]\,d\Omega.
$$

In the workflow approximation, these derivative/curl-coupling weights are not carried explicitly in the mixing tensor.  
Instead, coupling is represented by the scalar triple overlap:

$$
G_{a\alpha b}=\int_\Omega Y_a^*\,Y_\alpha\,Y_b\,d\Omega,
$$

and Faraday scaling is handled separately by $F_b$.

What is dropped (relative to exact $V$):

- explicit angular-derivative product $\nabla_s Y_a^*\cdot\nabla_s Y_b$ inside the coupling integral,
- associated $l,m$-dependent prefactors from toroidal normalization and derivative operators,
- any additional geometric coupling terms that appear in a full vector-VSH product decomposition.

Interpretation:

- kept: mode-selection geometry and broad spectral coupling via Gaunt tensor $G$ plus Faraday factor $F_b$,
- dropped: higher-fidelity vector differential structure in the coupling kernel.
So the dependence chain into $A_{ab}$ (system matrix) is:

$$
\sigma_n \to A_n \to A_\alpha \to M_{ab}(A_\alpha) \to A_{ab}.
$$

## 3. Self-Consistent Spectral Solve

The workflow forms:

$$
\mathcal A_{ab}=\delta_{ab}-S_{ac}M_{cb}(A),
$$

and solves:

$$
\mathcal A_{ab}\,b_{\mathrm{tot},b}=b_{\mathrm{ext},a},
$$

so:

$$
b_{\mathrm{tot},a}=(\mathcal A^{-1})_{ab}\,b_{\mathrm{ext},b}.
$$

Currents:

$$
K_a=M_{ab}(A)\,b_{\mathrm{tot},b}.
$$

Substitute:

$$
K_a=M_{ab}(A)\,(\mathcal A^{-1})_{bc}\,b_{\mathrm{ext},c}.
$$

This is nonlinear in conductivity because $M$ depends on admittance $A$, and admittance depends on $\sigma$.

## 4. Currents to Field Gradients at 100 km

At observation points $x^{(u)}$ (100 km altitude):

$$
B_i(x^{(u)})=T_{ia}(x^{(u)})K_a,
$$

$$
G_{ip}(x^{(u)})=\frac{\partial B_i}{\partial x_p}
=H_{ipa}(x^{(u)})K_a.
$$

Compactly:

$$
G_{ip}^{(u)}
=H_{ipa}^{(u)}\,M_{ab}(A)\,(\mathcal A^{-1})_{bc}\,b_{\mathrm{ext},c}.
$$

## 5. Gradient Magnitude Used in Plots

The plotted RSS gradient magnitude is:

$$
g^{(u)}=\left(G_{ip}^{(u)}\overline{G_{ip}^{(u)}}\right)^{1/2},
$$

with sum on $i,p$.

## 6. Is There a Single Matrix from Conductivity to Gradients?

Not globally. The map

$$
\sigma \rightarrow A \rightarrow M(A) \rightarrow \mathcal A^{-1}(A) \rightarrow K \rightarrow G \rightarrow g
$$

is nonlinear.

## 7. Local Linearization (Jacobian Form)

Around reference $\sigma^0$, perturbation $\delta\sigma_r$ gives:

$$
\delta g^{(u)}\approx J_{ur}\,\delta\sigma_r,
$$

with

$$
J_{ur}=\left.\frac{\partial g^{(u)}}{\partial \sigma_r}\right|_{\sigma^0}.
$$

Before RSS reduction:

$$
\delta G_{ip}^{(u)}=L_{ipr}^{(u)}\,\delta\sigma_r,
$$

where $L$ includes chain-rule contractions through $M$, $\mathcal A^{-1}$, and $A(\sigma)$.

## 8. Practical Interpretation

- Admittance equals conductivity directly ($A=\sigma$).
- Self-consistent feedback still introduces nonlinearity through $(I-SM)^{-1}$.
- Inversion from gradient data is therefore a regularized nonlinear inverse problem; local Jacobian approximations are valid near a chosen reference model.
