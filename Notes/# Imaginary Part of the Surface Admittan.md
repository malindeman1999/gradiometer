# Imaginary Part of the Surface Admittance of a Thin Conducting Sphere

## Setup and Definitions

Consider a **thin conducting spherical shell** of radius \(a\), thickness \(t \ll a\), and bulk conductivity \(\sigma\).
Define the **surface conductivity**
\[
\sigma_s = \sigma t
\]

The electromagnetic response of the shell is conveniently described using a **surface impedance**
\[
Z_s = R_s + i X_s
\]
and its inverse, the **surface admittance**
\[
Y_s = \frac{1}{Z_s}
\]

---

## Surface Impedance

For a thin, good conductor at angular frequency \(\omega\):

- **Resistive part**
  \[
  R_s = \frac{1}{\sigma_s}
  \]

- **Reactive part (inductive)**
  \[
  X_s \approx \omega \frac{\mu_0 a}{2}
  \]

Thus,
\[
Z_s \approx \frac{1}{\sigma_s} + i\,\omega \frac{\mu_0 a}{2}
\]

---

## Surface Admittance

The surface admittance is
\[
Y_s = \frac{1}{R_s + iX_s}
\]

Its **imaginary part** is
\[
\operatorname{Im}(Y_s)
=
-\frac{X_s}{R_s^2 + X_s^2}
=
-\frac{\omega (\mu_0 a/2)}{(1/\sigma_s)^2 + \omega^2 (\mu_0 a/2)^2}
\]

---

## Frequency Limits

### Low-frequency (resistive-dominated)
\[
\operatorname{Im}(Y_s) \approx
-\,\omega \frac{\mu_0 a}{2}\,\sigma_s^2
\]

### High-frequency (inductive-dominated)
\[
\operatorname{Im}(Y_s) \approx
-\,\frac{2}{\omega \mu_0 a}
\]

---

## Physical Interpretation

- The imaginary part of the surface admittance is **negative**, indicating an **inductive response**.
- Surface currents on a sphere must circulate around closed loops (great circles).
- These currents store **magnetic energy**, producing an inductive reactance.
- This behavior is geometric and persists even for an otherwise purely Ohmic conductor.

---

## Key Result

\[
\boxed{
\operatorname{Im}(Y_s) < 0 \quad \text{(inductive)}
}
\]

The inductive term scales with:
- radius \(a\),
- frequency \(\omega\),
- and the magnetic permeability \(\mu_0\).
