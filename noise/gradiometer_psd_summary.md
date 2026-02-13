# Gradiometer PSD, Frozen-In Flux, and Correlation Time

## Assumptions
- **Frozen-in flux / Taylor hypothesis**: magnetic structures are advected past the sensors at speed \(V\), so spatial structure maps to time via  
  \[
  x \approx V t,\qquad k = \frac{2\pi f}{V}.
  \]
- The magnetic field is treated as **statistically stationary** and approximately homogeneous over the sensor separation.
- The gradiometer consists of **two sensors separated by distance \(d\)**, measuring either a difference or a finite-difference gradient.

---

## Gradiometer as a Linear Filter
Under Taylor’s hypothesis, a two-sensor differencer is a **linear, time-invariant (LTI) filter** acting on the single-point magnetic field time series.

If the underlying magnetic field has PSD \(S_B(f)\), then the PSD of the gradiometer output is:

### Difference signal
\[
\boxed{
S_{\Delta B}(f)
=
4\,\sin^2\!\left(\frac{\pi f d}{V}\right)\, S_B(f)
}
\]

### Gradient estimate (\(\Delta B/d\))
\[
\boxed{
S_{\partial B}(f)
=
\frac{4}{d^2}\,\sin^2\!\left(\frac{\pi f d}{V}\right)\, S_B(f)
}
\]

Thus, **the net PSD is obtained by multiplying the solar-wind PSD by the gradiometer transfer function in power units**.

---

## High-Pass Behavior
For low frequencies (\(f \ll V/d\)):

\[
\sin\!\left(\frac{\pi f d}{V}\right)
\approx
\frac{\pi f d}{V}
\]

so

\[
S_{\Delta B}(f) \propto f^2\, S_B(f).
\]

This means:
- The gradiometer **suppresses low-frequency power**.
- A power-law PSD \(S_B(f)\propto f^\alpha\) is converted to  
  \[
  S_{\Delta B}(f)\propto f^{\alpha+2}.
  \]

For typical solar-wind slopes (\(\alpha\lesssim -1\)), this removes the low-frequency divergence.

---

## Correlation Time Implications
- A pure solar-wind PSD with slope \(\alpha \le -1\) **does not have a finite variance or correlation time** if extended to \(f\to 0\).
- After gradiometer differencing, the low-frequency behavior is regularized.
- The **gradiometer output can have a finite variance and a well-defined effective correlation time**, limited by:
  - the sensor separation \(d\),
  - the flow speed \(V\),
  - and the finite measurement bandwidth.

In practice, the longest timescale is set by the **lowest resolved frequency** or by the **outer scale** of the turbulence.

---

## Exact Origin of the Transfer Function
More explicitly, for two sensors:

\[
S_{\Delta B}(f)
=
S_{11}(f)+S_{22}(f)-2\operatorname{Re}\{S_{12}(f)\}.
\]

Assuming homogeneity and frozen-in advection:

\[
S_{12}(f) = S_B(f)\,e^{i 2\pi f d/V},
\]

which yields the \(\sin^2\) transfer function above.  
The “filter” is therefore fundamentally a consequence of the **cross-spectrum phase delay** between sensors.

---

## Important Caveats
- **Directional effects**: only the separation projected along the flow matters:  
  \[
  d_\parallel = \mathbf d \cdot \hat{\mathbf V}.
  \]
- **Flow variability**: if \(V\) varies significantly, the mapping is approximate; analyses are usually done over short intervals with quasi-constant \(V\).
- **Spectral nulls**: the transfer function has zeros at  
  \[
  f = n\,\frac{V}{d},\quad n\in\mathbb{Z},
  \]
  so it is not a monotonic high-pass at higher frequencies.
- **Nonstationarity**: true random walks or secular trends in the field are not fixed by differencing.
- **Finite bandwidth**: all practical “correlation times” are effective, not asymptotic.

---

## Bottom Line
Yes — **multiplying the solar-wind PSD by the gradiometer power transfer function gives the PSD seen by the gradiometer**.  
The gradiometer naturally high-pass filters the signal, often turning a formally divergent solar-wind PSD into a well-behaved spectrum with a meaningful correlation time.
