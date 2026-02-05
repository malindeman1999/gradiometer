# Why the low-frequency RMS can match across gradiometer lengths

When the gradiometer output is reported as a **gradient** (nT/m), the low?frequency response becomes nearly **independent of baseline length**. At low frequencies, the differencing response scales like

- **difference response** ~ (2? f d / v)

but the gradient output divides by the baseline length **d**, giving

- **gradient response** ~ (2? f / v)

So the baseline length cancels out in the gradient, and the **low?frequency gradient RMS** can be similar for 10 m and 100 km baselines.

If you want the longer baseline to ?cancel more? at low frequencies, compare the **difference** output instead of the gradient. That will retain the baseline?length dependence.
