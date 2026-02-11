# Gradiometer Stencil Order vs Sensor Noise Requirement

This note explains why a higher-order gradiometer stencil can require quieter sensors for the same target gradient noise.

## Setup

Assume:

- sensor noises are uncorrelated,
- each sensor has the same RMS noise `sigma_s`,
- gradiometer output is a weighted sum with coefficients `c_i`,
- gradient mode divides by baseline length `L`.

For gradient output, noise variance is:

`sigma_g^2 = (sigma_s^2 / L^2) * sum_i(c_i^2)`

So for a required gradient RMS `sigma_g,target`, required per-sensor RMS is:

`sigma_s,req = sigma_g,target * L / sqrt(sum_i(c_i^2))`

## Key Consequence

For fixed `L` and fixed target `sigma_g,target`, the required sensor RMS is inversely proportional to `sqrt(sum_i(c_i^2))`.

- Larger `sum_i(c_i^2)` means stronger noise amplification.
- Stronger amplification means each sensor must be quieter.

Higher-order derivative stencils often use larger alternating weights, so `sum_i(c_i^2)` can increase substantially with stencil order.

## Example (Current Code Conventions)

From `solar_wind_noise/solar_wind_functions.py`:

- 2-point difference: `sum(c_i^2) = 2`
- 5-point central stencil: `sum(c_i^2) = 14.444...`

Required sensor-noise ratio (same `L`, same target gradient RMS):

`sigma_s,5 / sigma_s,2 = sqrt(2 / 14.444...) = 0.372`

Interpretation:

- 5-point sensors must be about `1/0.372 = 2.68x` quieter than 2-point sensors.

## Why this can feel counterintuitive

If you expect `sqrt(N)` improvement, that expectation matches averaging-like estimators with similar weights.  
The high-order derivative stencil here is not an equal-weight average, so that scaling does not apply.

## Practical guidance

- If the requirement is minimum sensor noise, a low-order stencil may be preferable.
- If the requirement is derivative accuracy / truncation error, higher-order stencils can help but may tighten sensor-noise requirements.
- Choose stencil order by balancing numerical accuracy against sensor-noise amplification.

## Related tool

Use `solar_wind_noise/gradiometer_noise_calculator.xlsx` to vary:

- baseline length `L`,
- number of points `N`,
- output type (`gradient` or `difference`),
- target RMS,
- band limits (`f_low`, `f_high`),

and see required sensor RMS and ASD targets.
