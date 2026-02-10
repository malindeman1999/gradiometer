# Europa Conductivity Baseline and Component Deviation Estimates

This summary is for the implemented `europa_snapshot` model in `workflow/conductivity_models/europa_snapshot.py`, using the current GUI default baseline:

- Baseline sheet conductivity: `sigma0 = 6.0e4 S`
- Source of baseline: `2 * seawater_conductivity_s_per_m * ocean_thickness_m = 2 * 0.3 * 100000`

All size estimates below are deviations from `sigma0` in Siemens (`S`) for the default snapshot parameters (`seed=7`, `lmax=36`).

| Component | Physical meaning | Model term | Default strength in code | Size estimate vs baseline (S) | What deviation should look like in data |
|---|---|---|---|---|---|
| Baseline (`sigma0`) | Global induced-ocean conductivity level | Constant term | `sigma0 = 6.0e4 S` | No spatial deviation (`0 S`) | Sets global induction amplitude/phase background. |
| Composition (`x_chem`) | Broad chemistry contrasts (chloride/sulfate style large-scale variation) | Added in log-conductivity, low-order smooth pattern | `chem_contrast = 0.35` | RMS `~2.0e4 S` (`~33.3%` of baseline); typical high-end deviation `~3.7e4 S`; absolute conductivity extrema `~2.81e4` to `~1.08e5 S` | Low-degree/hemispheric structure, smooth gradients over large regions, mostly low-`l` spectral power. |
| Exchange (`x_exchange`) | Localized chaos/plume exchange anomalies | Sum of spherical Gaussians (then standardized) | `n_exchange_sites=4`, `exchange_amp=0.45`, `exchange_width_deg=18` | RMS `~1.18e5 S` (`~196.7%` of baseline); 95th deviation scale `~1.47e5 S`; absolute conductivity extrema `~4.50e2` to `~1.36e6 S` | Patchy localized hotspots/coldspots, high spatial contrast, strongest local gradients, elevated mid/high-`l` content near anomaly scales. |
| Flow (`x_flow`) | Large-scale anisotropy from transport/circulation proxy | Equatorially weighted elongated pattern in log-conductivity | `flow_anisotropy = 0.20` | RMS `~1.21e4 S` (`~20.2%` of baseline); typical high-end deviation `~2.5e4 S`; absolute conductivity extrema `~3.49e4` to `~9.68e4 S` | Zonal/elongated bands (especially low-latitude), directionally biased structure rather than isolated spots. |
| Background (`x_bg`) | Residual small-amplitude stochastic structure | Band-limited smooth residual in log-conductivity | `background_amp = 0.08` | RMS `~4.8e3 S` (`~8.0%` of baseline); typical high-end deviation `~8.1e3 S`; absolute conductivity extrema `~5.17e4` to `~6.92e4 S` | Fine, low-amplitude mottling; weak high-`l` tail without dominant coherent features. |

## Notes on interpretation

- The component scales above are computed from the implemented snapshot decomposition fields (`x_chem`, `x_exchange`, `x_flow`, `x_bg`) converted to conductivity deviations.
- RMS values are reported both in `S` and as percent of baseline (`RMS / sigma0`).
- Extrema in the table are absolute conductivity values (`sigma`, in `S`), not `sigma - sigma0`.
- In this implementation, `x_exchange` is standardized after summing patches, so it tends to dominate spatial variance regardless of `exchange_amp` value (as long as `n_exchange_sites > 0`).
- The final conductivity map uses `sigma = sigma0 * exp(x_total)` with mean renormalization to `sigma0`, so local positive anomalies can grow faster than linear estimates due to the exponential mapping.

## Reference Ocean-Conductivity Scenarios

Using `ocean_thickness = 100 km` and the workflow convention
`sigma0 ~= 2 * sigma_bulk * thickness`:

| Scenario | Approx bulk conductivity `sigma_bulk` (S/m) | Implied sheet conductivity `sigma0` (S) |
|---|---:|---:|
| Earth-like seawater composition | `~3 to 5 S/m` | `~6.0e5 to 1.0e6 S` |
| Saturated Europan salt brine (composition-dependent) | `~10 to 30 S/m` | `~2.0e6 to 6.0e6 S` |

These are order-of-magnitude reference values. Actual Europan ocean conductivity depends strongly on temperature, pressure, ionic composition (chloride/sulfate/carbonate mix), and phase state.
