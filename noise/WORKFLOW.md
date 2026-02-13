# Solar wind noise workflow

This folder contains a small toolkit to model solar-wind magnetic-field noise, shape it by a PSD, and (optionally) apply instrument effects like gradiometer differencing and low-pass filtering. The intended workflow is:

1) Pick the physical scenario
   - Decide the gradiometer baseline (e.g., 10 m or 100 km).
   - Decide the sampling rate / duration (period) you want to simulate.

2) Build the PSD envelope
   - Use the built-in solar-wind power-law envelope (three slopes).
   - Optionally add SQUID noise after gradiometer/filters are applied.

3) Apply instrument effects (optional)
   - Gradiometer differencing (frozen-in flow model) using a time delay.
   - Low-pass filtering to represent bandwidth limits.

4) Generate noise samples and verify normalization
   - Create time-domain noise shaped by the PSD envelope.
   - Compare PSD-integrated variance vs time-domain variance.

5) Plot and inspect
   - PSD sweeps and time-domain samples.
   - Space-domain view by mapping time to distance via flow speed.

Recommended scripts to run
- Quick PSD sweep + time samples (realistic ranges):
  - check_noise_real_range.py
- Shorter/faster PSD sweep:
  - check_noise2.py
- End-to-end generation and plotting (10 m baseline):
  - generate_noise_10_m_high_freq.py
- End-to-end generation and plotting (100 km baseline):
  - generate noise_100_km_high_freq.py

Core components and what they do
- noise_functions.py
  - Defines the Noise class, which:
    - builds time/frequency grids and rFFT normalization
    - generates white noise and shapes it with a PSD envelope
    - computes PSD from FFT and integrates it for variance checks
    - supports solar-wind PSD envelopes and adding SQUID PSD
    - applies low-pass filters and gradiometer differencing in frequency domain
    - plots PSD sweeps and time-domain samples

- solar_wind_power_law.py
  - Implements a three-slope power-law PSD model via combined_power_law.
  - Includes a plotting demo in the __main__ block.

- squid_noise.py
  - Models SQUID noise vs frequency (fT/vHz), enforces a floor, and converts
    it to PSD (nT^2/Hz) for addition to the solar-wind envelope.

- generate_noise_10_m_high_freq.py / generate noise_100_km_high_freq.py
  - Standalone scripts that:
    - generate white noise
    - shape it by the solar-wind PSD
    - check PSD vs time-domain variance
    - plot PSD, time-domain noise, and space-domain noise
    - parameterize gradiometer baseline and flow speed

- check_noise_real_range.py / check_noise2.py
  - Small runners around Noise that set the PSD envelope, apply filters,
    and plot PSD sweeps and samples for specific periods and sample rates.

- rfft_normalization.py
  - Verifies rFFT normalization by comparing frequency- and time-domain power.

- delay.py
  - Demonstrates delaying a signal via frequency-domain phase shift.

Notes / caveats
- Some scripts contain older or duplicated helper functions (e.g., uniform_PSD
  defined twice in the generate_* scripts). The Noise class in noise_functions.py
  is the most consistent entry point.
- check_noise.py appears to reference helpers that are not in scope; treat it
  as legacy scratch code unless updated.
