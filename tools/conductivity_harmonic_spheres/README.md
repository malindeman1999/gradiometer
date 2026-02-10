# Conductivity Harmonic Sphere PDFs

Renders real-valued conductivity harmonic maps on spheres and saves PDFs.

- For each selected degree `l`, it plots all `m=0..l`.
- Each mode is formed by pairing `+m` with `-m` so the reconstructed field is real-valued.
- Uses GUI-default gridding path (`build_roundtrip_grid`).

## Run (default requested set)

```powershell
python tools\conductivity_harmonic_spheres\plot_lm_harmonic_spheres.py
```

This generates PDFs for:

- `l = 1, 2, 4, 8, 16, 32`

in:

- `tools\conductivity_harmonic_spheres\output`

## Quick test (up to l=2)

```powershell
python tools\conductivity_harmonic_spheres\plot_lm_harmonic_spheres.py --degrees 1 2
```

## Notes

- Color scale is symmetric around zero and shown per sphere panel (`delta sigma (S)`).
- Default per-mode amplitude is normalized to RMS `1.0 S` (`--rms-s`).

