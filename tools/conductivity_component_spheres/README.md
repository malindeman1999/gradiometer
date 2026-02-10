# Conductivity Component Sphere Plots

Plots five conductivity maps in one window using the same grid/model path as the nonuniform GUI:

- baseline (`sigma0`)
- composition-only
- exchange-only
- flow-only
- background-only

All panels share one conductivity color scale bar.

## Run

From repository root:

```powershell
python tools\conductivity_component_spheres\plot_conductivity_components.py
```

Save to file instead of opening a window:

```powershell
python tools\conductivity_component_spheres\plot_conductivity_components.py --output tools\conductivity_component_spheres\component_maps.png
```

## Options

- `--lmax` (default `36`)
- `--sigma0` (default GUI baseline `6.0e4 S`)
- `--seed` (default `7`)
- `--output` image path (optional)

