# TumorPDE

TumorPDE is a research codebase for PDE-based modeling of brain tumor progression from medical imaging.

Current status:
- Infiltration (growth-only) model is implemented and tested.
- Deformation model is under development and not yet validated.

## Implemented model: tumor infiltration PDE

The main implemented model is a reaction-diffusion equation solved on a patient-specific 3D brain domain:

`du/dt = D * div(v(x) grad u) + alpha * u * (1 - u)`

- `u(x,t)`: tumor cell density
- `D`: global diffusion scale (learned)
- `alpha`: proliferation rate (learned)
- `v(x)`: fixed tissue diffusivity map built from GM/WM/CSF segmentations

The initial condition is a localized Gaussian with learnable center `x0`.

## Calibration algorithm (sensitivity equations)

Parameter estimation uses gradient-based optimization with explicit sensitivity equations:
- sensitivities: `phi = du/dD`, `psi = du/dalpha`, `eta = du/dx0`
- loss: voxelwise mismatch (default MSE) between simulated and observed tumor masks
- optimizer: `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)`

This supports:
- single-scan calibration (`D`, `alpha`, `x0`)
- multi-scan calibration with inferred scan-alignment times

## Repository structure

- `tumorpde/`
- `tumorpde/models/growth.py`: infiltration PDE forward solver + calibration
- `tumorpde/models/deform.py`: deformation-coupled model (incomplete)
- `tumorpde/models/_base.py`: shared model base classes and optimization utilities
- `tumorpde/volume_domain.py`: patient-specific computational domain
- `tumorpde/calc/`: losses, linear algebra utilities, geometry helpers
- `examples/fd-growth-1d.ipynb`: 1D infiltration demo
- `examples/fd-deform-1d.ipynb`: 1D deformation prototype demo
- `examples/patient-reconstruct/`: 3D patient reconstruction workflow
- `test/`: derivative/sensitivity/loss utility tests used during development

## Install

From the repository root:

```bash
pip install -e .
```

## Presentation example (multi-scan infiltration result)

`examples/patient-reconstruct/results/gifs_multiscan/STT.gif`

![STT multi-scan infiltration result](examples/patient-reconstruct/results/gifs_multiscan/STT.gif)

## See also

For the full patient reconstruction pipeline, scripts, and outputs, see:
- `examples/patient-reconstruct/readme.md`
