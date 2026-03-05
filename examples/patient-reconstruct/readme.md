# Patient Reconstruction (3D)

This folder contains the end-to-end workflow for patient-specific reconstruction using the implemented infiltration model.

## Scope and status

- Implemented and used here: growth-only infiltration PDE (`TumorInfiltraFD`)
- Not included in this workflow: deformation model (`TumorDeformFD`, still incomplete)

## Data layout

Patient dataset root:
- `data/PatienTumorMultiScan2024/`

Per-patient layout:

```text
./examples/data/PatienTumorMultiScan2024/<name>
|-- <name>1_brain/<name>1_brain.nii
|-- <name>1_tumor/<name>1_tumor.nii
|-- <name>2_brain/<name>2_brain.nii
`-- <name>2_tumor/<name>2_tumor.nii
```

- `<name>` is the patient identifier.
- `patient_list.txt` stores metadata including scan dates and batch IDs.

Brain atlas:
- `data/Atlas/` (MNI template for registration)

## Pipeline

1. Brain registration and tissue preprocessing
2. Build diffusivity map from tissue probabilities
3. Solve infiltration PDE by finite differences
4. Calibrate parameters by sensitivity-equation gradients + L-BFGS-B
5. Render plots and GIFs

## Key scripts

- `registration.py`: register one patient to atlas space
- `run_registration.sh`: batch registration on Slurm
- `patient-fd-inverse.py`: single/multi-scan calibration entry point
- `patient-fd-forward.py`: forward replay with saved calibrated parameters
- `run_model_multiscan.sh`: Slurm launcher for multi-scan calibration
- `gif-maker.py`: convert generated frames into GIF animations
- `run_gifmaker.sh`: batch GIF generation on Slurm
- `expriment_helpers.py`: I/O, visualization, and helper utilities

## Common usage

Calibrate one patient (multi-scan):

```bash
python patient-fd-inverse.py -p STT -i 1,2 -t 0 -s 0 -m 1 -f 0 -r 1
```

Batch run on Slurm:

```bash
bash run_model_multiscan.sh
```

Generate GIFs:

```bash
bash run_gifmaker.sh multiscan
```

## Outputs

- `results/plots_*`: overlay plots of observed vs simulated tumor
- `results/gifs_*`: animation of reconstructed progression
- `results/simulation*`: saved NIfTI simulation states
- `results/output_files`: logs and intermediate files
- `results/parameters.txt`: calibrated parameters

## Example result (multi-scan infiltration)

`results/gifs_multiscan/STT.gif`

![STT multi-scan infiltration result](results/gifs_multiscan/STT.gif)
