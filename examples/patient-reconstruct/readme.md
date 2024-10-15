## Data

**Patient data**: Stored under `data/PatienTumorMultiScan2024`. The file structure is the same for all patients:
```
./examples/data/PatienTumorMultiScan2024/<name>
├── <name>1_brain
│   └── <name>1_brain.nii
├── <name>1_tumor
│   └── <name>1_tumor.nii
├── <name>2_brain
│   └── <name>2_brain.nii
└── <name>2_tumor
    └── <name>2_tumor.nii
```
The `<name>` is the patient's name code. The digit after `<name>` suggests the number of the brain scan. Under each patient's root folder, the raw images of the (first) two scan are visualized in the jpg files with name `<name><scan-id>.jpg`. 

**Brain atlas**:
Stored under `data/Atlas`, the MNI template brain for SPM8

## Brain Registration

Run
```
registration.py <name>  # choice 1: one specific patient
sbatch registration.sh  # choice 2: all patients on a Slurm HPC
```
The program will produce several files under each patient's folder
* `<name>_brain_normalized.nii.gz`: registered brain data array
* `<name>_brain_registered.jpg`: registered brain image
* `<name>_gm_normalized.nii.gz`: registered gray matter data array
* `<name>_wm_normalized.nii.gz`: registered white matter data array
* `<name>_csf_normalized.nii.gz`: : registered cerebrospinal fluid matter


## Reconstruction

**Current algorithm**:
* Forward simulation: finite difference solution of PDE
* Model calibration: gradients from sensitivities + LBFGS optimizer 

```
patient-fd-inverse.py <name> <scan-id>  # for one patient
sbatch patient-fd-inverse.sh <scan-id>  # for all patients on a Slurm HPC
```

**Results**:
* `plots`: reconstructed tumor image. Red shades: manually marked tumor segmentation. Blue shades: reconstructed tumor progression
* `gifs`: gif animation made from plots
* `output_files`: parameter estimation and other intermediate outputs

