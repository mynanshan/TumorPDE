#!/project/6006512/muye/env/torch/bin/python
# SBATCH --job-name=ants_registration     # Job name
# SBATCH --account=def-jiguocao
# SBATCH --ntasks=1                   # Number of tasks (1 per patient)
# SBATCH --cpus-per-task=16            # Number of CPU cores per task (adjust as needed)
# SBATCH --mem-per-cpu=2G                     # Memory per task (adjust as needed)
# SBATCH --time=01:00:00              # Time limit

import ants
import numpy as np
import os
import sys
sys.path.append(os.getcwd())


atlas_path = "../data/Atlas/"
atlas_t1 = ants.image_read(os.path.join(
    atlas_path, "atlas_t1.nii"), reorient=True)
atlas_mask = ants.image_read(os.path.join(
    atlas_path, "atlas_mask.nii"), reorient=True)
atlas_gm = ants.image_read(os.path.join(
    atlas_path, "atlas_gm.nii"), reorient=True)
atlas_wm = ants.image_read(os.path.join(
    atlas_path, "atlas_wm.nii"), reorient=True)
atlas_csf = ants.image_read(os.path.join(
    atlas_path, "atlas_csf.nii"), reorient=True)

