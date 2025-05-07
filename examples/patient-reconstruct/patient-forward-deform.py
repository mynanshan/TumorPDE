#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=deform-fd
#SBATCH --account=def-jiguocao
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output="output_files/deform-fd-%j.out"
import os
import sys
sys.path.append(os.getcwd())

from tumorpde.models.deform import TumorDeformFD
from tumorpde.volume_domain import VolumeDomain

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"The current device is: {device}")

from datetime import datetime
import pandas as pd

from expriment_helpers import read_patient_data, weighted_center
from expriment_helpers import _vis_brain_scan

patient = "DANG-ZHOU-CAI-RANG"
indexes = [2,3]
ref_scan = 2

data = read_patient_data(patient, test=False, mask_ids=indexes, ref=ref_scan)

data_path = data["dir_path"]
brain_raw = data["t1"]
gm = data["gm"]
wm = data["wm"]
csf = data["csf"]
tumor_list = data["tumor"]
aff_info = data["aff_info"]
header = data["header"]
del data

matters = torch.cat([
    torch.tensor(gm, device=device).unsqueeze(0),
    torch.tensor(wm, device=device).unsqueeze(0),
    torch.tensor(csf, device=device).unsqueeze(0)
], dim=0)

geom = VolumeDomain((gm + wm + csf), [1.,1.,1.])

# Define the time domain
t0, t1 = 0., 1.

alpha = 20.
D = 100.
M = 1.
kappa = 1.
cx = weighted_center(tumor_list[0]) + 0.5  # initialized the center
D_ratio = 0.3
kappa_ratios = [0.1, 0.5]

def init_density_func(x, params = None | torch.Tensor, rmax = 0.1):
    return rmax * torch.as_tensor(tumor_list[0], device=device)

init_density_params = {"rmax": 0.3}

# create the PDE model
pde = TumorDeformFD(geom, matters,
                    D, alpha, M, kappa,
                    D_ratio, kappa_ratios,
                    init_density_func=init_density_func,
                    init_other_params=init_density_params)

dt = 0.001
state, t = pde.solve(None, dt, t1 = 0.1)

tmp_brain = 0.1 * state.brain_density[0].detach().cpu().numpy() + \
    0.9 * state.brain_density[1].detach().cpu().numpy()
_vis_brain_scan(state.tumor_density.detach().cpu().numpy(),
                [tmp_brain], tumor_list, figsize=(5,5), main_title="")