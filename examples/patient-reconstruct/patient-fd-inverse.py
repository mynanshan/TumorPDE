#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=fd
#SBATCH --account=def-jiguocao
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output="output_files/patient-fd-%j.out"


# import psutil
# import GPUtil

# # Get CPU information
# cpu_count = psutil.cpu_count(logical=True)
# cpu_percent = psutil.cpu_percent(interval=1)

# # Get Memory information
# memory_info = psutil.virtual_memory()

# # Get GPU information
# gpus = GPUtil.getGPUs()

# # Format GPU information
# gpu_info = ""
# for gpu in gpus:
#     gpu_info += f"""
#     GPU ID: {gpu.id}
#     Name: {gpu.name}
#     Total Memory: {gpu.memoryTotal} MB
#     Available Memory: {gpu.memoryFree} MB
#     Used Memory: {gpu.memoryUsed} MB
#     GPU Load: {gpu.load * 100:.1f}%
#     Temperature: {gpu.temperature} °C
#     """

# # Format the output
# formatted_output = f"""
# System Information:
# --------------------
# CPU:
# - Total CPUs: {cpu_count}
# - CPU Usage: {cpu_percent}%

# Memory:
# - Total Memory: {memory_info.total / (1024 ** 3):.2f} GB
# - Available Memory: {memory_info.available / (1024 ** 3):.2f} GB
# - Used Memory: {memory_info.used / (1024 ** 3):.2f} GB
# - Memory Usage: {memory_info.percent}%

# GPU:
# {gpu_info}
# """

# print(formatted_output)

import os
import sys
sys.path.append(os.getcwd())

from tumorpde.models.growth import TumorInfiltraFD
from tumorpde.volume_domain import VolumeDomain

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"The current device is: {device}")

import json
from datetime import datetime

from scan_utils import visualize_model_fit, visualize_model_fit_multiscan, weighted_center
from expriment_helpers import read_patient_data, append_parameters_to_file

## ============================================================
##                         Load data
## ------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='Process patient data.')
parser.add_argument('-p', '--patient', required=True, help='Patient identifier')
parser.add_argument('-t', '--test', type=int, choices=[0, 1], required=True, help='Test mode (0 or 1)')
parser.add_argument('-s', '--single_scan', type=int, choices=[0, 1], default=1,
                    required=False, help='Whether to run the single-scan calibration')
parser.add_argument('-m', '--multi_scan', type=int, choices=[0, 1], default=1,
                    required=False, help='Whether to run the multi-scan calibration')
args = parser.parse_args()

patient = args.patient
print(f"Modelling Patient: {patient}")

data = read_patient_data(patient, test=args.test)

dir_path = data["dir_path"]
# brain = data["brain"]
brain_raw = data["brain_raw"]
gm = data["gm"]
wm = data["wm"]
csf = data["csf"]
tumor1 = data["tumor1"]
tumor2 = data["tumor2"]
visualize_pos = data["visualize_pos"]
del data

paramfile_path = os.path.join(dir_path, "parameters.txt")

print(f"Tumor array shape: {tumor1.shape}")
print(f"Brain array shape: {brain_raw.shape}")

## ============================================================
##                   Define a Brain Domain
## ------------------------------------------------------------


# formulate the diffusivity field
print(f"Max check > gm: {gm.max()}, wm: {wm.max()}, csf: {csf.max()}")
print(f"Min check > gm: {gm.min()}, wm: {wm.min()}, csf: {csf.min()}")
vox = (0.1 * gm + 0.9 * wm + 0.02 * csf).copy()
del gm, wm, csf

# Define the spatial domain 
# assume the voxel has equal width along each axis pixel
# this is only valid when the image has been properly re-scaled
geom = VolumeDomain(vox, [1.,1.,1.])

# Define the time domain
t0, t1 = 0., 1.


## ============================================================
##                   FD Forward Simulation
## ------------------------------------------------------------


# set up parameters
rho = 20.
if args.test == 1:
    rho = 2.
D = 100.
if args.test == 1:
    D = 10.
cx = weighted_center(tumor1) + 0.5  # initialized the center
print(f"Initial location: {cx}")
init_peak_height = 0.01
init_peak_width = 2.
init_density_params  = {"w": init_peak_width, "h": init_peak_height, "rmax": 4.}
if args.test == 1:
    init_density_params['rmax'] = 1.
max_iter = 200
if args.test == 1:
    max_iter = 20

fd_pde = TumorInfiltraFD(geom, cx, D, rho,
                    init_density_params=init_density_params, device=device)

# method = "Nelder-Mead"
method = "L-BFGS-B"

fd_pde.compile(log_transform=True, rescale_params=True)

if args.single_scan == 1:

    print("Calibrating model for single scan")

    result = fd_pde.calibrate_model(
        obs=torch.as_tensor(tumor1 / tumor1.max(), device=device), dt=0.001,
        max_iter=max_iter, method=method,
        verbose=True, message_period=max_iter//10)

    print("Finish calibration")

    print(f"""
    Initial parameters:
        D = {D}, rho={rho}, x0={cx}
    Calibrated parameters:
        D = {result['D']}, rho={result['rho']}, x0={result['x0']}
    """)

    print("Start plotting")

    plot_dir = os.path.join(dir_path, "plots", patient)
    os.makedirs(plot_dir, exist_ok=True)

    u, _, _ = fd_pde.solve(
        dt=0.001, t1=2 * t1, D=result['D'], rho=result['rho'], x0=result['x0'],
        plot_func=visualize_model_fit, plot_period=50,
        plot_args = {'save_dir': plot_dir, 'file_prefix': patient,
                    "brain": brain_raw, "tumor1": tumor1, "tumor2": tumor2,
                    "slice_fracs": visualize_pos, "show": False, "main_title": patient},
        save_all=False)

    append_parameters_to_file(
        paramfile_path, patient, "single scan",
        result['D'], result['rho'], result['x0'])

if args.multi_scan == 1:

    print("Calibrating model for multi-scan")

    result = fd_pde.calibrate_model_multiscan(
        obs=torch.cat([
            torch.as_tensor(tumor1 / tumor1.max(), device=device).unsqueeze(0),
            torch.as_tensor(tumor2 / tumor2.max(), device=device).unsqueeze(0)
        ], dim=0), dt=0.001, max_iter=max_iter,
        prepare_stage=True if args.single_scan == 0 else False, max_iter_prepare=max_iter//2,
        method=method, verbose=True, message_period=max_iter//10)

    print("Finish calibration")

    print(f"""
    Initial parameters:
        D = {D}, rho={rho}, x0={cx}
    Calibrated parameters:
        D = {result['D']}, rho={result['rho']}, x0={result['x0']}
        t_scan = {result['t_scan']}
    """)
    # Load the scan dates from the JSON file
    with open(os.path.join(dir_path, "scan-dates.json"),
              'r', encoding='utf-8') as f:
        scan_dates = json.load(f)

    # Get the scan dates for the current patient
    patient_scan_dates = scan_dates.get(patient)

    if patient_scan_dates:
        # Convert the scan dates to datetime objects
        scan_date1 = datetime.strptime(patient_scan_dates[0], "%Y%m%d").date()
        scan_date2 = datetime.strptime(patient_scan_dates[1], "%Y%m%d").date()

        # Calculate the difference between the two dates
        date_difference = float((scan_date2 - scan_date1).days)

        print(f"Scan dates for patient {patient}: {patient_scan_dates}")
        print(f"Difference between the two scan dates: {date_difference} days")
    else:
        print(f"No scan dates found for patient {patient}")

    print("Start plotting")

    plot_dir = os.path.join(dir_path, "plots_multiscan", patient)
    os.makedirs(plot_dir, exist_ok=True)

    u, _, _ = fd_pde.solve(
        dt=0.001, t1=1.5*t1, D=result['D'], rho=result['rho'], x0=result['x0'],
        plot_func=visualize_model_fit_multiscan, plot_period=50,
        plot_args = {'save_dir': plot_dir, 'file_prefix': patient,
                    "brain": brain_raw, "tumor1": tumor1, "tumor2": tumor2,
                    "t_scan": result['t_scan'], "real_t_diff": date_difference,
                    "time_unit": "day",
                    "slice_fracs": visualize_pos, "show": False,
                    "main_title": patient},
        save_all=False)

    append_parameters_to_file(
        paramfile_path, patient, "multi scan",
        result['D'], result['rho'], result['x0'])



