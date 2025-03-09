#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=fd-forward
#SBATCH --account=def-jiguocao
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output="output_files/patient-forward-%j.out"


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
import pandas as pd

from expriment_helpers import read_patient_data, weighted_center
from expriment_helpers import visualize_model_fit, visualize_model_fit_multiscan
from expriment_helpers import append_parameters_to_file


## ============================================================
##                         Load data
## ------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='Process patient data.')
parser.add_argument('-p', '--patient', required=True, help='Patient identifier')
parser.add_argument('-i', '--index', type=str, required=True, help="Indexes of scans.")
parser.add_argument('-s', '--single_scan', type=int, choices=[0, 1], default=1,
                    required=False, help='Whether to run the single-scan calibration')
parser.add_argument('-m', '--multi_scan', type=int, choices=[0, 1], default=1,
                    required=False, help='Whether to run the multi-scan calibration')
parser.add_argument('-f', '--fixed_init', type=int, choices=[0, 1], default=0,
                    required=False, help='Whether to run the fix-init calibration')
parser.add_argument('-r', '--ref_scan', type=int, default=1,
                    required=False, help='Which scan is used as the ref_scan')
args = parser.parse_args()

patient = args.patient

# Convert the input string to a list of integers
try:
    indexes = list(map(int, args.index.split(',')))
except ValueError:
    print("Error: Please provide a valid comma-separated list of integers.")
indexes = list(map(int, args.index.split(',')))

print(f"Modelling Patient: {patient}. Scan indexes {indexes}.")
print(datetime.today())

data = read_patient_data(patient, test=False, mask_ids=indexes, ref=args.ref_scan)

data_path = data["dir_path"]
# brain = data["brain"]
brain_raw = data["t1"]
gm = data["gm"]
wm = data["wm"]
csf = data["csf"]
tumor_list = data["tumor"]
aff_info = data["aff_info"]
header = data["header"]
del data

if (len(tumor_list) <= 1):
        raise ValueError("At least two scans are required.")

res_path = "./results"
paramfile_path = os.path.join(res_path, "parameters.txt")

print(f"Tumor array shape: {tumor_list[0].shape}")
print(f"Brain array shape: {brain_raw.shape}")

## ============================================================
##                   Define a Brain Domain
## ------------------------------------------------------------


# formulate the diffusivity field
print(f"Max check > gm: {gm.max()}, wm: {wm.max()}, csf: {csf.max()}")
print(f"Min check > gm: {gm.min()}, wm: {wm.min()}, csf: {csf.min()}")
vox = (0.05 * gm + 0.94 * wm + 0.01 * csf).copy()
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

# read parameters
params_records = pd.read_csv("results/parameters.txt", sep="\t")
first_record = params_records[params_records['Patient'] == patient].iloc[0]

# set up parameters
rho = first_record['rho']
D = first_record['D']

# method = "Nelder-Mead"
method = "L-BFGS-B"
basic_plot_args = {
            "file_prefix": [patient, patient+"_vfield"],
            "underlays": [brain_raw, vox],
            "tumors": tumor_list,
            "show": False, "main_title": patient
            }
save_args = {'affine': aff_info, 'header': header, 'patient': patient}


if args.single_scan == 1:

    print("Calibrating model for single scan")

    cx = (first_record['x0[0]'], first_record['x0[1]'], first_record['x0[2]'])
    print(f"Initial location: {cx}")
    init_peak_height = 0.01
    init_peak_width = 2.
    init_density_params  = {"w": init_peak_width, "h": init_peak_height, "rmax": 4.}
    max_iter = 200

    fd_pde = TumorInfiltraFD(
        geom, D, rho, init_learnable_params=cx,
        init_other_params=init_density_params, device=device)

    print("Start plotting")

    plot_dir = os.path.join(res_path, "plots", patient)
    os.makedirs(plot_dir, exist_ok=True)

    plot_args = basic_plot_args.copy()
    plot_args['save_dir'] = plot_dir

    save_dir = os.path.join(res_path, "simulation", patient)
    os.makedirs(save_dir, exist_ok=True)

    u, _, _ = fd_pde.solve(
        dt=0.001, t1=1.2 * t1, D=D, rho=rho, init_params=cx,
        plot_func=visualize_model_fit, plot_period=50,
        plot_args = plot_args,
        save_all=False, save_dir=save_dir,
        save_period=500, save_args=save_args)

if args.multi_scan == 1:

    print("Calibrating model for multi-scan")

    cx = (first_record['x0[0]'], first_record['x0[1]'], first_record['x0[2]'])
    t1 = first_record['t1']
    print(f"Initial location: {cx}")
    init_peak_height = 0.01
    init_peak_width = 2.
    init_density_params  = {"w": init_peak_width, "h": init_peak_height, "rmax": 4.}
    max_iter = 200

    fd_pde = TumorInfiltraFD(
        geom, D, rho, init_learnable_params=cx,
        init_other_params=init_density_params, device=device)

    # Read the data into a pandas DataFrame
    df = pd.read_csv(os.path.join(data_path, "patient_list.txt"))

    # Function to extract and convert dates
    def extract_and_convert_dates(name, indexes):
        # Find the row corresponding to the name
        row = df[df['name'] == name]

        if row.empty:
            return None

        # Extract the relevant dates
        dates = []
        for i in indexes:
            col_name = f'date{i}'
            date_str = row[col_name].values[0]
            if pd.notna(date_str):
                date_obj = datetime.strptime(str(date_str), '%Y%m%d')
                dates.append(date_obj)

        return dates

    # Get the scan dates for the current patient
    patient_scan_dates = extract_and_convert_dates(patient, indexes)

    if patient_scan_dates:
        # Convert the scan dates to datetime objects
        scan_date1 = datetime.strptime(patient_scan_dates[0], "%Y%m%d").date()
        scan_date2 = datetime.strptime(patient_scan_dates[-1], "%Y%m%d").date()

        # Calculate the difference between the two dates
        date_difference = float((scan_date2 - scan_date1).days)

        print(f"Scan dates for patient {patient}: {patient_scan_dates}")
        print(f"Difference between the two scan dates: {date_difference} days")
    else:
        raise ValueError(f"No scan dates found for patient {patient}")

    print("Start plotting")

    plot_dir = os.path.join(res_path, "plots_multiscan", patient)
    os.makedirs(plot_dir, exist_ok=True)

    plot_args = basic_plot_args.copy()
    plot_args['save_dir'] = plot_dir
    plot_args["t_scan"] =  [t1, 1.]
    plot_args["real_t_diff"] = date_difference
    plot_args["time_unit"] = "day"

    save_dir = os.path.join(res_path, "simulation_multiscan", patient)
    os.makedirs(save_dir, exist_ok=True)

    u, _, _ = fd_pde.solve(
        dt=0.001, t1=1.2*t1, D=D, rho=rho, init_params=cx,
        plot_func=visualize_model_fit_multiscan, plot_period=50,
        plot_args = plot_args,
        save_all=False, save_dir=save_dir,
        save_period=500, save_args=save_args)


if args.fixed_init == 1:

    def init_density_func(x, rmax = 0.1):
        return rmax * torch.as_tensor(tumor_list[0], device=device)

    init_density_params = {"rmax": 0.1}

    fd_pde = TumorInfiltraFD(
        geom, D, rho,
        init_density_func=init_density_func,
        init_other_params=init_density_params, device=device)

    print("Start plotting")

    plot_dir = os.path.join(res_path, "plots_fixinit", patient)
    os.makedirs(plot_dir, exist_ok=True)

    plot_args = basic_plot_args.copy()
    plot_args['save_dir'] = plot_dir

    save_dir = os.path.join(res_path, "simulation_fixinit", patient)
    os.makedirs(save_dir, exist_ok=True)

    u, _, _ = fd_pde.solve(
        dt=0.001, t1=1.2 * t1, D=D, rho=rho,
        plot_func=visualize_model_fit, plot_period=50,
        plot_args = plot_args,
        save_all=False, save_dir=save_dir,
        save_period=500, save_args=save_args)
