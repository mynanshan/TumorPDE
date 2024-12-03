#!/bin/bash
#SBATCH --job-name=fd_parallel_2d     # Job name
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=3:00:00
#SBATCH --array=0-4                # Job array, one job for each patient
#SBATCH --output="output_files/patient-fd-2d-%A_%a.out"

# Usage: patient-fd-inverse.sh <tumor slice id>

# Array of patient IDs
patients=("HR" "LY" "STT" "XXH" "YXB" "WXS")

# Select the patient based on the array index
patient=${patients[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected patient
python3 patient-fd-inverse-2d.py -p=$patient -t=$1 -s=$2 -m=$3

echo "Finished!"
