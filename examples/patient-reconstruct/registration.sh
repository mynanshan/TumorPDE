#!/bin/bash
#SBATCH --job-name=ants_parallel     # Job name
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1                   # Number of tasks (1 per patient)
#SBATCH --cpus-per-task=16            # Number of CPU cores per task (adjust as needed)
#SBATCH --mem-per-cpu=4G                     # Memory per task (adjust as needed)
#SBATCH --array=0-6                  # Job array, one job for each patient
#SBATCH --time=01:00:00              # Time limit

# Array of patient IDs
patients=("HR" "LY" "STT" "XXH" "YXB" "WXS" "WZL")

# Select the patient based on the array index
patient=${patients[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the selected patient
python3 registration.py $patient $1

echo "Finished!"
