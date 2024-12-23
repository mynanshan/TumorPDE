#!/bin/bash
#SBATCH --job-name=skull_stripping     # Job name
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1                   # Number of tasks (1 per patient)
#SBATCH --cpus-per-task=16            # Number of CPU cores per task (adjust as needed)
#SBATCH --mem-per-cpu=4G                     # Memory per task (adjust as needed)
#SBATCH --time=01:00:00              # Time limit

export PATH=$PATH:~/projects/def-jiguocao/muye/softwares/ants-2.5.4/bin/

# Ensure script is called with the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <Patient ID> <Scan ID>"
    exit 1
fi

# Input arguments
patient_id="$1"
scan_id="$2"

echo "Start skull stripping. Patient $patient_id. Scan $scan_id."

# Define paths and file prefixes
data="../data/PatienTumorMultiScan2024/$patient_id"
atlas="../data/Atlas"
prefix="${patient_id}${scan_id}"
input_file="${data}/${prefix}_brain_unstripped.nii"

# Run antsBrainExtraction.sh
antsBrainExtraction.sh -d 3 \
    -a "$input_file" \
    -e "${atlas}/atlas_t1.nii" \
    -m "${atlas}/atlas_mask.nii" \
    -o "${data}/${prefix}_"

# Decompress and rename the mask file
gunzip "${data}/${prefix}_BrainExtractionBrain.nii.gz"
mv "${data}/${prefix}_BrainExtractionBrain.nii" "${data}/${prefix}_brain.nii"

echo "Skull stripping complete. Output written to ${data}/${prefix}_brain.nii."

