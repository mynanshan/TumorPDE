#!/bin/bash

# TODO: !!! Align all patients's images to the template to unify the scales

source settings.sh

# Define the metainfo file
metainfo="${DATA_DIR}/patient_list.txt"

# Determine which batch to be processed
if [ $# -eq 0 ]; then    # No argument provided
    target_batch=-1
elif [ $# -eq 1 ]; then  # One argument provided, set x to the argument
    target_batch=$1
else
    echo "Only 1 argument can be accepted. Usage: $0 <Target Batch ID>."
    exit 1
fi

while IFS=',' read -r name date1 date2 date3 datatype batchid numscan; do
    # skip the header
    if [ "$name" == "name" ]; then
        continue
    fi

    # if target_id is provided, skip non-target patients
    if [ "$target_batch" -ne -1 ] && [ "$batchid" -ne "$target_batch" ]; then
        continue  # Skip to the next iteration
    fi

    # Submit the job using sbatch with patient and scan_id as arguments
    echo "Run model fitting for patient:" "$name"
    # sbatch patient-fd-inverse.py -p "$name" -i 2,3 -t 1 -s 0 -m 0 -f 1 -r 3
    sbatch --gpus-per-node=1 patient-fd-inverse.py -p "$name" -i 1,2 -t 0 -s 0 -m 1 -f 0 -r 1

done < "$metainfo"
