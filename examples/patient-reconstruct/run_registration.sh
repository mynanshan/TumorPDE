#!/bin/bash

source settings.sh

# Define the setting file
# ref_id_file="registration_ref_id.txt"
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

# # Read ref_id.txt into an associative array
# declare -A ref_ids
# while IFS=',' read -r patient ref_id; do
#     ref_ids["$patient"]=$ref_id
# done < "$ref_id_file"

# Loop through each line in the input file
while IFS=',' read -r name date1 date2 date3 datatype batchid numscan; do
    # skip the header
    if [ "$name" == "name" ]; then
        continue
    fi
    
    # if target_id is provided, skip non-target patients
    if [ "$target_batch" -ne -1 ] && [ "$batchid" -ne "$target_batch" ]; then
        continue  # Skip to the next iteration
    fi

    # # Check if the patient exists in the ref_id associative array
    # if [[ -n "${ref_ids[$name]}" ]]; then
    #     ref_id="${ref_ids[$name]}"
    # else
    #     ref_id=1
    # fi

    # # Submit the job using sbatch with patient and scan_id as arguments
    # echo "Run registration for patient:" "$name" "num scans:" "$numscan" "ref id:" "$ref_id"
    # sbatch registration.py "$name" "$numscan" "$ref_id"
    # Submit the job using sbatch with patient and scan_id as arguments
    echo "Run registration for patient:" "$name" "num scans:" "$numscan"
    sbatch registration.py "$name" "$numscan"
done < "$metainfo"