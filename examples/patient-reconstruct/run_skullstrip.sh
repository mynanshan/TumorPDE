#!/bin/bash

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

# Loop through each line in the metainfo file
while IFS=',' read -r name date1 date2 date3 datatype batchid numscan; do
    # skip the header
    if [ "$name" == "name" ]; then
        continue
    fi

    # if target_id is provided, skip non-target patients
    if [ "$target_batch" -ne -1 ] && [ "$batchid" -ne "$target_batch" ]; then
        continue  # Skip to the next iteration
    fi
    
    # Loop through scan_id from 1 to count
    for (( scan_id=1; scan_id<=numscan; scan_id++ )); do
        # Submit the job using sbatch with patient and scan_id as arguments
        echo "Run skull-stipping for patient:" "$name" "scan_index:" "$scan_id"
        sbatch skullstrip.sh "$name" "$scan_id"
    done
done < "$metainfo"