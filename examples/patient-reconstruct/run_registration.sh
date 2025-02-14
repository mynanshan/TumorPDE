#!/bin/bash

# Define the input file
scan_count_file="scan_counts.txt"
ref_id_file="registration_ref_id.txt"

# Read ref_id.txt into an associative array
declare -A ref_ids
while IFS=',' read -r patient ref_id; do
    ref_ids["$patient"]=$ref_id
done < "$ref_id_file"

# Loop through each line in the input file
while IFS=',' read -r path patient count; do
    # Check if the patient exists in the ref_id associative array
    if [[ -n "${ref_ids[$patient]}" ]]; then
        ref_id="${ref_ids[$patient]}"
    else
        ref_id=1
    fi

    # Submit the job using sbatch with patient and scan_id as arguments
    echo "Run skull-stipping for patient:" "$patient" "num scans:" "$count" "ref id:" "$ref_id"
    sbatch registration.py "$patient" "$count" "$ref_id"
done < "$scan_count_file"