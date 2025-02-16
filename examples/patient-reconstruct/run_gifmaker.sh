
#!/bin/bash

source settings.sh

# Define the metainfo file
metainfo="${DATA_DIR}/patient_list.txt"

if [ $# -eq 0 ]; then    # No argument provided
    echo "Experimen Identifier is required as an argument."
    exit 1
fi

exprname=$1

# Determine which batch to be processed
if [ $# -eq 1 ]; then    # No argument provided
    target_batch=-1
elif [ $# -eq 2 ]; then  # One argument provided, set x to the argument
    target_batch=$2
else
    echo "Only 2 argument can be accepted. Usage: $0 <Experiment Identifier> <Target Batch ID>."
    exit 1
fi

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

    echo "Run gifmaker for patient:" "$name" "Experiment:" "$exprname"

    input="${RESULT_DIR}/plots_${exprname}/${name}"
    output="${RESULT_DIR}/gifs_${exprname}"
    mkdir -p "$output"

    sbatch gif-maker.py "-i" "$input" "-o" "$output" "-p" "$name" 
    # python gif-maker.py "-i" "$input" "-o" "$output" "-p" "$name" 
done < "$metainfo"
