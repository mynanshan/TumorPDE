#!/bin/bash

output="scan_counts.txt"
source settings.sh

# Clear the output file
> "$output"

# Process each directory in the current directory
for dir in ${DATA_DIR}/*/; do
    dir_name="${dir%/}"  # Remove trailing slash
    max_num=0
    max_name=""

    # Process each file within the directory
    for file in "$dir"*; do
        # Skip directories
        [ -d "$file" ] && continue

        filename=$(basename "$file")
        name_part="${filename%.*}"  # Remove extension
        prefix="${name_part%%_*}"   # Get part before first underscore

        # Extract the name (letters) and number from the prefix
        name_part_before_number="${prefix%%[0-9]*}"
        number=$(echo "$prefix" | grep -oE '[0-9]+$')

        # Convert to integer and update max_num/name if larger
        if [ -n "$number" ]; then
            num=$((number))
            if [ $num -gt $max_num ]; then
                max_num=$num
                max_name="$name_part_before_number"
            fi
        fi
    done

    # Append the result to the output file
    echo "$dir_name,$max_name,$max_num" >> "$output"
done
