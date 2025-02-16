#!/bin/bash

# Function to escape special regex characters in a string
escape_regex() {
    sed -e 's/[][\.^$*+?(){\\|]/\\&/g' <<< "$1"
}

# Check if any arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 directory1 directory2 ..."
    exit 1
fi

# echo "$@"

# Process each directory provided as an argument
for dir in "$@"; do
    # Verify directory exists
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory '$dir' not found. Skipping."
        continue
    fi

    # Escape directory name for regex matching
    escaped_path=$(escape_regex "$dir")
    escaped_name=$(escape_regex $( basename "$dir"))
    # echo "Path is ${escaped_path}"
    # echo "Patient is ${escaped_name}"

    # Process all .nii files in current directory
    for file in "$dir"*.nii; do
        # echo "Found files: ${file}"

        # Skip if no files exist
        [ -e "$file" ] || continue

        # Check for main file pattern: <dir><number>.nii
        if [[ "$file" =~ ^"${escaped_path}${escaped_name}"([0-9]+)\.nii$ ]]; then
            newname="${escaped_path}${escaped_name}${BASH_REMATCH[1]}_t1_unstripped.nii"
            mv -- "$file" "$newname"
            echo "Renamed: $file -> $newname"
        
        # Check for mask file pattern: <dir><number>_mask.nii
        elif [[ "$file" =~ ^"${escaped_path}${escaped_name}"([0-9]+)_mask\.nii$ ]]; then
            newname="${escaped_path}${escaped_name}${BASH_REMATCH[1]}_t1mask.nii"
            mv -- "$file" "$newname"
            echo "Renamed: $file -> $newname"
        fi
    done
done