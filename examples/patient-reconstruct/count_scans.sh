#!/bin/bash

# deepseek's code: quite good ---------------------
source settings.sh

metainfo="${DATA_DIR}/patient_list.txt"
# tmpfile="tmp_info.txt"
tmpfile=$(mktemp)

compute_max_num() {
    local dir="$1"
    local max_num=0

    # Process each file in the directory if it exists
    if [ -d "$dir" ]; then
        for file in "$dir"/*; do
            # Skip directories
            [ -d "$file" ] && continue

            filename=$(basename "$file")
            name_part="${filename%.*}"  # Remove extension
            prefix="${name_part%%_*}"   # Get part before first underscore

            # Extract the number from the prefix
            number=$(echo "$prefix" | grep -oE '[0-9]+$')

            if [ -n "$number" ]; then
                num=$((number))
                if [ $num -gt $max_num ]; then
                    max_num=$num
                fi
            fi
        done
    fi

    echo "$max_num"
}

# Process the header
read -r header < "$metainfo"
IFS=, read -ra headers <<< "$header"

numscan_col=-1
for i in "${!headers[@]}"; do
    if [[ "${headers[$i]}" == "numscan" ]]; then
        numscan_col=$i
        break
    fi
done

# Add numscan column if not present
if (( numscan_col == -1 )); then
    header="$header,numscan"
    numscan_col=${#headers[@]}
fi

echo "$header" > "$tmpfile"

# Process each patient line
tail -n +2 "$metainfo" | while IFS= read -r line || [[ -n "$line" ]]; do
    IFS=, read -ra fields <<< "$line"
    name="${fields[0]}"
    dir="${DATA_DIR}/${name}"

    max_num=$(compute_max_num "$dir")

    # Ensure the fields array is large enough
    while (( ${#fields[@]} <= numscan_col )); do
        fields+=("")
    done

    fields[$numscan_col]="$max_num"

    # Join fields into a line and append to tmpfile
    updated_line=$(IFS=,; echo "${fields[*]}")
    echo "$updated_line" >> "$tmpfile"
done

# Replace the original file with the updated one
mv "$tmpfile" "$metainfo"

# my code: slower ------------------------
# #!/bin/bash

# source settings.sh
# metainfo="${DATA_DIR}/patient_list.txt"
# output="tmp_info.txt"

# > "$output"
# echo "name,date1,date2,date3,datatype,batchid,numscan" >> "$output"

# # Loop through each line in the input file
# while IFS=',' read -r name date1 date2 date3 datatype batchid; do
#     if [ "$name" == "name" ]; then
#         continue
#     fi

#     # Process each directory in the current directory
#     dir="${DATA_DIR}/${name}/"
#     max_num=0
#     max_name=""

#     # Process each file within the directory
#     for file in "$dir"*; do
#         # Skip directories
#         [ -d "$file" ] && continue

#         filename=$(basename "$file")
#         name_part="${filename%.*}"  # Remove extension
#         prefix="${name_part%%_*}"   # Get part before first underscore

#         # Extract the name (letters) and number from the prefix
#         # name_part_before_number="${prefix%%[0-9]*}"
#         number=$(echo "$prefix" | grep -oE '[0-9]+$')

#         # Convert to integer and update max_num/name if larger
#         if [ -n "$number" ]; then
#             num=$((number))
#             if [ $num -gt $max_num ]; then
#                 max_num=$num
#                 # max_name="$name_part_before_number"
#             fi
#         fi
#     done

#     # Append the result to the output file
#     echo "${name},${date1},${date2},${date3},${datatype},${batchid},${max_num}" >> "$output"
# done < "$metainfo"