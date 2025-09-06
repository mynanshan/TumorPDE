#!/bin/bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 <batchid-int> <source-str> [csv=patient_list.txt] [script=examples/patient-reconstruct/make_tumorvis.py]"
  exit 1
fi

batchid="${1:-3}"
source="${2:-simulation}"
csv="${3:-examples/data/PatienTumorMultiScan2024/patient_list.txt}"
script="${4:-examples/patient-reconstruct/make_tumorvis.py}"

echo "batchid: $batchid"
echo "source: $source"
echo "csv: $csv"
echo "script: $script"

# name,date1,date2,date3,batchid,batchid,numscan
# Field 1 is name, field 6 is batchid (comma-separated)
awk -F, -v dt="$batchid" 'NR>1 && $6==dt {print $1}' "$csv" \
| while IFS= read -r name; do
    echo "Running for: $name"
    python "$script" --patient "$name" --type "$source"
  done
