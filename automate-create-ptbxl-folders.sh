#!/bin/bash

echo "Creating 22 folders of ptb-xl..."

# Loop to generate numbers from 00000 to 21000 in steps of 1000
for (( num=0; num<=21000; num+=1000 )); do
    input=$(printf "%05d" "$num")
    python prepare_ptbxl_data.py -i "ptb-xl/records100/$input" -d "ptb-xl/ptbxl_database.csv" -s "ptb-xl/scp_statements.csv" -o "$input"
done

