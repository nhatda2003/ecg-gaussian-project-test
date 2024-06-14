#!/bin/bash

echo "Creating 22 folders of ptb-xl..."
# Define the list of inputs
inputs=($(printf "%05d " {0000..21000..1000}))  # Generates inputs from "01000" to "21000"

# Loop through each input
for input in "${inputs[@]}"
do
    # Run the command for each input
    python prepare_ptbxl_data.py -i "ptb-xl/records100/$input" -d "ptb-xl/ptbxl_database.csv" -s "ptb-xl/scp_statements.csv" -o "$input"
done

