#!/bin/bash

# Loop through sets 8, 7, 6, ... 1
for set_number in {10..1}; do
    echo "Running commands for set $set_number"
    python generate_gaussianset.py -d 250n250a-trainset$set_number -s 250n250a-trainset$set_number-g-5pt-x10
    echo "Completed commands for set $set_number"
    echo "--------------------------------------------------"
done

