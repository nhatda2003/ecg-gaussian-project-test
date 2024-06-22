#!/bin/bash

# Function to train the model
train_model() {
    train_data=$1
    model_name=$2

    # Train the model
    python train_model.py -d "$train_data" -vl 1500n1500avalidationset1 -m model -n "$model_name" -v

    # Check the exit status of the training command
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for $train_data"
        exit 1
    fi

    echo "Completed training for $train_data"
}

# Function to run and evaluate the model
run_and_evaluate() {
    test_data=$1
    model_name=$2

    # Run the model
    python run_model.py -d "$test_data" -m model -mn "$model_name" -o test_output"$test_data"

    # Check the exit status of the running command
    if [ $? -ne 0 ]; then
        echo "Error: Running model failed for $test_data"
        exit 1
    fi

    # Evaluate the model
    python evaluate_model.py -d "$test_data" -o test_output"$test_data"-"$model_name" -s scores-250n250a-g-5pt.csv

    # Check the exit status of the evaluation command
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $test_data"
        exit 1
    fi

    echo "Completed running and evaluating for $test_data"
}

# List of data folders for testing
test_folders=("1500n1500atestset1" "1500n1500atestset2" "1500n1500atestset3")

# Define model names
declare -A model_names=(
    # ["50n50aset1-g-9pt"]="lenet5-50n50aset1-g-9pt"
    # ["50n50aset2-g-9pt"]="lenet5-50n50aset2-g-9pt"
    # ["50n50aset3-g-9pt"]="lenet5-50n50aset3-g-9pt"
    # ["50n50aset4-g-9pt"]="lenet5-50n50aset4-g-9pt"
    # ["50n50aset5-g-9pt"]="lenet5-50n50aset5-g-9pt"
    # ["50n50aset6-g-9pt"]="lenet5-50n50aset6-g-9pt"
    # ["50n50aset7-g-9pt"]="lenet5-50n50aset7-g-9pt"
    # ["50n50aset8-g-9pt"]="lenet5-50n50aset8-g-9pt"
    # ["50n50aset9-g-9pt"]="lenet5-50n50aset9-g-9pt"
    # ["50n50aset10-g-9pt"]="lenet5-50n50aset10-g-9pt"

    # ["50n50aset1-g-7pt"]="lenet5-50n50aset1-g-7pt"
    # ["50n50aset2-g-7pt"]="lenet5-50n50aset2-g-7pt"
    # ["50n50aset3-g-7pt"]="lenet5-50n50aset3-g-7pt"
    # ["50n50aset4-g-7pt"]="lenet5-50n50aset4-g-7pt"
    # ["50n50aset5-g-7pt"]="lenet5-50n50aset5-g-7pt"
    # ["50n50aset6-g-7pt"]="lenet5-50n50aset6-g-7pt"
    # ["50n50aset7-g-7pt"]="lenet5-50n50aset7-g-7pt"
    # ["50n50aset8-g-7pt"]="lenet5-50n50aset8-g-7pt"
    # ["50n50aset9-g-7pt"]="lenet5-50n50aset9-g-7pt"
    # ["50n50aset10-g-7pt"]="lenet5-50n50aset10-g-7pt"

    # ["50n50aset1-g-5pt"]="lenet5-50n50aset1-g-5pt"
    # ["50n50aset2-g-5pt"]="lenet5-50n50aset2-g-5pt"
    # ["50n50aset3-g-5pt"]="lenet5-50n50aset3-g-5pt"
    # ["50n50aset4-g-5pt"]="lenet5-50n50aset4-g-5pt"
    # ["50n50aset5-g-5pt"]="lenet5-50n50aset5-g-5pt"
    # ["50n50aset6-g-5pt"]="lenet5-50n50aset6-g-5pt"
    # ["50n50aset7-g-5pt"]="lenet5-50n50aset7-g-5pt"
    # ["50n50aset8-g-5pt"]="lenet5-50n50aset8-g-5pt"
    # ["50n50aset9-g-5pt"]="lenet5-50n50aset9-g-5pt"
    # ["50n50aset10-g-5pt"]="lenet5-50n50aset10-g-5pt"


    # ["50n50aset1-g-2pt"]="lenet5-50n50aset1-g-2pt"
    # ["50n50aset2-g-2pt"]="lenet5-50n50aset2-g-2pt"
    # ["50n50aset3-g-2pt"]="lenet5-50n50aset3-g-2pt"
    # ["50n50aset4-g-2pt"]="lenet5-50n50aset4-g-2pt"
    # ["50n50aset5-g-2pt"]="lenet5-50n50aset5-g-2pt"
    # ["50n50aset6-g-2pt"]="lenet5-50n50aset6-g-2pt"
    # ["50n50aset7-g-2pt"]="lenet5-50n50aset7-g-2pt"
    # ["50n50aset8-g-2pt"]="lenet5-50n50aset8-g-2pt"
    # ["50n50aset9-g-2pt"]="lenet5-50n50aset9-g-2pt"
    # ["50n50aset10-g-2pt"]="lenet5-50n50aset10-g-2pt"


    # ["50n50aset1"]="lenet5-50n50aset1"
    # ["50n50aset2"]="lenet5-50n50aset2"
    # ["50n50aset3"]="lenet5-50n50aset3"
    # ["50n50aset4"]="lenet5-50n50aset4"
    # ["50n50aset5"]="lenet5-50n50aset5"
    # ["50n50aset6"]="lenet5-50n50aset6"
    # ["50n50aset7"]="lenet5-50n50aset7"
    # ["50n50aset8"]="lenet5-50n50aset8"
    # ["50n50aset9"]="lenet5-50n50aset9"
    # ["50n50aset10"]="lenet5-50n50aset10"

    # ["250n250aset1"]="lenet5-250n250aset1"
    # ["250n250aset2"]="lenet5-250n250aset2"
    # ["250n250aset3"]="lenet5-250n250aset3"
    # ["250n250aset4"]="lenet5-250n250aset4"
    # ["250n250aset5"]="lenet5-250n250aset5"
    # ["250n250aset6"]="lenet5-250n250aset6"
    # ["250n250aset7"]="lenet5-250n250aset7"
    # ["250n250aset8"]="lenet5-250n250aset8"
    # ["250n250aset9"]="lenet5-250n250aset9"
    # ["250n250aset10"]="lenet5-250n250aset10"

    # ["250n250aset1-g-3pt"]="lenet5-250n250aset1-g-3pt"
    # ["250n250aset2-g-3pt"]="lenet5-250n250aset2-g-3pt"
    # ["250n250aset3-g-3pt"]="lenet5-250n250aset3-g-3pt"
    # ["250n250aset4-g-3pt"]="lenet5-250n250aset4-g-3pt"
    # ["250n250aset5-g-3pt"]="lenet5-250n250aset5-g-3pt"
    # ["250n250aset6-g-3pt"]="lenet5-250n250aset6-g-3pt"
    # ["250n250aset7-g-3pt"]="lenet5-250n250aset7-g-3pt"
    # ["250n250aset8-g-3pt"]="lenet5-250n250aset8-g-3pt"
    # ["250n250aset9-g-3pt"]="lenet5-250n250aset9-g-3pt"
    # ["250n250aset10-g-3pt"]="lenet5-250n250aset10-g-3pt"

    # ["250n250aset1-g-5pt"]="lenet5-250n250aset1-g-5pt"
    # ["250n250aset2-g-5pt"]="lenet5-250n250aset2-g-5pt"
    # ["250n250aset3-g-5pt"]="lenet5-250n250aset3-g-5pt"
    # ["250n250aset4-g-5pt"]="lenet5-250n250aset4-g-5pt"
    # ["250n250aset5-g-5pt"]="lenet5-250n250aset5-g-5pt"
    # ["250n250aset6-g-5pt"]="lenet5-250n250aset6-g-5pt"
    # ["250n250aset7-g-5pt"]="lenet5-250n250aset7-g-5pt"
    # ["250n250aset8-g-5pt"]="lenet5-250n250aset8-g-5pt"
    # ["250n250aset9-g-5pt"]="lenet5-250n250aset9-g-5pt"
    # ["250n250aset10-g-5pt"]="lenet5-250n250aset10-g-5pt"


    ["250n250aset1-g-7pt"]="lenet5-250n250aset1-g-7pt"
    ["250n250aset2-g-7pt"]="lenet5-250n250aset2-g-7pt"
    ["250n250aset3-g-7pt"]="lenet5-250n250aset3-g-7pt"
    ["250n250aset4-g-7pt"]="lenet5-250n250aset4-g-7pt"
    ["250n250aset5-g-7pt"]="lenet5-250n250aset5-g-7pt"
    ["250n250aset6-g-7pt"]="lenet5-250n250aset6-g-7pt"
    ["250n250aset7-g-7pt"]="lenet5-250n250aset7-g-7pt"
    ["250n250aset8-g-7pt"]="lenet5-250n250aset8-g-7pt"
    ["250n250aset9-g-7pt"]="lenet5-250n250aset9-g-7pt"
    ["250n250aset10-g-7pt"]="lenet5-250n250aset10-g-7pt"
)

# Define model order
model_order=(
    # "50n50aset1-g-9pt"
    # "50n50aset2-g-9pt"
    # "50n50aset3-g-9pt"
    # "50n50aset4-g-9pt"
    # "50n50aset5-g-9pt"
    # "50n50aset6-g-9pt"
    # "50n50aset7-g-9pt"
    # "50n50aset8-g-9pt"
    # "50n50aset9-g-9pt"
    # "50n50aset10-g-9pt"

    # "50n50aset1-g-7pt"
    # "50n50aset2-g-7pt"
    # "50n50aset3-g-7pt"
    # "50n50aset4-g-7pt"
    # "50n50aset5-g-7pt"
    # "50n50aset6-g-7pt"
    # "50n50aset7-g-7pt"
    # "50n50aset8-g-7pt"
    # "50n50aset9-g-7pt"
    # "50n50aset10-g-7pt"

    # "50n50aset1-g-5pt"
    # "50n50aset2-g-5pt"
    # "50n50aset3-g-5pt"
    # "50n50aset4-g-5pt"
    # "50n50aset5-g-5pt"
    # "50n50aset6-g-5pt"
    # "50n50aset7-g-5pt"
    # "50n50aset8-g-5pt"
    # "50n50aset9-g-5pt"
    # "50n50aset10-g-5pt"

    # "250n250aset1-g-3pt"
    # "250n250aset2-g-3pt"
    # "250n250aset3-g-3pt"
    # "250n250aset4-g-3pt"
    # "250n250aset5-g-3pt"
    # "250n250aset6-g-3pt"
    # "250n250aset7-g-3pt"
    # "250n250aset8-g-3pt"
    # "250n250aset9-g-3pt"
    # "250n250aset10-g-3pt"

    # "250n250aset1-g-5pt"
    # "250n250aset2-g-5pt"
    # "250n250aset3-g-5pt"
    # "250n250aset4-g-5pt"
    # "250n250aset5-g-5pt"
    # "250n250aset6-g-5pt"
    # "250n250aset7-g-5pt"
    # "250n250aset8-g-5pt"
    # "250n250aset9-g-5pt"
    # "250n250aset10-g-5pt"

    "250n250aset1-g-7pt"
    "250n250aset2-g-7pt"
    "250n250aset3-g-7pt"
    "250n250aset4-g-7pt"
    "250n250aset5-g-7pt"
    "250n250aset6-g-7pt"
    "250n250aset7-g-7pt"
    "250n250aset8-g-7pt"
    "250n250aset9-g-7pt"
    "250n250aset10-g-7pt"
)

# Train the models
for train_data in "${model_order[@]}"; do
    model_name="${model_names[$train_data]}"
    train_model "$train_data" "$model_name"
done

# Run and evaluate the models with test data
for test_data in "${test_folders[@]}"; do
    for train_data in "${model_order[@]}"; do
        model_name="${model_names[$train_data]}"
        run_and_evaluate "$test_data" "$model_name"
    done
done

echo "All training, running, and evaluation completed successfully"
