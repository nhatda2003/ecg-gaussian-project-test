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
    python evaluate_model.py -d "$test_data" -o test_output"$test_data"-"$model_name" -s scores.csv

    # Check the exit status of the evaluation command
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $test_data"
        exit 1
    fi

    echo "Completed running and evaluating for $test_data"
}

# Associative array to map training data to model names
declare -A model_names=(
    
    ["50n50aset2"]="lenet5-50n50aset2"
    ["50n50aset3"]="lenet5-50n50aset3"
    #["50n50aset4"]="lenet5-50n50aset4"
    #["2500n2500atrainset1"]="lenet5-2500n2500atrainset1"
    #["50n50aset1"]="lenet5-50n50aset1"
)

# List of data folders for testing
test_folders=("1500n1500atestset1" "1500n1500atestset2" "1500n1500atestset3")

# Train the models
for train_data in "${!model_names[@]}"; do
    model_name="${model_names[$train_data]}"
    train_model "$train_data" "$model_name"
done

# Run and evaluate the models with test data
for test_data in "${test_folders[@]}"; do
    for train_data in "${!model_names[@]}"; do
        model_name="${model_names[$train_data]}"
        run_and_evaluate "$test_data" "$model_name"
    done
done

echo "All training, running, and evaluation completed successfully"
