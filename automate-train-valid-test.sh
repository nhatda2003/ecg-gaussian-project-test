#!/bin/bash


#Thu tu check de chay
# validfolder
# .csv score folder
# testfolders
# trainfolder-modelnames

# Function to train the model
train_model() {
    train_data=$1
    model_name=$2

    # Train the model
    python train_model.py -d "$train_data" -vl 1500n1500a-validset1 -m model -n "$model_name" -v

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
    python evaluate_model.py -d "$test_data" -o test_output"$test_data"-"$model_name" -s scores-lenet5kernel50-matrixprofile.csv

    # Check the exit status of the evaluation command
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $test_data"
        exit 1
    fi

    echo "Completed running and evaluating for $test_data"
}

# List of data folders for testing
test_folders=("1500n1500a-testset1" "1500n1500a-testset2" "1500n1500a-testset3")

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


    # ["50n50a-trainset1"]="lenet5-50n50a-trainset1"
    # ["50n50a-trainset2"]="lenet5-50n50a-trainset2"
    # ["50n50a-trainset3"]="lenet5-50n50a-trainset3"
    # ["50n50a-trainset4"]="lenet5-50n50a-trainset4"
    # ["50n50a-trainset5"]="lenet5-50n50a-trainset5"
    # ["50n50a-trainset6"]="lenet5-50n50a-trainset6"
    # ["50n50a-trainset7"]="lenet5-50n50a-trainset7"
    # ["50n50a-trainset8"]="lenet5-50n50a-trainset8"
    # ["50n50a-trainset9"]="lenet5-50n50a-trainset9"
    # ["50n50a-trainset10"]="lenet5-50n50a-trainset10"


    # ["250n250a-trainset1"]="lenet5-250n250a-trainset1"
    # ["250n250a-trainset2"]="lenet5-250n250a-trainset2"
    # ["250n250a-trainset3"]="lenet5-250n250a-trainset3"
    # ["250n250a-trainset4"]="lenet5-250n250a-trainset4"
    # ["250n250a-trainset5"]="lenet5-250n250a-trainset5"
    ["250n250a-trainset6"]="lenet5-250n250a-trainset6"
    # ["250n250a-trainset7"]="lenet5-250n250a-trainset7"
    # ["250n250a-trainset8"]="lenet5-250n250a-trainset8"
    # ["250n250a-trainset9"]="lenet5-250n250a-trainset9"
    # ["250n250a-trainset10"]="lenet5-250n250a-trainset10"

    # ["250n250a-trainset1-g-3pt"]="lenet5-250n250a-trainset1-g-3pt"
    # ["250n250a-trainset2-g-3pt"]="lenet5-250n250a-trainset2-g-3pt"
    # ["250n250a-trainset3-g-3pt"]="lenet5-250n250a-trainset3-g-3pt"
    # ["250n250a-trainset4-g-3pt"]="lenet5-250n250a-trainset4-g-3pt"
    # ["250n250a-trainset5-g-3pt"]="lenet5-250n250a-trainset5-g-3pt"
    # ["250n250a-trainset6-g-3pt"]="lenet5-250n250a-trainset6-g-3pt"
    # ["250n250a-trainset7-g-3pt"]="lenet5-250n250a-trainset7-g-3pt"
    # ["250n250a-trainset8-g-3pt"]="lenet5-250n250a-trainset8-g-3pt"
    # ["250n250a-trainset9-g-3pt"]="lenet5-250n250a-trainset9-g-3pt"
    # ["250n250a-trainset10-g-3pt"]="lenet5-250n250a-trainset10-g-3pt"

    # ["250n250a-trainset1-g-5pt"]="lenet5-250n250a-trainset1-g-5pt"
    # ["250n250a-trainset2-g-5pt"]="lenet5-250n250a-trainset2-g-5pt"
    # ["250n250a-trainset3-g-5pt"]="lenet5-250n250a-trainset3-g-5pt"
    # ["250n250a-trainset4-g-5pt"]="lenet5-250n250a-trainset4-g-5pt"
    # ["250n250a-trainset5-g-5pt"]="lenet5-250n250a-trainset5-g-5pt"
    # ["250n250a-trainset6-g-5pt"]="lenet5-250n250a-trainset6-g-5pt"
    # ["250n250a-trainset7-g-5pt"]="lenet5-250n250a-trainset7-g-5pt"
    # ["250n250a-trainset8-g-5pt"]="lenet5-250n250a-trainset8-g-5pt"
    # ["250n250a-trainset9-g-5pt"]="lenet5-250n250a-trainset9-g-5pt"
    # ["250n250a-trainset10-g-5pt"]="lenet5-250n250a-trainset10-g-5pt"

    # ["250n250a-trainset1-g-7pt"]="lenet5-250n250a-trainset1-g-7pt"
    # ["250n250a-trainset2-g-7pt"]="lenet5-250n250a-trainset2-g-7pt"
    # ["250n250a-trainset3-g-7pt"]="lenet5-250n250a-trainset3-g-7pt"
    # ["250n250a-trainset4-g-7pt"]="lenet5-250n250a-trainset4-g-7pt"
    # ["250n250a-trainset5-g-7pt"]="lenet5-250n250a-trainset5-g-7pt"
    # ["250n250a-trainset6-g-7pt"]="lenet5-250n250a-trainset6-g-7pt"
    # ["250n250a-trainset7-g-7pt"]="lenet5-250n250a-trainset7-g-7pt"
    # ["250n250a-trainset8-g-7pt"]="lenet5-250n250a-trainset8-g-7pt"
    # ["250n250a-trainset9-g-7pt"]="lenet5-250n250a-trainset9-g-7pt"
    # ["250n250a-trainset10-g-7pt"]="lenet5-250n250a-trainset10-g-7pt"

    # ["250n250a-trainset1-g-3pt-x50"]="lenet5-250n250a-trainset1-g-3pt-x50"
    # ["250n250a-trainset2-g-3pt-x50"]="lenet5-250n250a-trainset2-g-3pt-x50"
    # ["250n250a-trainset3-g-3pt-x50"]="lenet5-250n250a-trainset3-g-3pt-x50"
    # ["250n250a-trainset4-g-3pt-x50"]="lenet5-250n250a-trainset4-g-3pt-x50"
    # ["250n250a-trainset5-g-3pt-x50"]="lenet5-250n250a-trainset5-g-3pt-x50"
    # ["250n250a-trainset6-g-3pt-x50"]="lenet5-250n250a-trainset6-g-3pt-x50"
    # ["250n250a-trainset7-g-3pt-x50"]="lenet5-250n250a-trainset7-g-3pt-x50"
    # ["250n250a-trainset8-g-3pt-x50"]="lenet5-250n250a-trainset8-g-3pt-x50"
    # ["250n250a-trainset9-g-3pt-x50"]="lenet5-250n250a-trainset9-g-3pt-x50"
    # ["250n250a-trainset10-g-3pt-x50"]="lenet5-250n250a-trainset10-g-3pt-x50"

    # ["250n250a-trainset1-g-5pt-x50"]="lenet5-250n250a-trainset1-g-5pt-x50"
    # ["250n250a-trainset2-g-5pt-x50"]="lenet5-250n250a-trainset2-g-5pt-x50"
    # ["250n250a-trainset3-g-5pt-x50"]="lenet5-250n250a-trainset3-g-5pt-x50"
    # ["250n250a-trainset4-g-5pt-x50"]="lenet5-250n250a-trainset4-g-5pt-x50"
    # ["250n250a-trainset5-g-5pt-x50"]="lenet5-250n250a-trainset5-g-5pt-x50"
    # ["250n250a-trainset6-g-5pt-x50"]="lenet5-250n250a-trainset6-g-5pt-x50"
    # ["250n250a-trainset7-g-5pt-x50"]="lenet5-250n250a-trainset7-g-5pt-x50"
    # ["250n250a-trainset8-g-5pt-x50"]="lenet5-250n250a-trainset8-g-5pt-x50"
    # ["250n250a-trainset9-g-5pt-x50"]="lenet5-250n250a-trainset9-g-5pt-x50"
    # ["250n250a-trainset10-g-5pt-x50"]="lenet5-250n250a-trainset10-g-5pt-x50"    

    # ["250n250a-trainset1-g-7pt-x50"]="lenet5-250n250a-trainset1-g-7pt-x50"
    # ["250n250a-trainset2-g-7pt-x50"]="lenet5-250n250a-trainset2-g-7pt-x50"
    # ["250n250a-trainset3-g-7pt-x50"]="lenet5-250n250a-trainset3-g-7pt-x50"
    # ["250n250a-trainset4-g-7pt-x50"]="lenet5-250n250a-trainset4-g-7pt-x50"
    # ["250n250a-trainset5-g-7pt-x50"]="lenet5-250n250a-trainset5-g-7pt-x50"
    # ["250n250a-trainset6-g-7pt-x50"]="lenet5-250n250a-trainset6-g-7pt-x50"
    # ["250n250a-trainset7-g-7pt-x50"]="lenet5-250n250a-trainset7-g-7pt-x50"
    # ["250n250a-trainset8-g-7pt-x50"]="lenet5-250n250a-trainset8-g-7pt-x50"
    # ["250n250a-trainset9-g-7pt-x50"]="lenet5-250n250a-trainset9-g-7pt-x50"
    # ["250n250a-trainset10-g-7pt-x50"]="lenet5-250n250a-trainset10-g-7pt-x50"    

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


    # ["250n250aset1-g-7pt"]="lenet5-250n250aset1-g-7pt"
    # ["250n250aset2-g-7pt"]="lenet5-250n250aset2-g-7pt"
    # ["250n250aset3-g-7pt"]="lenet5-250n250aset3-g-7pt"
    # ["250n250aset4-g-7pt"]="lenet5-250n250aset4-g-7pt"
    # ["250n250aset5-g-7pt"]="lenet5-250n250aset5-g-7pt"
    # ["250n250aset6-g-7pt"]="lenet5-250n250aset6-g-7pt"
    # ["250n250aset7-g-7pt"]="lenet5-250n250aset7-g-7pt"
    # ["250n250aset8-g-7pt"]="lenet5-250n250aset8-g-7pt"
    # ["250n250aset9-g-7pt"]="lenet5-250n250aset9-g-7pt"
    # ["250n250aset10-g-7pt"]="lenet5-250n250aset10-g-7pt"

    # ["2500n2500a-train-checklaiset1"]="lenet5-kernel5-2500n2500a-train-checklai"

    # ["2500n2500a-trainset1"]="lenet5-kernel5-2500n2500a-trainset1"

)

# Define model order
model_order=(

    # "50n50a-trainset1"
    # "50n50a-trainset2"
    # "50n50a-trainset3"
    # "50n50a-trainset4"
    # "50n50a-trainset5"
    # "50n50a-trainset6"
    # "50n50a-trainset7"
    # "50n50a-trainset8"
    # "50n50a-trainset9"
    # "50n50a-trainset10"


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

    # "250n250a-trainset1"
    # "250n250a-trainset2"
    # "250n250a-trainset3"
    # "250n250a-trainset4"
    # "250n250a-trainset5"
    "250n250a-trainset6"
    # "250n250a-trainset7"
    # "250n250a-trainset8"
    # "250n250a-trainset9"
    # "250n250a-trainset10"

    # "250n250a-trainset1-g-3pt"
    # "250n250a-trainset2-g-3pt"
    # "250n250a-trainset3-g-3pt"
    # "250n250a-trainset4-g-3pt"
    # "250n250a-trainset5-g-3pt"
    # "250n250a-trainset6-g-3pt"
    # "250n250a-trainset7-g-3pt"
    # "250n250a-trainset8-g-3pt"
    # "250n250a-trainset9-g-3pt"
    # "250n250a-trainset10-g-3pt"

    # "250n250a-trainset1-g-5pt"
    # "250n250a-trainset2-g-5pt"
    # "250n250a-trainset3-g-5pt"
    # "250n250a-trainset4-g-5pt"
    # "250n250a-trainset5-g-5pt"
    # "250n250a-trainset6-g-5pt"
    # "250n250a-trainset7-g-5pt"
    # "250n250a-trainset8-g-5pt"
    # "250n250a-trainset9-g-5pt"
    # "250n250a-trainset10-g-5pt"

    # "250n250a-trainset1-g-7pt"
    # "250n250a-trainset2-g-7pt"
    # "250n250a-trainset3-g-7pt"
    # "250n250a-trainset4-g-7pt"
    # "250n250a-trainset5-g-7pt"
    # "250n250a-trainset6-g-7pt"
    # "250n250a-trainset7-g-7pt"
    # "250n250a-trainset8-g-7pt"
    # "250n250a-trainset9-g-7pt"
    # "250n250a-trainset10-g-7pt"

    # "250n250a-trainset1-g-9pt"
    # "250n250a-trainset2-g-9pt"
    # "250n250a-trainset3-g-9pt"
    # "250n250a-trainset4-g-9pt"
    # "250n250a-trainset5-g-9pt"
    # "250n250a-trainset6-g-9pt"
    # "250n250a-trainset7-g-9pt"
    # "250n250a-trainset8-g-9pt"
    # "250n250a-trainset9-g-9pt"
    # "250n250a-trainset10-g-9pt"

    # "250n250a-trainset1-g-3pt-x50"
    # "250n250a-trainset2-g-3pt-x50"
    # "250n250a-trainset3-g-3pt-x50"
    # "250n250a-trainset4-g-3pt-x50"
    # "250n250a-trainset5-g-3pt-x50"
    # "250n250a-trainset6-g-3pt-x50"
    # "250n250a-trainset7-g-3pt-x50"
    # "250n250a-trainset8-g-3pt-x50"
    # "250n250a-trainset9-g-3pt-x50"
    # "250n250a-trainset10-g-3pt-x50"
    
    # "250n250a-trainset1-g-5pt-x50"
    # "250n250a-trainset2-g-5pt-x50"
    # "250n250a-trainset3-g-5pt-x50"
    # "250n250a-trainset4-g-5pt-x50"
    # "250n250a-trainset5-g-5pt-x50"
    # "250n250a-trainset6-g-5pt-x50"
    # "250n250a-trainset7-g-5pt-x50"
    # "250n250a-trainset8-g-5pt-x50"
    # "250n250a-trainset9-g-5pt-x50"
    # "250n250a-trainset10-g-5pt-x50"

    # "250n250a-trainset1-g-7pt-x50"
    # "250n250a-trainset2-g-7pt-x50"
    # "250n250a-trainset3-g-7pt-x50"
    # "250n250a-trainset4-g-7pt-x50"
    # "250n250a-trainset5-g-7pt-x50"
    # "250n250a-trainset6-g-7pt-x50"
    # "250n250a-trainset7-g-7pt-x50"
    # "250n250a-trainset8-g-7pt-x50"
    # "250n250a-trainset9-g-7pt-x50"
    # "250n250a-trainset10-g-7pt-x50"


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

    # "250n250aset1-g-7pt"
    # "250n250aset2-g-7pt"
    # "250n250aset3-g-7pt"
    # "250n250aset4-g-7pt"
    # "250n250aset5-g-7pt"
    # "250n250aset6-g-7pt"
    # "250n250aset7-g-7pt"
    # "250n250aset8-g-7pt"
    # "250n250aset9-g-7pt"
    # "250n250aset10-g-7pt"

    # "2500n2500a-train-checklaiset1"

    # "2500n2500a-trainset1"
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
