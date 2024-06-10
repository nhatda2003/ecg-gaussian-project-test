
import argparse
import os
import sys

from helper_code import *
from main_model import load_dx_model, run_dx_model

###########################################NHATedit
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pickle
import os


import torch
import torch.nn as nn
import torch.optim as optim
#####################################################3

# Parse arguments.
def get_parser():
    description = 'Run the trained Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-mn', '--model_name', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--allow_failures', action='store_true')
    return parser

# Run the code.
def run(args):
    # Load model(s).
    if args.verbose:
        print('Loading the Challenge model...')

    # You can use these functions to perform tasks, such as loading your model(s), that you only need to perform once.
    #NHATedit
    #digitization_model = load_digitization_model(args.model_folder, args.verbose) ### Teams: Implement this function!!!
    dx_model = load_dx_model(args.model_folder, args.model_name, args.verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if args.verbose:
        print('Finding the Challenge data...')

    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    args.output_folder = f"{args.output_folder}_{args.model_name}"
    os.makedirs(f"{args.output_folder}", exist_ok=True)

    # Run the team's model(s) on the Challenge data.
    if args.verbose:
        print('Running the Challenge model(s) on the Challenge data...')

    #NHATnote: chay qua va perfrom model tren tung cai record mot.
    # Iterate over the records.
    for i in range(num_records):
        if args.verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        data_record = os.path.join(args.data_folder, records[i])
        output_record = os.path.join(args.output_folder, records[i])

      
        signal = None  # No meaning
       
        try: 
            dx = run_dx_model(dx_model, data_record, signal, args.verbose) ### Teams: Implement this function!!!
        except:
            if args.allow_failures:
                if args.verbose >= 2:
                    print('... dx classification failed.')
                dx = None
            else:
                raise
        
        #Save the predicted label
        output_path = os.path.split(output_record)[0]
        #print(output_path)
        os.makedirs(output_path, exist_ok=True)
        pickle.dump(dx, open(output_record+'_label.npy', 'wb'), protocol=4)


    if args.verbose:
        print('Done.')

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))