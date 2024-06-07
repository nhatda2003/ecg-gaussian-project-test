import joblib
import argparse
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from helper_code import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nhat_modifysignal import *
from datetime import datetime

###############
def run(args):
    print('Generating gaussianset...')
    data_folder = args.data_folder

    records = find_records(data_folder)
    num_records = len(records)
    width = len(str(num_records))

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')


    ###Generate parameter
    number_of_virtual_signal = 5
    

    for i in range(num_records):
 
        print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        print(record,"!!")
                    
        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record) #NHATnote: Doan nay lay ra label Normal/ Abnormal
        print(dx,"SAdasd")

        
        #SIGNAL input load
        signal, label = load_raw_data_ptbxl(100,record) #dang la lay lead 1, hoac lead 2 -> clear hon
        

        #SIGNAL generate
        subfolder_name = args.generate_folder
        
        #Original signal
        record_dat = os.path.join(data_folder, records[i] + "_raw100.npy")
        record_hea = os.path.join(data_folder, records[i] + "_label.npy")

        subfolder_path = os.path.join(os.getcwd(), subfolder_name)
        
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        new_file_name1 = "_"+f"{i+1}_raw100.npy"  # Specify your desired name here
        new_file_name2 = "_"+f"{i+1}_label.npy"  # Specify your desired name here
        new_file_path1 = os.path.join(subfolder_path, new_file_name1)
        new_file_path2 = os.path.join(subfolder_path, new_file_name2)
        shutil.copy(record_dat, new_file_path1)
        shutil.copy(record_hea, new_file_path2)

        
        #Gen_signal
        for j in range(1, number_of_virtual_signal+1):
            #Get the same .hea file
            new_file_name_hea = "_"+f"{i+1+num_records*j}_label.npy"  # Specify your desired name here
            new_file_path_hea = os.path.join(subfolder_path, new_file_name_hea)
            shutil.copy(record_hea, new_file_path_hea)
            
            
            #Save .npy the generate signal
            modify_signal = nhat_modify(signal, i)
            new_file_name_npy = "_"+f"{i+1+num_records*j}_raw100.npy" 
            new_file_name_npy_final = os.path.join(subfolder_path, new_file_name_npy)
            pickle.dump(modify_signal, open(new_file_name_npy_final, 'wb'), protocol=4)
            
            # placeholder_file = f"{i+1+num_records*j}.dat"
            # placeholder_file = os.path.join(subfolder_path, placeholder_file)
            # with open(placeholder_file, 'w') as f:
            #     pass  # This line creates a blank file
    print(f"Finished generating x{number_of_virtual_signal} gaussianset")


        
# Parse arguments.
def get_parser():
    description = 'Extracting smallset normal_abnormal...'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-s', '--generate_folder', type=str, required=True)
    
    return parser


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))