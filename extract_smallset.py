
import numpy as np
import os
import shutil
import sys
import argparse
from helper_code import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from datetime import datetime

###############

def run(args):
    data_folder = args.data_folder
    records = find_records(data_folder)

    num_records = len(records)
    width = len(str(num_records))
    
    #Set before generate
    num_sets = 3
    num_save = 3000 #abnorm + norm = num_save 50n50a -> num_save = 100
    
    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    print('Extracting normal_abnormal small dataset')

    counter_save_n = 0
    counter_save_a = 0
    counter_sets = 0
        

    for i in range(num_records):
        print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        record = os.path.join(data_folder, records[i]) #Example: testfolder08000npy/08000_lr
        dx = load_dx(record) #NHATnote: Doan nay lay ra label Normal/ Abnormal
        print(dx, "dx")
        #print(record)
        if counter_save_n +  counter_save_a < num_save and counter_sets < num_sets:

            if dx == ['Normal'] and counter_save_n<(num_save//2):
             
                counter_save_n +=1
                #print(records[i])
                record_dat = os.path.join(data_folder, records[i] + "_raw100.npy")
                record_hea = os.path.join(data_folder, records[i] + "_label.npy")
                subfolder_name = args.extract_folder
                # Create the subfolder if it doesn't exist
                subfolder_path = os.path.join(os.getcwd(), subfolder_name+f"set{counter_sets+1}")
                
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                print(subfolder_path)
                    
                # Change the name of the copied file
                new_file_name1 = "_"+f"{counter_save_n + counter_save_a}_raw100.npy"  # Specify your desired name here
                new_file_name2 = "_"+f"{counter_save_n + counter_save_a}_label.npy"  # Specify your desired name here
                new_file_path1 = os.path.join(subfolder_path, new_file_name1)
                new_file_path2 = os.path.join(subfolder_path, new_file_name2)
                
                shutil.copy(record_dat, new_file_path1)
                shutil.copy(record_hea, new_file_path2)
                
                # #Change .hea name
                # hea_file_path = new_file_path2
                # new_file_name = f"{counter_save_n + counter_save_a}"  # Specify the new file name
                # change_file_name_in_hea_file(hea_file_path, new_file_name)
                                
                
            if dx == ['Abnormal'] and counter_save_a<(num_save//2):
                counter_save_a +=1
                record_dat = os.path.join(data_folder, records[i] + "_raw100.npy")
                record_hea = os.path.join(data_folder, records[i] + "_label.npy")
                subfolder_name = args.extract_folder
                # Create the subfolder if it doesn't exist
                subfolder_path = os.path.join(os.getcwd(), subfolder_name+f"set{counter_sets+1}")
                
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                print(subfolder_path)
                    
                # Change the name of the copied file
                new_file_name1 = "_"+f"{counter_save_n + counter_save_a}_raw100.npy"  # Specify your desired name here
                new_file_name2 = "_"+f"{counter_save_n + counter_save_a}_label.npy"  # Specify your desired name here
                new_file_path1 = os.path.join(subfolder_path, new_file_name1)
                new_file_path2 = os.path.join(subfolder_path, new_file_name2)
                
                
                shutil.copy(record_dat, new_file_path1)
                shutil.copy(record_hea, new_file_path2)
                
                #     #Change .hea name
                # hea_file_path = new_file_path2
                # new_file_name = f"{counter_save_n + counter_save_a}"  # Specify the new file name
                # change_file_name_in_hea_file(hea_file_path, new_file_name)

                
        elif counter_sets<num_sets:
            counter_sets += 1
            counter_save_n = 0
            counter_save_a = 0
        else:
            print(f"Finished save {num_save*num_sets} files as small set(s)")
            print(f"Original folder is {data_folder}")
            return
                

        
# Parse arguments.
def get_parser():
    description = 'Extracting smallset normal_abnormal...'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-s', '--extract_folder', type=str, required=True)
    
    return parser


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
