#!/usr/bin/env python

# Load libraries.
import argparse
import ast
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-s', '--statements_file', type=str, required=True) # scp_statements.csv
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Load the PTB-XL database.
    df = pd.read_csv(args.database_file, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load the SCP statements.
    dg = pd.read_csv(args.statements_file, index_col=0)

    # Identify the header files.
    records = find_records_ptbxl(args.input_folder)
  

    # Update the header files and copy the signal files.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)

        #raise Exception ("??")
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        # recording_date_string = row['recording_date']
        # date_string, time_string = recording_date_string.split(' ')
        # yyyy, mm, dd = date_string.split('-')
        # date_string = f'{dd}/{mm}/{yyyy}'

        # age = row['age']
        # age = cast_int_float_unknown(age)

        # sex = row['sex']
        # if sex == 0:
        #     sex = 'Male'
        # elif sex == 1:
        #     sex = 'Female'
        # else:
        #     sex = 'Unknown'

        # height = row['height']
        # height = cast_int_float_unknown(height)

        # weight = row['weight']
        # weight = cast_int_float_unknown(weight)

        # Extract the diagnostic superclasses.
        scp_codes = row['scp_codes']
        if 'NORM' in scp_codes:
            dx = 'Normal'
        else:
            dx = 'Abnormal'
        
        #Nhat edit, skip .hea
        out_folder = os.path.join(os.getcwd(), args.output_folder)
        #print(out_folder)
        os.makedirs(out_folder, exist_ok=True)
        #raise Exception("create folder")
        this_path = os.path.join(args.input_folder, record_basename)
        #print(this_path)
        output_path = os.path.join(args.output_folder, record_basename)
        
        data = [wfdb.rdsamp(this_path)]
        data = np.array([signal for signal, meta in data]) #data
        data = (data[0].T)
        data = data[0]   #Take lead I for the task
        
        #Test plot if need
        
        # plt.plot(data)
        # plt.title('Plot of 1D NumPy Array')
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.show()
        # raise Exception("plot ok")
        
        #Define [Normal Abnormal]
        if dx == 'Normal':
            label = 0
        elif dx == 'Abnormal':
            label = 1
        else:
            raise Exception("Error with dx, non Normal or Abnormal")
        pickle.dump(data, open(output_path+'_raw100.npy', 'wb'), protocol=4)
        pickle.dump(label, open(output_path+'_label.npy', 'wb'), protocol=4)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))