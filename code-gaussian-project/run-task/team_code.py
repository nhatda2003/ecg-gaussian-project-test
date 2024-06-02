#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

########################################################################NHATedit
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from nhat_modifysignal import *


##################################################################################


###############Get time
from datetime import datetime

###############

# NHATedit
# Define the CNN model
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=20, stride=2) #NHATnote: modify kernel_size, stride bigger to watch regions
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * 60, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(self.drop(self.bn1(torch.relu(self.conv1(x)))))
        x = self.pool(self.drop(self.bn2(torch.relu(self.conv2(x)))))
        x = self.pool(self.drop(self.bn3(torch.relu(self.conv3(x)))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
#Instantaniate model
model = ECGClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

    



#NHATedit modify to read each file at once
def load_raw_data_ptbxl(sampling_rate, path):
    print("path in load_raw_data_ptbxl:", path)
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path)]
            #data = wfdb.rdrecord(path)
            #print("asdasdasdasdasdasdasdasd",data)
            #raise Exception("Stop check wfdb")
            data = np.array([signal for signal, meta in data]) #data
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    if len(data) == 1: #Check if this is the original full 12 lead data or just load .npy from single lead I/II
        data = (data[0].T)
        data = data[0]   
    #NHATnote: Default: Take lead I
    return data
###############################################################################NHATedit


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    print(verbose)
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    # features = list()
    # dxs = list()
    ######################################NHATedit
    model.train()
    num_epochs = 10
    
    
    ##################################NHATedit function to change the name in .hea file
    def change_file_name_in_hea_file(hea_file_path, new_file_name):
        first = 0
        
        with open(hea_file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith('#'):
                
                line = line
                new_lines.append(line)
            else:
                if first == 0:
                    first +=1
                    parts = line.split()
                    parts[0] = new_file_name
                else:
                    parts = line.split()
                    var = new_file_name+".dat"
                    parts[0] = var  # Change the file name
                parts.append("\n")
                new_line = ' '.join(parts)
    
                new_lines.append(new_line)

        # Write the modified lines back to the .hea file
        with open(hea_file_path, 'w') as f:
            f.writelines(new_lines)
        """    /*
        # Example usage:
        hea_file_path = 'path/to/your/file.hea'
        new_file_name = 'new_file_name.dat'  # Specify the new file name
        change_file_name_in_hea_file(hea_file_path, new_file_name)

    */"""
    ######################################
    
    
    ###Flags check to Saving 50n50a from datafolder only, nothing else
    extract_50n50a_only = False
    counter_save_n = 0
    counter_save_a = 0
    
    ###Flags check to generate virtual signal
    generate_virtual = False
    
    ###Generate parameter
    number_of_virtual_signal = 5
    
    if extract_50n50a_only and generate_virtual:
        raise Exception("extract_50n50a_only and generate_virtual both True, should choose 1")
    
    if (generate_virtual == True or extract_50n50a_only == True):
        num_epochs = 1
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(num_records):
            #print(num_records)
            #print("hehe")
            #raise Exception("HUHU")
            if verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            record = os.path.join(data_folder, records[i])
            print(record,"!!")
                        
            # Extract the features from the image, but only if the image has one or more dx classes.
            dx = load_dx(record) #NHATnote: Doan nay lay ra label Normal/ Abnormal
            print(dx,"SAdasd")

            if extract_50n50a_only == True:
                if counter_save_n +  counter_save_a < 100:
                    if dx == ['Normal'] and counter_save_n<50:
                        counter_save_n +=1
                        record_dat = os.path.join(data_folder, records[i] + ".dat")
                        record_hea = os.path.join(data_folder, records[i] + ".hea")
                        subfolder_name = "testfolder50n50a"
                        # Create the subfolder if it doesn't exist
                        subfolder_path1 = os.path.join(os.path.dirname(record_dat), subfolder_name)
                        subfolder_path2 = os.path.join(os.path.dirname(record_hea), subfolder_name)
                        
                        if not os.path.exists(subfolder_path1):
                            os.makedirs(subfolder_path1)
                        if not os.path.exists(subfolder_path2):
                            os.makedirs(subfolder_path2)
                            
                        # Change the name of the copied file
                        new_file_name1 = f"{counter_save_n + counter_save_a}.dat"  # Specify your desired name here
                        new_file_name2 = f"{counter_save_n + counter_save_a}.hea"  # Specify your desired name here
                        new_file_path1 = os.path.join(subfolder_path1, new_file_name1)
                        new_file_path2 = os.path.join(subfolder_path2, new_file_name2)
                        
                        shutil.copy(record_dat, new_file_path1)
                        shutil.copy(record_hea, new_file_path2)
                        
                        #Change .hea name
                        hea_file_path = new_file_path2
                        new_file_name = f"{counter_save_n + counter_save_a}"  # Specify the new file name
                        change_file_name_in_hea_file(hea_file_path, new_file_name)
                                        
                        
                    if dx == ['Abnormal'] and counter_save_a<50:
                        counter_save_a +=1
                        record_dat = os.path.join(data_folder, records[i] + ".dat")
                        record_hea = os.path.join(data_folder, records[i] + ".hea")
                        subfolder_name = "testfolder50n50a"
                        # Create the subfolder if it doesn't exist
                        subfolder_path1 = os.path.join(os.path.dirname(record_dat), subfolder_name)
                        subfolder_path2 = os.path.join(os.path.dirname(record_hea), subfolder_name)
                        
                        if not os.path.exists(subfolder_path1):
                            os.makedirs(subfolder_path1)
                        if not os.path.exists(subfolder_path2):
                            os.makedirs(subfolder_path2)
                            
                        # Change the name of the copied file
                        new_file_name1 = f"{counter_save_n + counter_save_a}.dat"  # Specify your desired name here
                        new_file_name2 = f"{counter_save_n + counter_save_a}.hea"  # Specify your desired name here
                        new_file_path1 = os.path.join(subfolder_path1, new_file_name1)
                        new_file_path2 = os.path.join(subfolder_path1, new_file_name2)
                        
                        
                        shutil.copy(record_dat, new_file_path1)
                        shutil.copy(record_hea, new_file_path2)
                        
                         #Change .hea name
                        hea_file_path = new_file_path2
                        new_file_name = f"{counter_save_n + counter_save_a}"  # Specify the new file name
                        change_file_name_in_hea_file(hea_file_path, new_file_name)
                        
                        
                else:
                    raise Exception("Finished save 100 file")
            #SIGNAL input load
            signal = load_raw_data_ptbxl(100,record) #dang la lay lead 1, hoac lead 2 -> clear hon
            
            if record == "generate_virtual_fix_std1phan5max/503":
                print(record)
                plt.plot(signal)
                plt.title('Signal Plot')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.grid(True)
                plt.show()        
                raise Exception("Test plot")
            
            #SIGNAL generate
            if generate_virtual:
                subfolder_name = "generate_virtual"
                
                #Original signal
                record_dat = os.path.join(data_folder, records[i] + ".dat")
                record_hea = os.path.join(data_folder, records[i] + ".hea")
                subfolder_path1 = os.path.join(os.path.dirname(record_dat), subfolder_name)
                subfolder_path2 = os.path.join(os.path.dirname(record_hea), subfolder_name)
                subfolder_path = os.path.join(os.path.dirname(record_hea), subfolder_name)
                
                if not os.path.exists(subfolder_path1):
                    os.makedirs(subfolder_path1)
                if not os.path.exists(subfolder_path2):
                    os.makedirs(subfolder_path2)
                new_file_name1 = f"{i+1}.dat"  # Specify your desired name here
                new_file_name2 = f"{i+1}.hea"  # Specify your desired name here
                new_file_path1 = os.path.join(subfolder_path1, new_file_name1)
                new_file_path2 = os.path.join(subfolder_path2, new_file_name2)
                shutil.copy(record_dat, new_file_path1)
                shutil.copy(record_hea, new_file_path2)
                hea_file_path = new_file_path2
                new_file_name = f"{i+1}"  # Specify the new file name
                change_file_name_in_hea_file(hea_file_path, new_file_name)
                
                #Gen_signal
               
                for j in range(1, number_of_virtual_signal+1):
                    #Get the same .hea file
                    new_file_name_hea = f"{i+1+num_records*j}.hea"  # Specify your desired name here
                    new_file_name_npy = f"{i+1+num_records*j}" 
                    new_file_path_hea = os.path.join(subfolder_path, new_file_name_hea)
                    shutil.copy(record_hea, new_file_path_hea)
                    new_file_name_hea = f"{i+1+num_records*j}"
                    change_file_name_in_hea_file(new_file_path_hea, new_file_name_hea)
                    
                    #Save .npy the generate signal
                    modify_signal = nhat_modify(signal, i)
                    new_file_name_npy_final = os.path.join(subfolder_path, new_file_name_npy)
                    pickle.dump(modify_signal, open(new_file_name_npy_final+'raw100.npy', 'wb'), protocol=4)
                    
                    placeholder_file = f"{i+1+num_records*j}.dat"
                    placeholder_file = os.path.join(subfolder_path, placeholder_file)
                    with open(placeholder_file, 'w') as f:
                        pass  # This line creates a blank file
                
                
               
               
            """
            signal_gen = f(signal) -> Viet ra 1 function de work voi signal
            copy cai file .hea ra doi ten thanh 101.hea
            pickle.dump(signal_gen, open(path+'raw100.npy', 'wb'), protocol=4)
            -> Can 1 file hea co ten va 1 file raw100.npy la duoc -> phai test  khi load lai ma chi luu co single lead 1000 points thi no work nhu nao
            
            phai test khi load lai thi sao
            """
            
            
            # print(min(signal))
            # x = np.arange(1000)
            # # Plot the signal
            # plt.plot(x, signal)
            # plt.title('Plot of Signal Data')
            # plt.xlabel('Index')
            # plt.ylabel('Signal Value')
            # plt.grid(True)
            # plt.show()
            # #raise Exception ("Test")
            

            label = dx
            
            ####
            X = signal  # Replace with your actual ECG signal of length 1000
            y = 0 if label == ['Normal'] else 1  # Replace with your actual label

            # Convert data to PyTorch tensors
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            y = torch.tensor(y, dtype=torch.long)  # Add batch dimension

            # Debugging: Print shapes
            print(f"Shape of X: {X.shape}")  # Should be [1, 1000]
            print(f"Shape of y: {y.shape}")  # Should be [1]

            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            y = torch.tensor(y, dtype=torch.long).unsqueeze(0)  # Add batch dimension to match the expected shape

            ####
      
            optimizer.zero_grad()
            outputs = model(X)  # Add channel dimension if needed
            loss = criterion(outputs, y)  # y is already 1D
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}')
            
            if generate_virtual == True:
                print("Generate virtual signals done!")

           
    
    os.makedirs(model_folder, exist_ok=True)
    
    model_name = 'dx_cnn_nhat_test1.pth'
    model_path = os.path.join(model_folder, model_name)
    torch.save(model.state_dict(), model_path)
    
    # # Save the model.
    # save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
 
    
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'dx_cnn_nhat_test1.pth')
    return torch.load(filename, map_location=torch.device('cpu'))  # Use map_location argument to load model parameters
    #NHATedit: changed the function

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
  
    signal = load_raw_data_ptbxl(100, record)
    
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch dimension
    #dx_model.eval()
    real_model = ECGClassifier()
    real_model.load_state_dict(dx_model)
    
    # Perform inference
    with torch.no_grad():
        outputs = real_model(signal_tensor)
        print(outputs)
        probabilities = torch.softmax(outputs, dim=1)
        print(probabilities)
        predicted_label_index = torch.argmax(probabilities, dim=1).item()
        print(predicted_label_index)
        predicted_label = 'Normal' if predicted_label_index == 0 else 'Abnormal'

    ##########################################################################
    
    if predicted_label == "Abnormal":
        print(record)
    else:
        print(1)
    labels = [predicted_label]

    return labels



################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

#NHATnote: Dang la extract feature

# Extract features.
def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        #print(image[0])
        #print("TESTSTETSET")
        #raise Exception('TESTSETSETEST') #NHATedit
        mean += np.mean(image)
        std += np.std(image)
    
    return np.array([mean, std])

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    
    #NHATnote: Su dung joblib de luu
    joblib.dump(d, filename, protocol=0)