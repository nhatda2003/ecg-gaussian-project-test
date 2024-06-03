import joblib
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






# Train classification model.
def train_dx_model(data_folder, model_folder, model_scenario_name, verbose):
    # Find data files.
    if verbose:
        print('Training the model...')
        print('Finding the data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')
        
    #Set to train
    model.train()

    #Start epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i in range(num_records):
            if verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            record = os.path.join(data_folder, records[i])
            print(record)
                        
            # Extract the features from the image, but only if the image has one or more dx classes.
            dx = load_dx(record) #NHATnote: Doan nay lay ra label Normal/ Abnormal
            print(dx, "Label readed")
 
            #SIGNAL and label load
            signal = load_raw_data_ptbxl(100,record) #dang la lay lead 1, hoac lead 2 -> clear hon
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

    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model.
    model_name = model_scenario_name + ".pth"
    model_path = os.path.join(model_folder, model_name)
    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"Done, model name: {model_scenario_name}")








# Other functions
##########################################################################################################################################
def load_dx_model(model_folder, model_name, verbose):
    model_name = model_name+".pth"
    filename = os.path.join(model_folder, model_name)
    return torch.load(filename, map_location=torch.device('cpu'))  # Use map_location argument to load model parameters
    #NHATedit: changed the function


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
    labels = [predicted_label]

    return labels


# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    
    #NHATnote: Su dung joblib de luu
    joblib.dump(d, filename, protocol=0)