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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import models
###############


#Resnet50
class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet50(torch.nn.Module):
    def __init__(self,in_channels=2,classes=125):
        super(ResNet50, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=5,stride=1,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x


# Initialize model, loss, and optimizer
model = ResNet50(in_channels=1,classes=2)
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

    num_epochs = 50
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(num_records):
            if verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            record = os.path.join(data_folder, records[i])
            print(record)
 
            #SIGNAL and label load
            signal, label = load_raw_data_ptbxl(100,record) #dang la lay lead 1, hoac lead 2 -> clear hon
            print("label loaded:", label)

            #Prepare
            signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Convert numpy array to tensor and add batch dimension
            #print(signal_tensor.shape)
            #raise Exception("???")
            label_tensor = torch.tensor(label, dtype=torch.long)
           
            optimizer.zero_grad()
            outputs = model(signal_tensor)
            loss = criterion(outputs, label_tensor.unsqueeze(0))  # Add batch dimension to target for the loss function
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            print(predicted," predicted")
            total += 1
            correct += (predicted == label_tensor).sum().item()
            print(f"Epoch {epoch}/{num_epochs}") 
            ## / {num_epochs} \nRunning_loss: {(running_loss / total):.4f}")
            #print(f"Running_accuracy: {(correct / total):.4f}")
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"######################Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


            # # Validation
            # model.eval()
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for signals, labels in val_loader:
            #         outputs = model(signals)
            #         _, predicted = torch.max(outputs, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()
            # val_acc = correct / total
            # print(f"Validation Accuracy: {val_acc:.4f}")
        
    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model.
    model_name = model_scenario_name + ".pth"
    model_path = os.path.join(model_folder, model_name)
    torch.save(model.state_dict(), model_path)

    if verbose:
        print()
        print(f"Done, model name: {model_scenario_name}")
    
    # if num_epochs == 1:
    #     print("num_label_norm",num_label_norm)
    #     print("num_label_abno",num_label_abno)








# Other functions
##########################################################################################################################################
def load_dx_model(model_folder, model_name, verbose):
    model_name = model_name+".pth"
    filename = os.path.join(model_folder, model_name)
    return torch.load(filename, map_location=torch.device('cpu'))  # Use map_location argument to load model parameters
    #NHATedit: changed the function


def run_dx_model(dx_model, record, signal, verbose):
  
    signal, label = load_raw_data_ptbxl(100, record)
    print("label:", label)
    
        #Test plot if need
    
    # plt.plot(signal)
    # plt.title('Plot of 1D NumPy Array')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()
    # raise Exception("plot ok")
    
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch dimension
    #dx_model.eval()
    real_model = ResNet50(in_channels=1,classes=2)
    real_model.load_state_dict(dx_model)
    
    # Perform inference
    with torch.no_grad():
        outputs = real_model(signal_tensor)
        #print(outputs)
       
        probabilities = torch.softmax(outputs, dim=1)
        print(probabilities)
        #raise Exception("Test out put")
        predicted_label_index = torch.argmax(probabilities, dim=1).item()
        print("predicted output:",predicted_label_index,"!!")
        #raise Exception("Test out put")
        predicted_label = 'Normal' if predicted_label_index == 0 else 'Abnormal'
        print("predicted_label",predicted_label)

    ##########################################################################
    #predicted_label = [predicted_label]

    return predicted_label_index


# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    
    #NHATnote: Su dung joblib de luu
    joblib.dump(d, filename, protocol=0)