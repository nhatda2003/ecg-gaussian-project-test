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

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
import numpy as np

from torchvision import models

import matrixprofile
###############



#######################################################################################################################################################3


# Resnet18
# class BasicBlock1D(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = torch.relu(out)
#         return out

# # Define the ResNet architecture for 1D signals
# class ResNet1D(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=2):
#         super(ResNet1D, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv1d(1, 64, kernel_size=32, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = torch.nn.functional.avg_pool1d(out, out.size(2))
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNet18_1D(num_classes):
#     return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes)

# # Load the model, loss function, and optimizer
# model = ResNet18_1D(num_classes=2)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Check if GPU is available and move the model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)



#################################################################################################################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #Lenet-5-2D
# #Defining the convolutional neural network
# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet5, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(6),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#         # Adaptive pooling layer to standardize the output size
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # output size of 7x7

#         # Fully connected layers
#         self.fc = nn.Linear(16 * 7 * 7, 120)  # Size must match the output of adaptive pool
#         self.fc1 = nn.Linear(120, 84)
#         self.fc2 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.adaptive_pool(out)
#         out = out.view(out.size(0), -1)  # Flatten the tensor
#         out = F.relu(self.fc(out))
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#         return out
    
# model = LeNet5(num_classes=2)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Check if GPU is available and move the model to GPU

# model = model.to(device)



##############################################################################################################################3
#Lenet-5-1D
#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=50, stride=1),  # Change the input channels from 3 to 1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(3696, 120)  # Adjust the input size based on the output size of the last convolutional layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Reshape the output of the convolutional layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = LeNet5(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Train classification model.
def train_dx_model(data_folder, validation_folder, model_folder, model_scenario_name, verbose):
    # Find data files.
    if verbose:
        print('Training the model...')
        print('Finding the data...')

    records = find_records(data_folder)
    
    #Add validation records
    validation_records = find_records (validation_folder)
    num_validation_records = len(validation_records)
    
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    #if verbose:
        #print('Extracting features and labels...')
        
    #Plot loss, accuracy by epochs
    train_loss_data = []
    train_accuracy_data = []
    validation_accuracy_data = []
        
    #Set to train
    # model.train()

    num_epochs = 10 #testing for x50

    #num_epochs = 30 #testing for 50n50a
    
    #num_epochs = 20  # Testing for 250n250a
    #num_epochs = 25  # Testing for 250n250a
    
    #num_epochs = 15 # for 2500n2500a

    # Outer loop for epochs
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for progress bar
        print(f"{model_scenario_name} {data_folder} {validation_folder}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        with tqdm(total=num_records, desc="Trainning", unit=" record") as pbar:
            for i in range(num_records):
                pbar.update(1)  # Update progress bar
                record = os.path.join(data_folder, records[i])
                #print(record)
    
                #SIGNAL and label load
                signal, label = load_raw_data_ptbxl(100,record) #dang la lay lead 1, hoac lead 2 -> clear hon

                #print("label loaded:", label)

                #Prepare for 1D
                stomp = matrixprofile.algorithms.stomp(signal, 20)['mp']
                #print(np.shape(stomp))
                signal_tensor = torch.tensor(stomp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Convert numpy array to tensor and add batch dimension
                label_tensor = torch.tensor(label, dtype=torch.long).to(device)
            
                #Prepare for 2D
                #signal_tensor = matrixprofile.compute(signal,preprocessing_kwargs={'window':20})
                #print(signal_tensor['pmp'])
                # pmp = matrixprofile.compute(signal,preprocessing_kwargs={'window':50})['pmp']
                # print(np.shape(pmp))
                # signal_tensor = torch.tensor(pmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                # label_tensor = torch.tensor(label, dtype=torch.long).to(device)
                

                optimizer.zero_grad()
                outputs = model(signal_tensor)
                loss = criterion(outputs, label_tensor.unsqueeze(0))  # Add batch dimension to target for the loss function
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                #print(outputs, "output")
                _, predicted = torch.max(outputs, 1)
                #print(predicted," predictedss")
                total += 1
                correct += (predicted == label_tensor).sum().item()
            

        #print(f"Epoch {epoch+1}/{num_epochs}") 
        
        Running_accuracy = correct / total
        #print(f"Running_accuracy: {(correct / total):.4f}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss_data.append(epoch_loss)
        train_accuracy_data.append(epoch_acc)
        
        #Validation
        model.eval()
        correct = 0
        total = 0

        # Use tqdm for progress bar
        with torch.no_grad():
            with tqdm(total=num_validation_records, desc="Validation", unit=" record") as pbar:
                for i in range(num_validation_records):
                    pbar.update(1)  # Update progress bar
                    validation_record = os.path.join(validation_folder, validation_records[i])
                    #print(validation_record)
        
                    #SIGNAL and label load
                    signal, label = load_raw_data_ptbxl(100,validation_record) #dang la lay lead 1, hoac lead 2 -> clear hon
                    #print("Validation label loaded:", label)
                    
                    #For 1-D
                    stomp = matrixprofile.algorithms.stomp(signal, 20)['mp']
                    signal_tensor = torch.tensor(stomp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Convert numpy array to tensor and add batch dimension
                    label_tensor = torch.tensor(label, dtype=torch.long).to(device)
                    
                    #For 2-D
                    # pmp = matrixprofile.compute(signal,preprocessing_kwargs={'window':20})['pmp']
                    # signal_tensor = torch.tensor(pmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    # label_tensor = torch.tensor(label, dtype=torch.long).to(device)
                    
                    
                    outputs = model(signal_tensor)
                    
                    _, predicted = torch.max(outputs, 1)
                    #print(predicted," validation predicted")
                    total += 1
                    correct += (predicted == label_tensor).sum().item()    
        val_acc = correct / total
        validation_accuracy_data.append(val_acc)
        print(f"Train_loss: {(running_loss / total):.4f}")
        print(f"Train_accuracy: {Running_accuracy:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print()
        
        #Check if train_accuracy too high:
        # if Running_accuracy >0.99:
        #     while len(train_loss_data) < num_epochs:
        #         train_loss_data.append(-1)
        #     while len(train_accuracy_data) < num_epochs:
        #         train_accuracy_data.append(-1)
        #     while len(validation_accuracy_data) < num_epochs:
        #         validation_accuracy_data.append(-1)


        #     break
        
    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model.
    model_name = model_scenario_name + ".pth"
    model_path = os.path.join(model_folder, model_name)
    torch.save(model.state_dict(), model_path)

    if verbose:
        print()
        print(f"Done, model name: {model_scenario_name}")
    
    
    
    
    
    # Plotting
    plt.plot(range(1, num_epochs + 1), train_loss_data, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), train_accuracy_data, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), validation_accuracy_data, label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Plot of Train and Validation Metrics')
    plt.legend()  # Show legend to distinguish between train and validation data


    # Find the 5 highest validation accuracy values and their corresponding epochs
    top_values = sorted(((acc, epoch) for epoch, acc in enumerate(validation_accuracy_data, start=1)), reverse=True)[:5]

    # Annotate the plot with the top 5 values
    for acc, epoch in top_values:
        plt.annotate(f'{acc:.2f}', (epoch, acc), textcoords="offset points", xytext=(-10,10), ha='center')

    # Save the plot
    plt.savefig(f'plot_{model_name}_{data_folder}_{num_epochs}ep_{num_validation_records}val_{num_records}train.png')  # Specify the file name and extension
    #plt.show()
    
    # Save the data of train, validation
    data_to_compare_folder = "data-to-compare"
    if not os.path.exists(data_to_compare_folder):
        os.makedirs(data_to_compare_folder)
    np.save(os.path.join(data_to_compare_folder, f"{model_scenario_name}train_loss_data.npy"), train_loss_data)
    np.save(os.path.join(data_to_compare_folder, f"{model_scenario_name}train_accuracy_data.npy"), train_accuracy_data)
    np.save(os.path.join(data_to_compare_folder, f"{model_scenario_name}validation_accuracy_data.npy"), validation_accuracy_data)






















# Other functions
##########################################################################################################################################
def load_dx_model(model_folder, model_name, verbose):
    model_name = model_name+".pth"
    filename = os.path.join(model_folder, model_name)
    return torch.load(filename, map_location=torch.device('cpu'))  # Use map_location argument to load model parameters
    #NHATedit: changed the function


def run_dx_model(dx_model, record, signal, verbose):
  
    signal, label = load_raw_data_ptbxl(100, record)
    #print("label:", label)
    
        #Test plot if need
    
    # plt.plot(signal)
    # plt.title('Plot of 1D NumPy Array')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.show()
    # raise Exception("plot ok")
    
    #Prepare for 1D    
    stomp = matrixprofile.algorithms.stomp(signal, 20)['mp']
    signal_tensor = torch.tensor(stomp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch dimension
    signal_tensor = signal_tensor.to(device)

    
    #Prepare for 2D
    #signal_tensor = matrixprofile.compute(signal,preprocessing_kwargs={'window':20})
    #print(signal_tensor['pmp'])
    # pmp = matrixprofile.compute(signal,preprocessing_kwargs={'window':20})['pmp']
    # signal_tensor = torch.tensor(pmp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # label_tensor = torch.tensor(label, dtype=torch.long).to(device)

    
    #Choose model
    real_model = LeNet5(num_classes=2)
    #real_model = ResNet50(in_channels=1,classes=2)
    #real_model = ResNet18_1D(num_classes=2)
    
    real_model.load_state_dict(dx_model)
    real_model = real_model.to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = real_model(signal_tensor)
        #print(outputs)
       
        probabilities = torch.softmax(outputs, dim=1)
        #print(probabilities)
        #raise Exception("Test out put")
        predicted_label_index = torch.argmax(probabilities, dim=1).item()
        #print("predicted output:",predicted_label_index,"!!")
        #raise Exception("Test out put")
        #redicted_label = 'Normal' if predicted_label_index == 0 else 'Abnormal'
        #print("predicted_label",predicted_label)

    ##########################################################################
    #predicted_label = [predicted_label]

    return predicted_label_index


# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'dx_model.sav')
    
    #NHATnote: Su dung joblib de luu
    joblib.dump(d, filename, protocol=0)
