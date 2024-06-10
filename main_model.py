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
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Lenet-5
#Defining the convolutional neural network
# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet5, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1, 6, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(6, 16, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
        
#         # Calculate the output size of layer2
#         self._to_linear = None
#         x = torch.randn(1, 1, 1000)  # Assuming input size of (1, 1000)
#         self.convs(x)
        
#         self.fc1 = nn.Linear(self._to_linear, 120)  # Adjust input size based on the calculated value
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(84, num_classes)
        
#     def convs(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         if self._to_linear is None:
#             self._to_linear = x[0].shape[0]*x[0].shape[1]
#             print(self._to_linear)
            
#         return x
        
#     def forward(self, x):
#         x = self.convs(x)
#         x = x.view(-1, self._to_linear)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         return x


#Lenet-5
#Defining the convolutional neural network
# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet5, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1, 6, kernel_size=5, stride=1),  # Change the input channels from 3 to 1
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(6, 16, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))
#         self.fc1 = nn.Linear(988*4, 120)  # Adjust the input size based on the output size of the last convolutional layer
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(84, num_classes)
        
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)  # Reshape the output of the convolutional layers
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         return out

# model = LeNet5(num_classes=2)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)








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
    if verbose:
        print('Extracting features and labels from the data...')
        
    #Plot loss, accuracy by epochs
    train_loss_data = []
    train_accuracy_data = []
    validation_accuracy_data = []
        
    #Set to train
    # model.train()

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
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
            print(outputs, "output")
            _, predicted = torch.max(outputs, 1)
            print(predicted," predictedss")
            total += 1
            correct += (predicted == label_tensor).sum().item()
            print(f"Epoch {epoch+1}/{num_epochs}") 
            print(f"Running_loss: {(running_loss / total):.4f}, Running_accuracy: {(correct / total):.4f}")

            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_loss_data.append(epoch_loss)
        train_accuracy_data.append(epoch_acc)
        
        #Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(num_validation_records):
                if verbose:
                    width = len(str(num_validation_records))
                    print("Validating...")
                    print(f'- {i+1:>{width}}/{num_validation_records}: {validation_records[i]}...')

                validation_record = os.path.join(validation_folder, validation_records[i])
                print(validation_record)
    
                #SIGNAL and label load
                signal, label = load_raw_data_ptbxl(100,validation_record) #dang la lay lead 1, hoac lead 2 -> clear hon
                print("Validation label loaded:", label)
                
                signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Convert numpy array to tensor and add batch dimension
                label_tensor = torch.tensor(label, dtype=torch.long)
                outputs = model(signal_tensor)
                
                _, predicted = torch.max(outputs, 1)
                print(predicted," validation predicted")
                total += 1
                correct += (predicted == label_tensor).sum().item()    
        val_acc = correct / total
        validation_accuracy_data.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.4f}")
        
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
    plt.show()
    
    # Save the data of train, validation
    np.save(f'{model_name} {data_folder} {num_epochs} {num_validation_records} {num_records} train_loss_data.npy', train_loss_data)
    np.save(f'{model_name} {data_folder} {num_epochs} {num_validation_records} {num_records} train_accuracy_data.npy', train_accuracy_data)
    np.save(f'{model_name} {data_folder} {num_epochs} {num_validation_records} {num_records} validation_accuracy_data.npy', validation_accuracy_data)







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
    
    #real_model = LeNet5(num_classes=2)
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
