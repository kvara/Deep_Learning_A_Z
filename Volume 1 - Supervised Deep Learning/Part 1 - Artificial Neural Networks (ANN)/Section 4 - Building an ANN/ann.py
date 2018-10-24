#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:12:00 2018

@author: giorgi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import torch
import torch.nn as nn
import torch.nn.functional as F
#from utils.nn import get_pytorch_optimizer, get_pytorch_criterion, get_pytorch_activation

X_train_tensor  = torch.from_numpy(X_train).float()
y_train_tensor  = torch.from_numpy(y_train).unsqueeze(1).float()
X_test_tensor   = torch.from_numpy(X_test).float()
y_test_tensor   = torch.from_numpy(y_test).unsqueeze(1).float()


class Net(nn.Module):
    def __init__(self,input_size,hidden_1_size,hidden_2_size,out_size):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size,hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size,out_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

net = Net(11,16,16,1)

batch_size = 10
epochs = 100
# MSELoss

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.001 ,nesterov = True,momentum = 0.9,dampening = 0)

length = y_train_tensor.size(0)

for epoch in range(epochs):
    loss_temp = 0
    
    for items , labels in zip(X_train_tensor,y_train_tensor):
        items = items
        labels = labels

        net.train()
        
        optimizer.zero_grad()
        outputs = net(items)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_temp += loss.data[0]

    print(f'loss is : {loss_temp / length} , in epoch : {epoch}')
    
net.eval()
outputs_train = net(X_train_tensor)
predict = (outputs_train> 0.5).float()
acc = y_train_tensor == predict
print(100 * acc.sum().float()/X_train_tensor.size(0))

net.eval()
outputs_test = net(X_test_tensor)
predict = (outputs_test > 0.5).float()
acc = y_test_tensor == predict
print(100 * acc.sum().float()/X_test_tensor.size(0))
