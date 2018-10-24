#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:10:25 2018

@author: giorgi
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_dir = "./dataset"

model_name = "squeezenet"
num_classes = 2
batch_size = 32
num_epochs = 25
feature_extract = True
input_size = 32

data_transforms = {
    'training_set': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_set': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['training_set', 'test_set']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['training_set', 'test_set']}



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1      = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2      = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.pool       = nn.MaxPool2d(kernel_size=2)

        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()

        self.fc1        = nn.Linear(32*15*15,128)
        self.fc2        = nn.Linear(128,1)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
#        x = self.relu(self.conv2(x))
#        x = self.pool(x)
        x = x.view(-1,7200)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01,momentum=0.9)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    iter_loss = 0.0
    correct = 0
    iterations = 0
    net.train()
    for i,(items,classes) in enumerate(dataloaders_dict['training_set']):
        items = items.float()
        classes = classes.unsqueeze(0).float().transpose(0,1)
        optimizer.zero_grad()
        outputs = net(items)
        loss = criterion(outputs,classes)
        iter_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        predicted = (outputs > 0.5).float()
        correct += (predicted == classes.data).sum()
        iterations += 1
    train_loss.append(iter_loss/iterations)
    train_accuracy.append(100*correct/len(dataloaders_dict['training_set'].dataset))
    
    
    iter_loss = 0.0
    correct = 0
    iterations = 0
    net.eval()
    for i , (items,classes) in enumerate(dataloaders_dict['test_set']):
        items = items.float()
        classes = classes.unsqueeze(0).float().transpose(0,1)
        outputs = net(items)
        loss += criterion(outputs,classes).data[0]
        predicted = (outputs > 0.5).float()
        correct += (predicted == classes.data).sum()
        iterations += 1
    test_loss.append(loss/iterations)
    test_accuracy.append(correct * 100.0 / len(dataloaders_dict['test_set']))

    print(f'epoch : {epoch} - train loss : {train_loss[-1]} train acc : {train_accuracy[-1]} test loss : {test_loss[-1]} test acc : {test_accuracy[-1]} ')
