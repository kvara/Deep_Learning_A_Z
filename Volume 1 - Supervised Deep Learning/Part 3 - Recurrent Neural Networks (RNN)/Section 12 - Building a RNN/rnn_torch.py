import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


dataset_train = pd.read_csv('./Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

prev_size = 60
hidden_size = 50
epochs = 200
num_layers = 2

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(prev_size , 1258):
    X_train.append(training_set_scaled[i - prev_size : i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train))

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))

class RNN(nn.Module):
    def __init__(self,prev_size,hidden_size,num_layers):
        super(RNN, self).__init__()
#        self.rnn1 = nn.LSTM(prev_size,hidden_size,dropout = 0.2)
        self.rnn1 = nn.GRU(input_size = prev_size , hidden_size = hidden_size , num_layers = num_layers , dropout =0.2)
        self.drop = nn.Dropout(0.2)
        self.fc2  = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc1  = nn.Linear(hidden_size,1)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new( self.num_layers , 1 , self.hidden_size).zero_())

    def forward(self,input,hidden):
        output , hn = self.rnn1(input,hidden)
        output = self.drop(output)
        output = self.relu(self.fc2(output))
        output = self.fc1(output)
        return output[0][0][0] , hn

model = RNN(prev_size,hidden_size,num_layers)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


model.train()

for epoch in range(0,epochs):
    loss_temp = 0
    hidden = model.init_hidden()

    for day in range(0,X_train.size(0)):

        X_train = Variable(X_train.float())
        y_train = Variable(y_train.float())

        hidden = Variable(hidden.data)
        
        optimizer.zero_grad()

        predict , hidden = model(X_train[day].unsqueeze(1),hidden)

        loss = loss_function(y_train[day],predict)
        loss.backward()
        optimizer.step()
        loss_temp += loss.data[0]
    loss_temp = loss_temp / X_train.size(0)
    
    print(f'epoch {epoch + 1} : loss is {loss_temp}')


# testing phase


# preparing data

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
y_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0] , 1 , X_test.shape[1]))
X_test = torch.from_numpy(X_test).float()

y_test = torch.from_numpy(np.array(inputs[-20:])).float()


# testing model

model.eval()
hidden = model.init_hidden()

predict_test = []
predict_train = []
loss_test = 0
loss_train = 0

for i in range(0,X_train.size(0)):
    p , _ = model(X_train[i].unsqueeze(1),hidden)
    loss = loss_function(y_train[i],p)
    loss_train += loss.data[0]
    predict_train.append(p)

print( f'train loss : {loss_train/X_train.size(0)}')
hidden = model.init_hidden()

for i in range(0,20):
    p , _ = model(X_test[i].unsqueeze(1),hidden)
    loss = loss_function(y_test[i],p)
    loss_test += loss.data[0]
    predict_test.append(p)

#print(predict)
print(f'test loss : {loss_test/20}')

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.plot(dataset_train['Open'])
plt.subplot(1, 2, 2)
plt.plot(predict_train)
plt.show()


plt.subplot(1, 2, 1)
plt.plot(dataset_test['Open'])
plt.subplot(1, 2, 2)
plt.plot(predict_test)
plt.show()
