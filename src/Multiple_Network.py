# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:21:10 2020

@author: abhishek
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from StochasticPool2D import StochasticPool2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride = 1, padding=1)
        self.pool1 = StochasticPool2D(2,2)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.conv3 = nn.Conv2d(64, 128, 3,1,1)
        self.pool2 = StochasticPool2D(2,2)
        self.conv4 = nn.Conv2d(128, 128, 3,1,1)
        self.conv5 = nn.Conv2d(128,128,3,1,1)
        self.pool3 = StochasticPool2D(2,2)
        self.fc1   = nn.Linear(6*6*128, 1024)
        self.fc2   = nn.Linear(1024,1024)
        self.fc3_1   = nn.Linear(1024, 7)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.fc3_2   = nn.Linear(1024, 7)
        self.fc3_3   = nn.Linear(1024, 7)
        self.fc3_4   = nn.Linear(1024, 7)
        self.fc3_5   = nn.Linear(1024, 7)
        self.fc3_6   = nn.Linear(1024, 7)
        
        self.w1 = torch.nn.Parameter(torch.rand(1))
        self.w2 = torch.nn.Parameter(torch.rand(1))
        self.w3 = torch.nn.Parameter(torch.rand(1))
        self.w4 = torch.nn.Parameter(torch.rand(1))
        self.w5 = torch.nn.Parameter(torch.rand(1))
        self.w6 = torch.nn.Parameter(torch.rand(1))
        
        
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        
        x = x.view(-1, 6*6*128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        #x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        #x = self.drop2(x)
        x_1 = (self.fc3_1(x))* self.w1
        x_2 = (self.fc3_2(x))* self.w2
        x_3 = (self.fc3_3(x))* self.w3
        x_4 = (self.fc3_4(x))* self.w4
        x_5 = (self.fc3_5(x))* self.w5
        x_6 = (self.fc3_6(x))* self.w6
        
        x = x_1 + x_2 + x_3 + x_4 + x_5 + x_6
        return x
    

#----------------------------- IMPOTING DATA---------------------------------------

data = pd.read_csv('C:/Users/abhis/Desktop/Emotion_Recog/fer2013.csv')

train = data[data['Usage']== 'Training']
val = data[data['Usage']== 'PrivateTest']
test = data[data['Usage']== 'PublicTest']

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomAffine(degrees=(-30,30), translate=(0.2,0.2), scale=(0.8,1.2), shear=10,fillcolor=1),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                ])

# Training Data
train_img = []

for i in train['pixels']:
    img = np.fromstring(i, dtype=float, sep=' ')
    img = img/255.0
    img = np.reshape(img, [1,48,48])
    img = torch.Tensor(img)
    img = transform(img)
    img = img.numpy()
    img = img.tolist()
    train_img.append(img)

train_img = np.array(train_img)


# Validation Data
val_img = []

for i in val['pixels']:
    img = np.fromstring(i, dtype=float, sep=' ')
    img = img/255.0
    img = np.reshape(img, [1,48,48])
    img = img.tolist()
    val_img.append(img)
    
val_img = np.array(val_img)

# Target Output
target_train = np.array(train['emotion'])
target_train = np.reshape(target_train, [target_train.size ])

target_val = np.array(val['emotion'])
target_val = np.reshape(target_val, [target_val.size])

#---------------------------------MODEL TRAINING--------------------------------------

def accuracy(y_var,output):
  acc = (y_var == output.argmax(-1)).float().detach().numpy()
  return float(100 * acc.sum() / len(acc))


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10
batch_size = 128
loss_history = []

for e in range(epochs):
    for i in range(0, train_img.shape[0], batch_size):

        x_mini = train_img[i: i+batch_size]
        y_mini = target_train[i: i+batch_size]
        
        x_mini = torch.Tensor(x_mini)
        y_mini = torch.LongTensor(y_mini)
        
        x_var = Variable(x_mini)
        y_var = Variable(y_mini)
    
        optimizer.zero_grad()
        output = net(x_var)
        loss = criterion(output, y_var)
        loss.backward()
        optimizer.step()
        
        if i%4096 == 0:
          acc = accuracy(y_var, output)
          print('Mini batch {}, loss: {} accuracy: {}'.format(i,loss, acc))
          loss_history.append(loss)
        
    x_validate = torch.Tensor(val_img)
    y_validate = torch.LongTensor(target_val)
    output_val = net(x_validate)
    loss_val = criterion(output_val, y_validate)
    acc_val = accuracy(y_validate, output_val)
    acc_train = accuracy(y_var, output)
    print("Epoch no {}: Train Loss={}, Train Accuracy={} Validation Loss={}, Validation Accuracy={}".format(e, loss, acc_train, loss_val, acc_val))

