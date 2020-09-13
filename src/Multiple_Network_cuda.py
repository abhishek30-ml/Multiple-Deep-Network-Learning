# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:21:10 2020

@author: abhishek
"""

# '/home/adc_2/Emotion_Dataset/fer2013.csv'


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
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = StochasticPool2D(2,2)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3,1,1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool2 = StochasticPool2D(2,2)
        self.conv4 = nn.Conv2d(128, 128, 3,1,1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,128,3,1,1)
        self.bn5   = nn.BatchNorm2d(128)
        self.pool3 = StochasticPool2D(2,2)
        self.conv6 = nn.Conv2d(128, 256, 3,1,1)
        self.bn6   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3,1,1)
        self.bn7   = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3,1,1)
        self.bn8   = nn.BatchNorm2d(256)
        self.pool4 = StochasticPool2D(2,2)

        self.fc1   = nn.Linear(6*6*128, 1024)
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2   = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.fc3_1   = nn.Linear(1024, 7)
        self.bn_fc3_1 = nn.BatchNorm1d(7)
        self.fc3_2   = nn.Linear(1024, 7)
        self.bn_fc3_2 = nn.BatchNorm1d(7)
        self.fc3_3   = nn.Linear(1024, 7)
        self.bn_fc3_3 = nn.BatchNorm1d(7)
        self.fc3_4   = nn.Linear(1024, 7)
        self.bn_fc3_4 = nn.BatchNorm1d(7)
        self.fc3_5   = nn.Linear(1024, 7)
        self.bn_fc3_5 = nn.BatchNorm1d(7)
        self.fc3_6   = nn.Linear(1024, 7)
        self.bn_fc3_6 = nn.BatchNorm1d(7)
        self.drop3 = nn.Dropout(p=0.15)
        
        self.weight_raw = torch.nn.Parameter(torch.rand(6,1), requires_grad = True)
        self.eps = 1e-7
        

        
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        #x = F.relu(self.bn6(self.conv6(x)))
        #x = F.relu(self.bn7(self.conv7(x)))
        #x = F.relu(self.bn8(self.conv8(x)))
        #x = self.pool4(x)

        x = x.view(-1, 6*6*128)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        
        x_1 = self.bn_fc3_1(self.fc3_1(x))
        x_1 = self.drop3(x_1)
        #x_1 = x_1* self.w1
        x_2 = self.bn_fc3_2(self.fc3_2(x))
        x_2 = self.drop3(x_2)
        #x_2 = x_2* self.w2
        x_3 = self.bn_fc3_3(self.fc3_3(x))
        x_3 = self.drop3(x_3)
       # x_3 = x_3* self.w3
        x_4 = self.bn_fc3_4(self.fc3_4(x))
        x_4 = self.drop3(x_4)
        #x_4 = x_4* self.w4
        x_5 = self.bn_fc3_5(self.fc3_5(x))
        x_5 = self.drop3(x_5)
        #x_5 = x_5* self.w5
        x_6 = self.bn_fc3_6(self.fc3_6(x))
        x_6 = self.drop3(x_6)
        #x_6 = x_6* self.w6
        
       
        weights = self.weight_raw / self.weight_raw.sum().clamp(min=self.eps)
        #x = x_1 + x_2 + x_3 + x_4 + x_5 + x_6
        return x_1, x_2, x_3, x_4, x_5, x_6, weights
    

#----------------------------- IMPOTING DATA---------------------------------------

data = pd.read_csv('/home/adc_2/Emotion_Dataset/fer2013.csv')

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
target_train = np.reshape(target_train, [target_train.size,1 ])

target_val = np.array(val['emotion'])
target_val = np.reshape(target_val, [target_val.size,1])

#---------------------------------MODEL TRAINING--------------------------------------

def loss_fn(out, target):
    
    #target = torch.LongTensor(target)
    #target = Variable(target)

    x_1 = F.softmax(out[0], 1)
    x_2 = F.softmax(out[1], 1)
    x_3 = F.softmax(out[2], 1)
    x_4 = F.softmax(out[3], 1)
    x_5 = F.softmax(out[4], 1)
    x_6 = F.softmax(out[5], 1)
    
    x_1 = torch.gather(x_1, 1, target)
    x_2 = torch.gather(x_2, 1, target)
    x_3 = torch.gather(x_3, 1, target)
    x_4 = torch.gather(x_4, 1, target)
    x_5 = torch.gather(x_5, 1, target)
    x_6 = torch.gather(x_6, 1, target)

    weights = out[6]
    
    summ = x_1*weights[0] + x_2*weights[1] + x_3*weights[2] + x_4*weights[3] + x_5*weights[4] + x_6*weights[5]
    multi = torch.sum((weights**2))
    #print("{}, {}".format(-torch.sum(torch.log(summ)) ,multi))
    #print("{}".format(summ))
    lamda = 400
    loss = - torch.sum(torch.log(summ)) + lamda*multi
    if torch.isnan(loss) == 1:
        print(summ)

    return loss


def accuracy(out,target):
    
    target = target.view(target.size()[0] )
    
    x_1 = F.softmax(out[0], 1)
    x_2 = F.softmax(out[1], 1)
    x_3 = F.softmax(out[2], 1)
    x_4 = F.softmax(out[3], 1)
    x_5 = F.softmax(out[4], 1)
    x_6 = F.softmax(out[5], 1)
    
    weights = out[6]
    
    result = x_1*weights[0] + x_2*weights[1] + x_3*weights[2] + x_4*weights[3] + x_5*weights[4] + x_6*weights[5]
    result = F.softmax(result, 1)
    
    acc = (target == result.argmax(-1)).float().cpu().numpy()
    #acc = float(100 * acc.sum() /len(acc))
    return acc

#device = torch.device("cuda")
net = Net()
net = net.cuda()
print(next(net.parameters()).is_cuda)
#net.to(device)
#optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 100
batch_size = 128
loss_history = []
loss_history_val = []
acc_history_val = []
acc_history_train = []

x_train = torch.Tensor(train_img)
Y_train = torch.LongTensor(target_train)

x_train = x_train.cuda()
Y_train = Y_train.cuda()

lr = 0.01
for e in range(epochs):
    
    if e>1 and e<=4:
        lr=0.007
    if e>=5 and e<=8:
        lr = 0.005
    if e>=9 and e<=15:
        lr = 0.001
    if e>=16 and e<=22:
        lr = 0.0005
    if e>=23 and e<=29:
        lr = 0.0001
    if e>=30:
        lr = 0.00005

    optimizer = torch.optim.Adagrad(net.parameters(), lr)
    net.train()
    for i in range(0, train_img.shape[0]-37, batch_size):

       # x_mini = train_img[i: i+batch_size]
       # y_mini = target_train[i: i+batch_size]
       # y_mini = np.reshape(y_mini, [y_mini.size,1])
        
       # x_mini = torch.Tensor(x_mini)
       # y_mini = torch.LongTensor(y_mini)
        
        
        x_var = Variable(x_train[i: i+batch_size], requires_grad = True)
        #y_var = Variable(y_mini)
        #x_var, y_mini = x_var.to(device), y_mini.to(device)
    
        optimizer.zero_grad()
        output = net(x_var)
        loss = loss_fn(output, Y_train[i: i+batch_size])
        loss.backward()
        optimizer.step()
        
        if i%4096 == 0:
            acc = accuracy(output, Y_train[i: i+batch_size])
            acc = float(100* acc.sum() / len(acc))
            print('Mini batch {}, loss: {}, accuracy: {}'.format(i, loss,  acc))

          
    net.eval()
    with torch.no_grad():
        bsize_val = 32
        bsize_train = 32
        x_validate = torch.Tensor(val_img)
        y_validate = torch.LongTensor(target_val)
        x_validate, y_validate = x_validate.cuda(), y_validate.cuda()

        acc_val_array = np.empty((0), np.float32)

        for j in range(0, val_img.shape[0], bsize_val):

            output_val = net(x_validate[j: j+bsize_val])
            #output_train = net(x_train)
            #loss_val = loss_fn(output_val, y_validate[j: j+bsize_val])
            acc_val = accuracy(output_val, y_validate[j: j+bsize_val])
            acc_val_array = np.append(acc_val_array, acc_val)

        acc_val = float(100*acc_val_array.sum() / len(acc_val_array))

        acc_train_array = np.empty((0), np.float32)
        for j in range(0, train_img.shape[0], bsize_train):
            acc_train = accuracy(net(x_train[j : j+bsize_train]), Y_train[j: j+bsize_train])
            acc_train_array = np.append(acc_train_array, acc_train)

        acc_train = float(100*acc_train_array.sum() / len(acc_train_array))
    
    
    if e>0 and bool(acc_val > max(acc_history_val)) :
        torch.save(net.state_dict(), 'my_model_2.pth')
    
    
    acc_history_train.append(acc_train)
    acc_history_val.append(acc_val)

    print("Epoch no {}:  Train Accuracy={}  Validation Accuracy={}".format(e, acc_train,  acc_val))
    if (e+1)%10 == 0 :
        plt.plot(range(e+1), acc_history_train)
        plt.plot(range(e+1), acc_history_val)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Accuracy')
        plt.legend(['train', 'validation'])
        plt.savefig('accu.png')



# trial = train_img[:2]
# tr = torch.Tensor(trial)

# target = target_train[4:6]
# target = np.reshape(target, [2,1])
# target = torch.LongTensor(target)


