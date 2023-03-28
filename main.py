#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:57:30 2022

@author: chuhsuanlin
"""

import os
import numpy as np
import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Make network code repeatable
torch.manual_seed(42)
torch.backends.cudnn.enabled = False

def load_data(batch_size):
    
    # load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor()])
    
    #batch_size = 4
    #mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainset,trainloader,testset,testloader



def imshow(trainset):
    

    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      npimg = trainset[i][0].numpy()
      npimg = np.squeeze(npimg)
      plt.imshow(npimg, cmap='gray', interpolation='none')
      plt.xticks([])
      plt.yticks([])
      



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size =5)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.pool(F.relu((x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        
        return x
    

   
def train_network(net, trainloader, testloader, epochs): 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_loss = []
    test_loss = []
    for e in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics       
            running_loss += loss.item()
            train_loss.append(running_loss)
            running_loss = 0
            
            
            if i % 1000 == 0:    # print every 1000 batches
                print(f' training  epoch: {e + 1}, batch:{i + 1:5d}')
               
            
        total = 0
        correct = 0    
        
        for i, data in enumerate(testloader):
            
            images, labels = data
            
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        test_loss.append(running_loss/i)
      
    return train_loss, test_loss


def main():
    
    torch.manual_seed(42)
    
    batch_size = 200
    epochs = 5
    
    trainset, trainloader, testset, testloader = load_data(batch_size)
    imshow(trainset);
    
    net = NeuralNetwork()
    train_loss, test_loss = train_network(net, trainloader, testloader, epochs)
    
    data_size = len(trainset)
    x_train = range(0,data_size*epochs,batch_size)  
    x_test = range(data_size,data_size*(epochs+1),data_size) 
    
    plt.figure()
    plt.plot(x_train,train_loss)
    plt.plot(x_test,test_loss,'ro')
    
    # save the model
    PATH = './MNIST.pth'
    #torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    main()

 
