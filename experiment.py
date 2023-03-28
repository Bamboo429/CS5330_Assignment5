#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:38:52 2022

@author: chuhsuanlin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:41:47 2022

@author: chuhsuanlin
"""


import torch
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import math
import numpy as np


        
# load MNIST dataset
def load_data(batch_size):
    
    
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainset,trainloader,testset,testloader


# set model parameter
class ModelParameter():
    def __init__(self, num_conv=1, kernel_size=5, dropout = 0.5):
        self.num_conv_layer = num_conv
        self.kernel_size = kernel_size
        self.dropout = dropout
        

# create model 
class NeuralNetwork(nn.Module):
    def __init__(self, parameter):
        
        super(NeuralNetwork, self).__init__()
        self.parameter = parameter
        padding = math.floor(self.parameter.kernel_size/2)
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size = self.parameter.kernel_size, padding = padding)
        self.conv3 = nn.Conv2d(10, 20, 5)
        self.dropout1 = nn.Dropout(self.parameter.dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        
        for i in range(self.parameter.num_conv_layer):
            x = self.conv2(x)
            
        x = self.conv3(x)
        x = self.dropout1(x)
        x = self.pool(F.relu((x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        
        return x
    

def train(net, train_loader, epoch, train_counter, train_loss, log_interval):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    #network.train()
    #for batch_idx, (data, target) in enumerate(trainloader):
    for batch_idx, data in enumerate(train_loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
                
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        if batch_idx % log_interval == 0:
            
        
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_loss.append(loss.item())
            train_counter.append((batch_idx*len(inputs)) + ((epoch-1)*len(train_loader.dataset)))
        
        # for test
        #if batch_idx == 2:
        #    break
        #torch.save(network.state_dict(), '/results/model.pth')
        #torch.save(optimizer.state_dict(), '/results/optimizer.pth')
      

def test(net, test_loader, test_loss, test_acc):
    
    #net.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            # calculate outputs by running images through the network
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()

        # cal average loss
        running_loss /= (len(test_loader.dataset)/len(inputs))
        test_loss.append(running_loss)    
        
        # cal accuracy
        acc = 100. * correct / len(test_loader.dataset)
        test_acc.append(acc)
        
        print(f'\nTest set: Avg. loss: {running_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    

def exp_kernel_size(trainloader, testloader):
       
    epochs = 5
    
    plt_legned = []
    all_test_acc = []
    
    # loop different kernel size 
    for size in range(3,7):
        
        print(f' ----- Train kernel_size = {size} ----- ')
        plt_legned.append(f'kernel size = {size}')
        
        # create model according to parameters
        parameter = ModelParameter(kernel_size=size)
        net = NeuralNetwork(parameter)
        # print model
        #summary(net, (1, 28, 28))
        
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]
        test_acc = []
    
        #train and test model
        test(net, testloader, test_loss, test_acc) 
        for epoch in range(1, epochs+1):
            train(net, trainloader, epoch, train_counter, train_loss, 10)
            test(net, testloader, test_loss, test_acc)
   
        # save accuracy
        all_test_acc.append(test_acc)
    

    
    all_test_acc = np.transpose(np.asarray(all_test_acc))
    
    # plot figure
    fig = plt.figure()
    plt.plot(test_counter, all_test_acc)
    plt.ylim([80, 100])
    plt.legend(plt_legned, loc='lower right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('accuarcy')
    fig
    
    return test_counter, all_test_acc

    
def exp_conv_number(trainloader, testloader):
    
    epochs = 5
    
    plt_legned = []
    all_test_acc = []
    
    # loop different number of conv layers
    for layer in range(1,6):
        
        print(f' ----- Train number of CONV layer = {layer} ----- ')
        plt_legned.append(f'CONV layer = {layer}')
        
        # create model according to parameters
        parameter = ModelParameter(num_conv=layer)
        net = NeuralNetwork(parameter)
        
        #print model
        #summary(net, (1, 28, 28))
        
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]
        test_acc = []
    
        #train and test model
        test(net, testloader, test_loss, test_acc) 
        for epoch in range(1, epochs+1):
            train(net, trainloader, epoch, train_counter, train_loss, 10)
            test(net, testloader, test_loss, test_acc)
   
        # save accuracy
        all_test_acc.append(test_acc)
        

    # plot
    all_test_acc = np.transpose(np.asarray(all_test_acc))
    
    fig = plt.figure()
    plt.plot(test_counter, all_test_acc)
    plt.ylim([80, 100])
    plt.legend(plt_legned, loc='lower right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('accuarcy')
    fig
    
    return test_counter, all_test_acc

def exp_dropout(trainloader, testloader):
    
    epochs = 5
    
    plt_legned = []
    all_test_acc = []
    
    # loop different dropout rate
    for dropout in np.arange(0.2,0.9,0.1):
        
        print(f' ----- Dropout rate = {dropout} ----- ')
        plt_legned.append(f'Dropout = {dropout:.2f}')
           
        # create model according to parameters               
        parameter = ModelParameter(dropout=dropout)
        net = NeuralNetwork(parameter)
        
        #print model
        #summary(net, (1, 28, 28))
        
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]
        test_acc = []
    
        #train and test model
        test(net, testloader, test_loss, test_acc) 
        for epoch in range(1, epochs+1):
            train(net, trainloader, epoch, train_counter, train_loss, 10)
            test(net, testloader, test_loss, test_acc)
   
        # save accuracy
        all_test_acc.append(test_acc)
        

    # plot
    all_test_acc = np.transpose(np.asarray(all_test_acc))
    
    fig = plt.figure()
    plt.plot(test_counter, all_test_acc)
    plt.ylim([80, 100])
    plt.legend(plt_legned, loc='lower right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('accuarcy')
    fig
    
    return test_counter, all_test_acc

def exp_batch_size():
    
    epochs = 5
    
    plt_legned = []
    all_test_acc = []
    
    # loop different batch size
    for size in range(100,600,100):
        
        
        batch_size = size 
        # set dataset and loader
        trainset, trainloader, testset, testloader = load_data(batch_size)
      
        print(f' ----- Batch size = {size} ----- ')
        plt_legned.append(f'batch size = {size}')
         
        # create model                  
        parameter = ModelParameter()
        net = NeuralNetwork(parameter)
        
        # print model
        #summary(net, (1, 28, 28))
        
        train_loss = []
        train_counter = []
        test_loss = []
        test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]
        test_acc = []
    
        #train and test model
        test(net, testloader, test_loss, test_acc) 
        for epoch in range(1, epochs+1):
            train(net, trainloader, epoch, train_counter, train_loss, 10)
            test(net, testloader, test_loss, test_acc)
   
        all_test_acc.append(test_acc)
        

    # plot
    all_test_acc = np.transpose(np.asarray(all_test_acc))
    
    fig = plt.figure()
    plt.plot(test_counter, all_test_acc)
    plt.ylim([80, 100])
    plt.legend(plt_legned, loc='lower right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('accuarcy')
    fig
    
    return test_counter, all_test_acc
    
    
def main():
    
    exp = 1
    # set parameter
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # load MNIST data
    batch_size = 200 
    trainset, trainloader, testset, testloader = load_data(batch_size)

    # run different experiment
    if exp == 1:
        test_counter1, all_test_acc1 = exp_kernel_size(trainloader, testloader)
    elif exp == 2:
        test_counter2, all_test_acc2 = exp_conv_number(trainloader, testloader)
    elif exp == 3:
        test_counter3, all_test_acc3 = exp_dropout(trainloader, testloader)
    elif exp == 4:
        test_counter4, all_test_acc4 = exp_batch_size()
      
    


if __name__ == "__main__":
    main()
    
    
      


