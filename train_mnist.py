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

def load_data(batch_size):
    
    # load MNIST dataset
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainset,trainloader,testset,testloader


def imshow(data):

    examples = enumerate(data)
    batch_idx, (example_data, example_targets) = next(examples)

         
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig


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
        
        
        #torch.save(network.state_dict(), '/results/model.pth')
        #torch.save(optimizer.state_dict(), '/results/optimizer.pth')
      

def test(net, test_loader, test_loss):
    
    #network.eval()
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
            #total += labels.size(0)
            correct += (predicted == labels).sum().item()

        running_loss /= (len(test_loader.dataset)/len(inputs))
        test_loss.append(running_loss)    
        #test_loss /= len(test_loader.dataset)
        #test_losses.append(test_loss)
        acc = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Avg. loss: {running_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
    

def main():
    
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    
    batch_size = 200
    epochs = 5
    
    trainset, trainloader, testset, testloader = load_data(batch_size)
    imshow(trainloader);
    
    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i*len(trainloader.dataset) for i in range(epochs + 1)]

    net = NeuralNetwork()
    summary(net, (1, 28, 28))
    
    test(net, testloader, test_loss) 
    for epoch in range(1, epochs+1):
        train(net, trainloader, epoch, train_counter, train_loss, 10)
        test(net, testloader, test_loss)
   
        
    fig = plt.figure()
    plt.plot(train_counter, train_loss, color='blue')
    plt.scatter(test_counter, test_loss, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    
    
    # save the model
    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)
    


if __name__ == "__main__":
    main()
    
    
      


