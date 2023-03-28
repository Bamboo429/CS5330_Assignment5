#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:32:41 2022

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
from train_mnist import NeuralNetwork

from torchsummary import summary
import cv2

def load_train_img():
    
    # load 1 train data for test
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    src_img = trainset[0][0]
      
    return src_img


def ana_1_layer(model):
    
    # load layer 1 parameters
    layer1_weight = model.conv1.weight

    # print the parameters
    print(model.conv1.weight)

    # plot the parameter
    fig = plt.figure()    
    for i in range(10):
        
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        
        tensor = layer1_weight[i,0]
        image = tensor.detach().numpy()
        plt.imshow(image)
    
        plt.title(f'Filter {i}')
        plt.xticks([])
        plt.yticks([])
    
    fig


def filter_1_layer(src_img, model ):
    
    # load layer 1 parameters
    layer1_weight = model.conv1.weight
    
    
    fig = plt.figure()
    for i in range(10):
        
        tensor = layer1_weight[i,0]
        weight = tensor.detach().numpy()
        
        # plot filter weight
        plt.subplot(5,4,2*i+1)
        plt.imshow(weight, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        
        
        # pass through filter (opencv filter2D function)
        src_img = (np.asarray(src_img) - 0.1307) / 0.3081
        filtered = cv2.filter2D(src = np.asarray(src_img), ddepth=-1, kernel = weight)
        
        # plot filtered image
        plt.subplot(5,4,2*i+2)
        plt.imshow(filtered, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    
    fig
   


# define truncated model
class Submodel(NeuralNetwork):

    # inherit original model layers
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = self.conv1(x)
        
        # second layer and
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.pool(F.relu((x)))
        '''
        return x
    
def data_preprocessing(img):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    
    img_tensor = transform(img)
    
    return img_tensor


def test_submodel(img_tensor, submodel):
    with torch.no_grad():

        outputs = submodel(img_tensor)
        
        out = outputs.detach().numpy()
        
        fig = plt.figure()
        for i in range(10):
            plt.subplot(5,2,i+1)
            plt.imshow(out[i], cmap='gray', interpolation='none')
            plt.xticks([])
            plt.yticks([])
            
        fig
      


def main():
    
    # load saved model
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./MNIST.pth'))
    model.eval()
    
    # load test data
    src_img = load_train_img()
    
    # analyze the first layer
    ana_1_layer(model)
    
    # filter image using opencv filter2D
    filter_1_layer(src_img, model)
    
    # call submodel
    submodel = Submodel();
    submodel.load_state_dict(torch.load('./MNIST.pth'))
    submodel.eval()

    # data preprocessing 
    img_tensor = data_preprocessing(src_img)
    
    # test submodel
    test_submodel(img_tensor, submodel)
    
        
    
if __name__ == "__main__":
    main()

  
  


  