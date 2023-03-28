#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:17:44 2022

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
import cv2

def load_data():

    # load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=0)
    
    return testset, testloader
    

def mnist_test(model, testloader):
     
    # load test data
    examples = enumerate(testloader)
    batch_idx, (inputs, labels) = next(examples)
    
    
    # test model
    with torch.no_grad():
        
        # calculate outputs through model
        outputs = model(inputs)
        torch.set_printoptions(precision=2)
        
        # highest energy as predict output
        _, max_indexs = torch.max(outputs, 1)
        
        #print(outputs)
        #print(max_indexs)
        
    
    
    fig = plt.figure()
    for i in range(10):
        
        # print parameter and output labels
        print(f'output value:\n {outputs[i]}')
        print(f'maxinum index: {max_indexs[i].item()} ') 
        print(f'correct label: {labels[i]} \n')
        
        # plot prediction value and imge
        if (i<9):
            plt.subplot(3,3,i+1)
            plt.tight_layout()
            plt.imshow(inputs[i][0], cmap='gray', interpolation='none')
            plt.title(f'Prediction: {labels[i]}')
            plt.xticks([])
            plt.yticks([])
     
    fig
          
  
# preprocessing for customer test images    
def data_preprocessing(img):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # resize, to gray image, and invert image
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray)
    
    # transform to pytorch tensor and normalize
    img_tensor = transform(inverted_image)
    
    return img_tensor


def custom_test(model, test_folder):
  
    # img folder
    digits_folder = test_folder
    
    i=0
    fig = plt.figure()
    # load image from test folder
    for x in os.listdir(digits_folder):
        if x.endswith(".png"): # load only .png file
                   
            # read image
            img_name = np.asarray(x)
            img = cv2.imread(digits_folder+x)
            
            # preprocessing
            img_tensor = data_preprocessing(img)
             
            # show image
            '''
            cv2.imshow(x, inverted_image)
            cv2.waitKey(0) 
            cv2.destroyAllWindows() 
            '''
            
            # match model input (usually have batch size)
            img_tensor= torch.unsqueeze(img_tensor, 0)
            
            with torch.no_grad():
                ## calculate outputs through model
                outputs = model(img_tensor)
                max_index = torch.argmax(outputs)
                
              
            # plot results    
            plt.subplot(3,4,i+1)
            plt.tight_layout()
            plt.imshow(img, cmap='gray', interpolation='none')
            plt.title(f'Prediction: {max_index}')
            plt.xticks([])
            plt.yticks([])
            
            i+=1
            print(f'Ground Turth:{x},  Prediction:{max_index} ')
        
    fig

def main():
    
    # load trained model and parameter
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./MNIST.pth'))
    model.eval()
    
    # load and test mnist test data
    testset, testloader = load_data()
    mnist_test(model, testloader)
    
    # test custom test data
    test_folder = './data/digits/'
    custom_test(model, test_folder)
    
    
if __name__ == "__main__":
    main()
    
    