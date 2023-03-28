#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:47:31 2022

@author: chuhsuanlin
"""

from torch.utils.data.dataset import Dataset
import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

from train_mnist import NeuralNetwork
import torch.nn.functional as F
import torch
from torchvision import datasets,transforms


class GreekDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        
        self.greek_label = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.transform = transform
        
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        if torch.is_tensor(index):
            index = index.tolist()

        # read img name and concate with folder
        img_name = os.path.join(self.root_dir, self.greek_label.iloc[index, 0])
        # read image
        img = cv2.imread(img_name)
        
        # data preprocessing
        img = cv2.resize(img, (28, 28), interpolation = None)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_not(blackAndWhiteImage)
        
        # get label value
        label = self.greek_label.iloc[index,1]        
        
        # transform and to tensor
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
        
    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.greek_label)
    

def load_greek_data(batch_size):
    
    # set folder path
    train_csv_file = './greek_train_label.csv'
    train_root_dir = './data/greek/'

    test_csv_file = './greek_test_label.csv'
    test_root_dir = './data/greek_test/'
    
    # set transform 
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = GreekDataset(csv_file=train_csv_file,
                           root_dir=train_root_dir, transform=transform)
    
    # create dataloader and dataset
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    test_dataset = GreekDataset(csv_file=test_csv_file,
                           root_dir=test_root_dir, transform=transform)
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    return train_dataset, train_loader, test_dataset, test_loader



# create truncated model
class Submodel(NeuralNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.pool(F.relu((x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        
        return x

# test function
def test(model, data_loader):
    with torch.no_grad():  
        for i, data in enumerate(data_loader):
                
            images, labels = data
            
            # put img in to model and get output(embedding)
            outputs = model(images)
    
    # print and check tensor size
    print (f'embedding space size: {outputs.size()}')
    return labels, outputs

# cal sum square difference        
def cal_distance(vector_idx, embedding_vector, labels, dic_greek):
     
     # read embedding value
     vector = embedding_vector[vector_idx]
     # get label for print
     vec_label = dic_greek[labels[vector_idx].item()]
     
     # cal sum square difference
     ssd = torch.sum(torch.square((vector - embedding_vector)),1)
     for i in range(len(embedding_vector)):
        print(f'SSD({vec_label}, {dic_greek[labels[i].item()]}): {ssd[i]:.2f}')

# KNN
def KNN(test_embedding, train_embedding, labels, K):
    
    # get how many classes
    uniqueClass = len(torch.unique(labels))
    # init output
    output = np.zeros(len(test_embedding))
    
    
    for i in range(len(test_embedding)):
        countClass = np.zeros(uniqueClass)
        
        embedding_vector = test_embedding[i]
        # cal sum square difference
        ssd = torch.sum(torch.square((embedding_vector - train_embedding)),1)
        # sort ssd difference
        index = torch.argsort(ssd)
        
        # find the smallest K difference and the correspond class +1
        for k in range(K):
            countClass[labels[index[k]].item()]+=1
        
        # biggest number of class is the final result(class)
        output[i] = np.argmax(countClass)
        
    return output

            
    
def main():
    
    # set dictionary for classification
    dic_greek = { 0 : 'alpha', 1 : 'beta', 2 : 'gamma', 3:'delta', 4:'epsilon', 5:'pi'}

    # model setting need to bigger than dataset
    batch_size = 100
    
    #load data
    train_dataset, train_loader, test_dataset, test_loader = load_greek_data(batch_size)
    
    # load model
    model = Submodel();
    model.load_state_dict(torch.load('./MNIST.pth'))
    model.eval()
    
    # get embedding vector of train data
    labels, train_embedding = test(model, train_loader)
        
    examples = enumerate(train_loader)
    batch_idx, (img, labels) = next(examples)
    
    # get one example from each label 
    alpha_index = (labels == 0).tolist().index(True)
    beta_index = (labels == 1).tolist().index(True)
    gamma_index = (labels == 2).tolist().index(True)

    # print the differnece between the selected example and others
    print('\ndistance - alpha(label 0)')
    cal_distance(alpha_index, train_embedding, labels, dic_greek)
    
    print('\ndistance - beta(label 1)')  
    cal_distance(beta_index, train_embedding, labels, dic_greek)
    
    print('\ndistance - gamma(label 2)')  
    cal_distance(gamma_index, train_embedding, labels, dic_greek)
      
    # get embedding vector of train data
    testlabel, test_embedding = test(model, test_loader)
    
    # KNN
    y = KNN(test_embedding, train_embedding, labels, 3)
    
    examples = enumerate(test_loader)
    batch_idx, (img, labels) = next(examples)
      
    # show the result 
    for i in range(len(test_embedding)):
        fig = plt.figure()
        plt.title(f'Prediction:{dic_greek[y[i]]}')
        plt.imshow(img[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        
    
    
if __name__ == "__main__":
    main()
 
    
    
            

            
            
            
            
            
            
            
            
