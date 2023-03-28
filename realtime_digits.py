#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:11:11 2022

@author: chuhsuanlin
"""

import cv2
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets,transforms



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



def main():
    
    # load trained model and parameter
    model = NeuralNetwork()
    model.load_state_dict(torch.load('./MNIST.pth'))
    model.eval()
        
    # open camera
    cap = cv2.VideoCapture(0)

    # for saving img result
    save=0
    
    while(True):
        
        # read frame form cap
        ret, frame = cap.read()
        
        # size of frame
        weight, height, a = frame.shape
        
        # cropt middle img to classify
        center_w = int(weight/2)
        center_h = int(height/2)
        h = 150
        crop_img = frame[center_w-h:center_w+h, center_h-h:center_h+h]
        
        # preprocessing
        img_tensor = data_preprocessing(crop_img)
        
        # match model input (usually have batch size)
        img_tensor= torch.unsqueeze(img_tensor, 0)
        
        # run the model and get the result
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # show on the video
        cv2.putText(crop_img, f'{predicted.item()}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', crop_img)
        
        key = cv2.waitKey(1)
        
        if key== ord('q'):
            break
        elif key == ord('s'):
            save += 1
            cv2.imwrite(f'{save}.png', crop_img)
          
    
    cap.release()
    cv2.destroyAllWindows()

    
    

if __name__ == "__main__":
    main()
