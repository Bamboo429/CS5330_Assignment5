#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:54:31 2022

@author: chuhsuanlin
"""

import os
import pandas as pd
import cv2
import numpy as np


def pre_preprocessing(img):
    # data preprocessing
    
    # resize to (28,28) same as model input size
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # to binary image
    (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # inverted image
    inverted_image = cv2.bitwise_not(img)
       
    return inverted_image

     
def create_dataset(greek_folder, save_folder, train):
    
    # empty dataframe for saving 
    csv_value = pd.DataFrame()
    csv_label = pd.DataFrame()

    # Create a new directory when it does not exist 
    isExist = os.path.exists(save_folder)
    if not isExist:       
        os.makedirs(save_folder)
  
    # loop all image in the folder
    for x in os.listdir(greek_folder):
        if x.endswith(".png"):
            # Prints only png file present in tthe Folder
            # print(x)
            
            # read image
            img_name = np.asarray(x)
            
            # data preprocessing
            img = cv2.imread(greek_folder+x)
            processed_img = pre_preprocessing(img)
            
            # show image
            '''
            cv2.imshow('img',inverted_image)
            cv2.waitKey(0) 
            cv2.destroyAllWindows() 
            '''    
            # save image 
            cv2.imwrite(save_folder+x, processed_img)
            
            # reshape for saving
            img_value = processed_img.reshape(1,-1)
            xx = np.append(img_name,img_value).reshape(1,-1)
            
            # save image value to pd
            df = pd.DataFrame(xx)
            csv_value = pd.concat([csv_value,df],axis = 0)
            
            # create label value
            label_name = x.split('_')[0]
            if (label_name == 'gamma'):
                label = 2
            elif (label_name == 'beta'):
                label = 1
            elif(label_name == 'alpha'):
                label = 0
            elif(label_name == 'delta'):
                label = 3
            elif(label_name == 'epsilon'):
                label = 4
            elif(label_name == 'pi'):
                label = 5
                
            # save label to pd
            yy = pd.DataFrame(np.append(x,label).reshape(1,-1))
            csv_label = pd.concat([csv_label,yy], axis = 0)
            
     
    if (train):
        # save label and data to csv files
        csv_value.to_csv('greek_train_data.csv',index=False, header = False)
        csv_label.to_csv('greek_train_label.csv',index=False, header = False)
    else:
        # save label and data to csv files
        csv_value.to_csv('greek_test_data.csv',index=False, header = False)
        csv_label.to_csv('greek_test_label.csv',index=False, header = False)
        
   
def main():
    
    # setting the folder path
    save_train_folder = './data/greek_processed/'
    greek_train_folder = './data/greek2/'
    
    save_test_folder = './data/greek_processed/'
    greek_test_folder = './data/greek_test/'
    
    # create dataset
    create_dataset(greek_train_folder, save_train_folder, train = True)
    create_dataset(greek_test_folder, save_test_folder, train = False)
    
        
if __name__ == "__main__":
    main()
 
    
