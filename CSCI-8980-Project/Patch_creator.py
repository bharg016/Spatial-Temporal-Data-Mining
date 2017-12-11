#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:53:22 2017

@author: Akhil
"""
import math
import numpy as np
from sklearn import preprocessing

def Patch_creator(image, X_array, Y_array, horizontal, vertical):
    
    #Get Features (pixels) and labels for the image
    pixels = image['X']
    labels = image['Y']
    
    #Used to encode
    lb = preprocessing.LabelBinarizer()
    lb.fit([1,2])
    
    #check to see if patch size is bigger than image
    if labels.shape > (horizontal, vertical):
        
        #Find how many patches can be achieved vertically and horizontally
        Num_Horizontal = math.floor(labels.shape[1] / horizontal)
        Num_Vert = math.floor(labels.shape[0] / vertical)
        
        #initalize vertical position
        vert_pos1 = 0
        vert_pos2 = vertical -1
        
        #initalize horizontal position
        horz_pos1 = horizontal * -1
        horz_pos2 = -1
        
        for vertical_shift in range (Num_Vert):
            
            #\Get all horizontal shifts for each vertical shift first
            for horizontal_shift in range(Num_Horizontal):
                
                horz_pos1 = horz_pos1 + horizontal
                horz_pos2 = horz_pos2 + horizontal
                
                #slices image on dimensions to get patch
                X_patch = pixels[vert_pos1:vert_pos2, horz_pos1 :horz_pos2, :].reshape(1,vertical-1,horizontal-1,7)
                
                #Binzaration and Encode classes (3)
                Y_patch = labels[vert_pos1:vert_pos2, horz_pos1:horz_pos2].reshape((vertical -1) * (horizontal -1))
                Encoded_Y_patch = lb.transform(Y_patch)
                Encoded_Y_patch = Encoded_Y_patch.reshape(1,Encoded_Y_patch.shape[0], Encoded_Y_patch.shape[1])
                
                #If you have an empty array, create a new one,
                #Else keep adding patches and info
                if X_array == []:
                    X_array = np.array(X_patch)
                elif X_array != []:
                    X_array = np.vstack( (X_array,X_patch))
                
                if Y_array == []:
                    Y_array = np.array(Encoded_Y_patch)
                elif Y_array != []:
                    Y_array = np.vstack( (Y_array,Encoded_Y_patch))
            
            #Shift vertical Patch
            vert_pos1 = vert_pos1 + vertical
            vert_pos2 = vert_pos2 + vertical
            
            #reset horizontal position
            horz_pos1 = (horizontal * -1) 
            horz_pos2 = -1
        
    return (X_array, Y_array)
 