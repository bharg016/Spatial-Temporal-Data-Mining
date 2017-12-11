# importing modules
from scipy.io import matlab
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
get_ipython().magic('matplotlib inline')
import scipy.stats
import tensorflow as ts
import keras
from keras.layers import Dense, Reshape, Flatten,Conv2D, MaxPooling2D,Conv2DTranspose, Activation, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn import preprocessing


#loading data from the server. This will download the dataset in your current directory and then load it.
dataset_name = 'ImageLevelDataset_Version2'
url = 'http://umnlcc.cs.umn.edu/WaterDatasets/' + dataset_name + '.zip'
urllib.request.urlretrieve(url,dataset_name + '.zip')
os.system('unzip ' + dataset_name + '.zip')
print('Dataset Loaded ...')

#Get image_names and id names
image_names = os.listdir(dataset_name)
ID_name = []
for image_file in image_names:
    ID_name.append(image_file.replace('data_','').replace('.mat',''))

# Visualize a lake image
ID = ID_name[811]
data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
X = data['X'].astype(np.float64)
min_val = 0
max_val = np.amax(X)
X[X[:, :, :] > max_val] = max_val
X[X[:, :, :] < min_val] = min_val

for b in range(X.shape[2]):
    X[:, :, b] = X[:, :, b] * 2.0 / (max_val - min_val)


plt.figure(figsize = (15,8))
plt.subplot(1,2,1)
plt.imshow(X[:,:,[0, 3, 2]])
plt.title('Color Composite Image')
plt.subplot(1,2,2)
plt.imshow(data['Y'])
plt.title('Class Label Map')
plt.show()


#Function to create model
def model_build(train_X):
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape= train_X.shape[1:]))
    model.add(BatchNormalization(axis = 1, momentum = 0.99, epsilon = .001))
    model.add(Conv2D(128, kernel_size = (2,2), activation = 'relu'))
    model.add(Conv2D(112, kernel_size = (2,2), activation = 'relu'))
    model.add(Conv2D(112, kernel_size = (3,3), activation = 'relu'))
    model.add(Conv2DTranspose(112,(4,4), activation ='relu'))
    model.add(Conv2DTranspose(64,(3,3), activation = 'relu'))
    model.add(Conv2DTranspose(3,(2,2), activation = 'relu'))
    model.add(Reshape(((train_X.shape[1] ** 2),3)))
    model.add(Activation('softmax'))
  

    
    
    model.compile(loss= 'categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    return(model)

#Function to get largest image

def smallest_image(IDs):
    smallest_shape = (10000,10000,7)
    for ID in ID_name:
        data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
        if data['X'].shape < smallest_shape:
            smallest_shape = data['X'].shape
            smallest_ID = ID
    return (smallest_ID)


#function to create patches
def Patch_creator(image, X_array, Y_array, horizontal, vertical):
    
    #Get Features (pixels) and labels for the image
    pixels = image['X'][:,:,(0,3,2)]
    labels = image['Y']
    
    #Used to encode
    lb = preprocessing.LabelBinarizer()
    lb.fit([1,2,3])
    
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
                X_patch = pixels[vert_pos1:vert_pos2, 
                                 horz_pos1 :horz_pos2, :].reshape(
                                 1,vertical-1,horizontal-1,pixels.shape[2])
                
                #Binzaration and Encode classes (3)
                Y_patch = labels[vert_pos1:vert_pos2, 
                                 horz_pos1:horz_pos2].reshape((vertical -1) * 
                                 (horizontal -1))
                Encoded_Y_patch = lb.transform(Y_patch)
                Encoded_Y_patch = Encoded_Y_patch.reshape(
                        1,Encoded_Y_patch.shape[0], Encoded_Y_patch.shape[1])
                
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


def Visualize_Patch(Patch_Index,test_Y, predicted_Y, horizontal, vertical):
    
    #Get the patches
    realpatch = test_Y[Patch_Index,:]
    predictedpatch = predicted_Y[Patch_Index,:]

    #Reshape the patches to have the max probability of the class in the original size
    predictedpatch = np.asarray(pd.DataFrame(predictedpatch).idxmax(1)).reshape(vertical-1,horizontal-1).astype(int)
    realpatch = np.asarray(pd.DataFrame(realpatch).idxmax(1) ).reshape(vertical-1,horizontal-1).astype(int)
    
    #plot
    plt.subplot(1,2,1)
    plt.imshow(predictedpatch)
    plt.title('Predicted Values')
    plt.subplot(1,2,2)
    plt.imshow(realpatch)
    plt.title('Real Values')

def get_Land_perc(test_Y):
    land_count = 0
    for patch_num in range(test_Y.shape[0]):
        land_count = land_count + sum(np.asarray(pd.DataFrame(test_Y[patch_num,:]).idxmax(1)) == 1)
    
    return(land_count / (test_Y.shape[0] * test_Y.shape[1]))

    
#Initalize and create Patches
train_X = train_Y = test_Y = test_X = valid_X = valid_Y = []
train_set = ID_name[0:700]
valid_set = ID_name[701:810]
test_set = ID_name[811:955]

vertical = 17
horizontal = 17

for ID in train_set:
    data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
    train_X,train_Y = Patch_creator(data, train_X, train_Y, vertical, horizontal )
    
for ID in valid_set:
    data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
    valid_X, valid_Y = Patch_creator(data, valid_X, valid_Y, vertical, horizontal)
    
for ID in test_set:
    data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
    test_X, test_Y = Patch_creator(data,test_X,test_Y,vertical, horizontal)
    

#Build Model
model = model_build(train_X)

#Train Model with Validation Set
model.fit(train_X, train_Y,
          batch_size= 16,
          epochs= 10,
          verbose=1,
          validation_data=(valid_X, valid_Y ))

#Evaluate Model
print(model.evaluate(test_X, test_Y))

predicted_values = model.predict(test_X)

Visualize_Patch(0,test_Y,predicted_values, vertical, horizontal)


    
land_percent = get_Land_perc(test_Y)
print(land_percent)