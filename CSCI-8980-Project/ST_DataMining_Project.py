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
from matplotlib import colors
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Dense, Reshape, Flatten,Conv2D, MaxPooling2D,Conv2DTranspose, Activation, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn import preprocessing
from keras.callbacks import History 


#Function to create Fully Convolutional Network
def FCN_build(train_X):
    
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
    model.add(Reshape(((train_X.shape[1] * train_X.shape[2]),3)))
    model.add(Activation('softmax') )
    

    model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    
    return(model)



def UNET_build(train_X):
    inputs = Input((train_X.shape[1], train_X.shape[2], train_X.shape[3]))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(112, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(112, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(112, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(112, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    reshape11 = Reshape(((train_X.shape[1] * train_X.shape[2]),3))(conv10)

    model = Model(inputs=[inputs], outputs=[reshape11])

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


    return model

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
    pixels = image['X']
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
                
                if sum(Encoded_Y_patch[0,:])[1] != Encoded_Y_patch.shape[1] and sum(Encoded_Y_patch[0,:])[0] != Encoded_Y_patch.shape[1] and sum(Encoded_Y_patch[0,:])[2] != Encoded_Y_patch.shape[1]:
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

#function to display results
def Visualize_Patch(Patch_Index,test_X, test_Y, predicted_Y, horizontal, vertical):
    
    #Get the patches
    realpatch = test_Y[Patch_Index,:]
    predictedpatch = predicted_Y[Patch_Index,:]
    X = test_X[Patch_Index,:].astype(np.float64)

    #Reshape the patches to have the max probability of the class in the original size
    predictedpatch = np.asarray(pd.DataFrame(predictedpatch).idxmax(1)).reshape(vertical-1,horizontal-1).astype(int)
    realpatch = np.asarray(pd.DataFrame(realpatch).idxmax(1) ).reshape(vertical-1,horizontal-1).astype(int)
    
    #Get X to 0 - 1 Range
    min_val = 0
    max_val = np.amax(X)
    X[X[:, :, :] > max_val] = max_val
    X[X[:, :, :] < min_val] = min_val

    for b in range(X.shape[2]):
        X[:, :, b] = X[:, :, b] * 2.0 / (max_val - min_val)
    
    #define color map
    cmap_dataset1 = colors.ListedColormap(['darkblue', 'green','white'])
    cmap_dataset2 = colors.ListedColormap(['darkblue','green'])

    #plot
    plt.subplot(1,3,1)
    plt.imshow(X[:,:,[0,3,2]],)
    plt.title('RGB Visualization')
    plt.subplot(1,3,2)
    plt.imshow(predictedpatch, cmap = cmap_dataset2)
    plt.title('Predicted Values')
    plt.subplot(1,3,3)
    plt.imshow(realpatch,cmap=cmap_dataset2)
    plt.title('Real Values')

def get_Land_perc(test_Y):
    land_count = 0
    for patch_num in range(test_Y.shape[0]):
        land_count = land_count + sum(np.asarray(pd.DataFrame(test_Y[patch_num,:]).idxmax(1)) == 1)
    
    return(land_count / (test_Y.shape[0] * test_Y.shape[1]))
    
# Visualize a lake image
def Visualize_Image(ID_index, ID_name):
    ID = ID_name[ID_index]
    data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
    X = data['X'].astype(np.float64)
    min_val = 0
    max_val = np.amax(X)
    X[X[:, :, :] > max_val] = max_val
    X[X[:, :, :] < min_val] = min_val

    for b in range(X.shape[2]):
        X[:, :, b] = X[:, :, b] * 2.0 / (max_val - min_val)

    plt.figure(figsize = (8,8))
    plt.subplot(1,2,1)
    plt.imshow(X[:,:,[0, 3, 2]],interpolation='nearest')
    plt.title('Color Composite Image')
    plt.subplot(1,2,2)
    plt.imshow(data['Y'],interpolation='nearest',cmap = colors.ListedColormap(['darkblue','green']))
    plt.title('Class Label Map')
    plt.show()
    
def metrics(predicted_labels, true_labels):
    #Intalize values
    #TP = correctly classify pixel as land
    #TN = correctly classify pixel as water
    #FP = incorrectly classify pixel as land
    #FN = incorrectly classify pixel as water
    
    FP = TP = TN = FN = 0
    
    #Create a dataframe for each patch
    for patch in range(predicted_labels.shape[0]):
        pred_df = pd.DataFrame(predicted_labels[patch,:]).idxmax(1)
        true_df = pd.DataFrame(true_labels[patch,:]).idxmax(1)
        
        #go through each row
        for row in range(pred_df.shape[0]):
            
            #If prediction is land
            if pred_df[row] == 1:
                if true_df[row] == 1:
                    TN += 1
                elif true_df[row] == 0:
                    FN += 1
                    
            #If prediction is water
            elif pred_df[row] == 0:
                if true_df[row] == 0:
                    TP += 1
                elif true_df[row] == 1:
                    FP += 1
    
    Precision = TP/ (TP + FP)
    Recall = TP/(TP + FN)
    F1_score = (2 * Recall * Precision) / (Recall + Precision)
    Accuracy = (TN + TP)/(TP + TN + FP + FN)
    return(F1_score,Accuracy,TP,FP,FN,TN)
    

if __name__ == '__main__':
    
    #loading data from the server. This will download the dataset in your current directory and then load it.
    dataset_name = 'ImageLevelDataset_Version2'
    #url = 'http://umnlcc.cs.umn.edu/WaterDatasets/' + dataset_name + '.zip'
    #urllib.request.urlretrieve(url,dataset_name + '.zip')
    os.system('unzip ' + dataset_name + '.zip')
    print('Dataset Loaded ...')

    #Get image_names and id names
    image_names = os.listdir(dataset_name)
    ID_name = []
    for image_file in image_names:
        ID_name.append(image_file.replace('data_','').replace('.mat',''))

    
    #Initalize and create Patches
    train_X = train_Y = test_Y = test_X = valid_X = valid_Y = []
    train_set = ID_name[0:700]
    valid_set = ID_name[701:810]
    test_set = ID_name[811:955]

    vertical = 33    
    horizontal = 33

    for ID in train_set:
        data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
        train_X,train_Y = Patch_creator(data, train_X, train_Y, vertical, horizontal )
    
    for ID in valid_set:
        data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
        valid_X, valid_Y = Patch_creator(data, valid_X, valid_Y, vertical, horizontal)
    
    for ID in test_set:
        data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
        test_X, test_Y = Patch_creator(data,test_X,test_Y,vertical, horizontal)
    

    #Create Models
    
    history = History()
    
   #UNET only works for size 16 * 16 images or higher
    if (vertical -1 ) % 16 == 0 and (horizontal -1)  % 16 == 0:
        
        #Build UNET
        UNET = UNET_build(train_X)
        UNET.summary()
        
        #Train UNET
        UNET.fit(train_X, train_Y,
          batch_size= 16,
          callbacks = [history],
          epochs= 1,
          verbose=1,
          validation_data=(valid_X, valid_Y ))
        
        #Predict Using UNET
        UNET_predicted_values = UNET.predict(test_X)
        
        #Get Metrics for UNET
        (F1,Accuracy, TP, FP, FN, TN) = metrics(UNET_predicted_values,test_Y)
        print("metrics for UNET:\nThe accuracy is %f \nThe F1 Score is %f \nThe number of pixels correctly classified as land is %d \nThe number of pixels correctly classified as water is %d \nThe number of pixels incorrectly classified as land is %d \nThe number of pixels incorrectly classified as water is %d"  % (Accuracy,F1, TN, TP, FN, FP) )
        
        #Visualize patch for UNET
        #Visualize_Patch(43,test_X,test_Y,UNET_predicted_values, vertical, horizontal)
    

    #Fit and Train FCN
    FCN = FCN_build(train_X)
    FCN.summary()    
    history = FCN.fit(train_X, train_Y,
          batch_size= 16,epochs= 30,
          verbose=1, callbacks = [history],
          validation_data=(valid_X, valid_Y ))

    #Generate FCN Predictions
    FCN_predicted_values = FCN.predict(test_X)
    
    #Get Metrics for FCN
    (F1, Accuracy, TP, FP, FN, TN) = metrics(FCN_predicted_values,test_Y)
    print("metrics for FCN:\nThe accuracy is %f \nThe F1 Score is %f \nThe number of pixels correctly classified as land is %d \nThe number of pixels correctly classified as water is %d \nThe number of pixels incorrectly classified as land is %d \nThe number of pixels incorrectly classified as water is %d"  % (Accuracy,F1, TN, TP, FN, FP) )  
   
    #Visualize Patch
    Visualize_Patch(65,test_X,test_Y,FCN_predicted_values, vertical, horizontal)

    land_percent = get_Land_perc(test_Y)
    print("Land percent is: %f"  % land_percent )
    
    plt
    #Predict patches for missing values in ImageDataSet1
    
    dataset_name = 'ImageLevelDataset_Version1'
    #url = 'http://umnlcc.cs.umn.edu/WaterDatasets/' + dataset_name + '.zip'
    #urllib.request.urlretrieve(url,dataset_name + '.zip')
    os.system('unzip ' + dataset_name + '.zip')
    print('Generating Patches for ImageLevelDataset_Version1...')

    #Get image_names and id names
    image_names = os.listdir(dataset_name)
    ID_name = []
    Dataset1_X = Dataset1_Y = []
    for image_file in image_names:
        ID_name.append(image_file.replace('data_','').replace('.mat',''))
    
    for ID in ID_name[0:15]:
        data = matlab.loadmat(dataset_name + '/data_' + ID +'.mat')
        Dataset1_X,Dataset1_Y = Patch_creator(data, Dataset1_X, Dataset1_Y, vertical, horizontal )
    
    #Make predictions with FCN for ImageLevelDataset_1
    Image1_pred_values = FCN.predict(Dataset1_X)
    Visualize_Patch(200,Dataset1_X,Dataset1_Y,Image1_pred_values,vertical,horizontal)
        