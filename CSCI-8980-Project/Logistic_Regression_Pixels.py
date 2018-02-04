# # Notes on Pixel Level Dataset Version 1

# importing modules
from scipy.io import matlab
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc



#loading data from the server. This will download the dataset in your curernt directory and then load it.
url = 'http://umnlcc.cs.umn.edu/WaterDatasets/PixelLevelDataset_Version2.mat'
urllib.request.urlretrieve(url,'PixelLevelDataset_Version2.mat')
data = matlab.loadmat('PixelLevelDataset_Version2.mat')

#Remove unecessary columns in dataset
for key in ['__header__', '__version__', '__globals__']:
    if key in data:
        del data[key]

#Standardize X variables
X = pd.DataFrame.from_dict(preprocessing.scale(data['X_all']))
X.columns = ['X1','X2','X3','X4','X5','X6','X7']

#Extract Labels
Y = pd.DataFrame.from_dict(data['Y_all'])
Y.columns = ['Label']

#Transform Y into enoded
label_encoder= preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)

#PCA to see explained variance of component
pca = PCA()
pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumalative_explained_variance = np.cumsum(explained_variance_ratio)

#Plot PCA componenents
with plt.style.context('seaborn-whitegrid'):
    plt.bar(range(7), explained_variance_ratio, align = 'center', alpha = 0.5, label = 'explained variance bar plot')
    plt.step(range(7), cumalative_explained_variance, where = 'mid', label = 'cumalative explained variance')
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Component')
    plt.legend(loc = 'best')

#Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state = 0)

#Simple Logistic Regression
model_LR = LogisticRegression(random_state=0)
model_LR.fit(X_train, np.ravel(y_train))
y_prob = model_LR.predict_proba(X_test)[:,1]
y_pred = model_LR.predict(X_test)


print(metrics.confusion_matrix(y_test,y_pred))
print ( model_LR.score(X_test, y_test))

auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

