#https://youtu.be/xvmtpWGBLjI
"""
Speeding up ML training via feature selection using BORUTA

Feature based segmentation using Random Forest

"""


import numpy as np
import cv2
import pandas as pd
from datetime import datetime 
from sklearn import metrics

import pickle
from matplotlib import pyplot as plt
import os

####################################################################
## READ TRAINING IMAGES AND EXTRACT FEATURES 
################################################################
image_dataset = pd.DataFrame()  #Dataframe to capture image features

img_path = "images/sandstone/Train_images/"
for image in os.listdir(img_path):  #iterate through each file 
    print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    #Reset dataframe to blank after each loop.
    
    input_img = cv2.imread(img_path + image)  #Read images
    
    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

################################################################
#START ADDING DATA TO THE DATAFRAME
        
        
    #Add pixel values to the data frame
    pixel_values = img.reshape(-1)
    df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
    df['Image_Name'] = image   #Capture image name as we read multiple images
    
    
############################################################################    
        #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                
                    
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
        
########################################
#Gerate OTHER FEATURES and add them to the data frame
                
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
#    variance_img = nd.generic_filter(img, np.var, size=3)
#    variance_img1 = variance_img.reshape(-1)
#    df['Variance s3'] = variance_img1  #Add column to original dataframe


######################################                    
#Update dataframe for images to include details for each image in the loop
    image_dataset = image_dataset.append(df)

###########################################################
# READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
    # WITH LABEL VALUES AND LABEL FILE NAMES
##########################################################
mask_dataset = pd.DataFrame()  #Create dataframe to capture mask info.

mask_path = "images/sandstone/Train_mask_APEER/"    
for mask in os.listdir(mask_path):  #iterate through each file to perform some action
    print(mask)
    
    df2 = pd.DataFrame()  #Temporary dataframe to capture info for each mask in the loop
    input_mask = cv2.imread(mask_path + mask)
    
    #Check if the input mask is RGB or grey and convert to grey if RGB
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    #Add pixel values to the data frame
    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask
    
    mask_dataset = mask_dataset.append(df2)  #Update mask dataframe with all the info from each mask

################################################################
 #  GET DATA READY FOR RANDOM FOREST (or other classifier)
    # COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET
###############################################################
dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets

#If you expect image and mask names to be the same this is where we can perform sanity check
#dataset['Image_Name'].equals(dataset['Mask_Name'])   
##
##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset.Label_Value != 0]

print(dataset["Label_Value"].unique())  #Look at the labels in our dataframe
print(dataset["Label_Value"].value_counts())
##########################################################
#Let us first balance dataset so when we randomply split data we get equal representation from all classes

from sklearn.utils import resample

#Separate majority and minority classes
dataset1 = dataset[dataset["Label_Value"] == 1]
dataset2 = dataset[dataset["Label_Value"] == 2]
dataset3 = dataset[dataset["Label_Value"] == 3]
dataset4 = dataset[dataset["Label_Value"] == 4]

# Upsample minority class and other classes separately
# If not, random samples from combined classes will be duplicated and we run into
#same issue as before, undersampled remians undersampled.
df1 = resample(dataset1, 
                                 replace=True,     # sample with replacement
                                 n_samples=50000,    # to match average class
                                 random_state=42) # reproducible results
 
df2 = resample(dataset2, 
                                 replace=True,     # sample with replacement
                                 n_samples=50000,    # to match average class
                                 random_state=42) # reproducible results

df3 = resample(dataset3, 
                                 replace=True,     # sample with replacement
                                 n_samples=50000,    # to match average class
                                 random_state=42) # reproducible results

df4 = resample(dataset4, 
                                 replace=True,     # sample with replacement
                                 n_samples=50000,    # to match average class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
dataset_resampled = pd.concat([df1, df2, df3, df4])
print(dataset_resampled['Label_Value'].value_counts())

X_resampled = dataset_resampled.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1) 
X_resampled_array = X_resampled.to_numpy()  #To work with Boruta... Can easily drop unnecessary columns

Y_resampled = dataset_resampled['Label_Value'].values

#Encode labels to 0, 1, 2, 3... so we can work with built-in libraries (e.g., calculate IoU for each class)
from sklearn.preprocessing import LabelEncoder
Y_resampled = LabelEncoder().fit_transform(Y_resampled)


##############################################
##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled_array, Y_resampled, test_size=0.2, random_state=20)

#####################################################
# Normal Random Forest fit wihout any feature selection
#####################################################
#Random Forest
#Import training classifier
from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
start = datetime.now() 
RF_model.fit(X_train, y_train)  #Fit model using all our data, all features. 
stop = datetime.now()

#Execution time of the model 
execution_time_RF = stop-start 
print("Random Forest execution time is: ", execution_time_RF)

#Predict and test accuracy

prediction_RF = RF_model.predict(X_test)
#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of Random Forest = ", metrics.accuracy_score(y_test, prediction_RF))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_RF = MeanIoU(num_classes=num_classes)  
IOU_RF.update_state(y_test, prediction_RF)
print("Mean IoU for Random Forest = ", IOU_RF.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_RF.get_weights()).reshape(num_classes, num_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,0] + values[1,2] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class 1 is: ", class1_IoU)
print("IoU for class 2 is: ", class2_IoU)
print("IoU for class 3 is: ", class3_IoU)
print("IoU for class 4 is: ", class4_IoU)


############################################################
# Can we speed up the training by not losing quality of results?
############################################

#Take only 5% data for Boruta feature selection.
#Otherwise the process will be very slow.
X_train_boruta, _, y_train_boruta, _ = train_test_split(X_train, y_train, test_size=0.95, random_state=20)

#############################################

# Define XGBOOST classifier to be used by Boruta
import xgboost as xgb
model = xgb.XGBClassifier()  #For Boruta

"""
Create shadow features – random features and shuffle values in columns
Train Random Forest / XGBoost and calculate feature importance via mean decrease impurity
Check if real features have higher importance compared to shadow features 
Repeat this for every iteration
If original feature performed better, then mark it as important 
"""

from boruta import BorutaPy

# define Boruta feature selection method. Experiment with n_estimators and max_iter
feat_selector = BorutaPy(model, n_estimators=200, verbose=2, random_state=42, max_iter=20)

# find all relevant features
feat_selector.fit(X_train_boruta, y_train_boruta)

# check selected features
print(feat_selector.support_)  #Should we accept the feature

# check ranking of features
print(feat_selector.ranking_) #Rank 1 is the best



"""
Review the features
"""
feature_names = np.array(X_resampled.columns)  #Convert dtype string?

# zip feature names, ranks, and decisions 
feature_ranks = list(zip(feature_names, 
                         feat_selector.ranking_, 
                         feat_selector.support_))

# print the results
for feat in feature_ranks:
    print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
    

# call transform() on X to filter it down to selected features
X_train_filtered = feat_selector.transform(X_train)  #Apply feature selection and return transformed data
X_test_filtered = feat_selector.transform(X_test)  #Apply feature selection and return transformed data

###################################################################
# Now fit data with subset of features using the same Random Forest
#Let us use the Random Forest classifier as before
#################################################################
#Import training classifier
from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
RF_model_BORUTA = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
start = datetime.now() 

RF_model_BORUTA.fit(X_train_filtered, y_train)
stop = datetime.now()

#Execution time of the model 
execution_time_RF_BORUTA = stop-start 
print("Random Forest execution time with BORUTA is: ", execution_time_RF_BORUTA)

#Predict and test accuracy

prediction_RF_BORUTA = RF_model_BORUTA.predict(X_test_filtered)
#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of Random Forest with BORUTA = ", metrics.accuracy_score(y_test, prediction_RF_BORUTA))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_RF_BORUTA = MeanIoU(num_classes=num_classes)  
IOU_RF_BORUTA.update_state(y_test, prediction_RF_BORUTA)
print("Mean IoU for Random Forest with BORUTA = ", IOU_RF_BORUTA.result().numpy())


#To calculate I0U for each class...
values_BORUTA = np.array(IOU_RF_BORUTA.get_weights()).reshape(num_classes, num_classes)
print(values_BORUTA)
class1_IoU_BORUTA = values_BORUTA[0,0]/(values_BORUTA[0,0] + values_BORUTA[0,1] + values_BORUTA[0,2] + values_BORUTA[0,3] + values_BORUTA[1,0]+ values_BORUTA[2,0]+ values_BORUTA[3,0])
class2_IoU_BORUTA = values_BORUTA[1,1]/(values_BORUTA[1,1] + values_BORUTA[1,0] + values_BORUTA[1,0] + values_BORUTA[1,2] + values_BORUTA[0,1]+ values_BORUTA[2,1]+ values_BORUTA[3,1])
class3_IoU_BORUTA = values_BORUTA[2,2]/(values_BORUTA[2,2] + values_BORUTA[2,0] + values_BORUTA[2,1] + values_BORUTA[2,3] + values_BORUTA[0,2]+ values_BORUTA[1,2]+ values_BORUTA[3,2])
class4_IoU_BORUTA = values_BORUTA[3,3]/(values_BORUTA[3,3] + values_BORUTA[3,0] + values_BORUTA[3,1] + values_BORUTA[3,2] + values_BORUTA[0,3]+ values_BORUTA[1,3]+ values_BORUTA[2,3])

print("IoU for class 1 with BORUTA is: ", class1_IoU_BORUTA)
print("IoU for class 2 with BORUTA is: ", class2_IoU_BORUTA)
print("IoU for class 3 with BORUTA is: ", class3_IoU_BORUTA)
print("IoU for class 4 with BORUTA is: ", class4_IoU_BORUTA)


###########################################################################
#Summary 
print("   ")
print("##### SUMMARY #######")
print("Random Forest execution time is: ", execution_time_RF)
print("Random Forest execution time with BORUTA is: ", execution_time_RF_BORUTA)
print("___________________________________________________")
print ("Accuracy of Random Forest = ", metrics.accuracy_score(y_test, prediction_RF))
print ("Accuracy of Random Forest with BORUTA = ", metrics.accuracy_score(y_test, prediction_RF_BORUTA))
print("___________________________________________________")
print("Mean IoU for Random Forest = ", IOU_RF.result().numpy())
print("Mean IoU for Random Forest with BORUTA = ", IOU_RF_BORUTA.result().numpy())
print("___________________________________________________")
print("######## IOU for each class ###################")
print("    ")
print("IoU for class 1 is: ", class1_IoU)
print("IoU for class 2 is: ", class2_IoU)
print("IoU for class 3 is: ", class3_IoU)
print("IoU for class 4 is: ", class4_IoU)
print("___________________________________________________")
print("IoU for class 1 with BORUTA is: ", class1_IoU_BORUTA)
print("IoU for class 2 with BORUTA is: ", class2_IoU_BORUTA)
print("IoU for class 3 with BORUTA is: ", class3_IoU_BORUTA)
print("IoU for class 4 with BORUTA is: ", class4_IoU_BORUTA)
