# https://youtu.be/kus4kmDhfdM

"""
Balancing imbalanced data to improve the accuracy of minority classes.

Demonstrated via semantic segmentation using feature extraction and Random Forest


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
## STEP 1:   READ TRAINING IMAGES AND EXTRACT FEATURES 
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
# STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
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
 #  STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)
    # COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET
###############################################################
dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets

#If you expect image and mask names to be the same this is where we can perform sanity check
#dataset['Image_Name'].equals(dataset['Mask_Name'])   
##
##If we do not want to include pixels with value 0 
##e.g. Sometimes unlabeled pixels may be given a value 0.
dataset = dataset[dataset.Label_Value != 0]

#Assign training features to X and labels to Y
#Drop columns that are not relevant for training (non-features)
X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1) 

#Assign label values to Y (our prediction)
Y = dataset["Label_Value"].values 

#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

####################################################################
# STEP 4: Define the classifier and fit a model with our training data
###################################################################
#Random Forest
#Import training classifier
from sklearn.ensemble import RandomForestClassifier
## Instantiate model with n number of decision trees
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
start = datetime.now() 

RF_model.fit(X_train, y_train)
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


###############################################
#Weighted RF, by providing class weight.
#Change the weight that each class has when calculating the “impurity” score of a chosen split point.
#Impurity (e.g. Gini) measures how mixed the groups of samples are for a given split in the training dataset 
#The calculation can be biased so that a mixture in favor of the minority class is favored, 

#Can provide custom weights or use 'balanced'
# The “balanced” mode uses the values of y to automatically adjust 
#weights inversely proportional to class frequencies in the input data
#print(dataset["Label_Value"].value_counts())
(unique, counts) = np.unique(y_train, return_counts=True)
print(unique, counts)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
print("Class weights are...:", class_weights)

my_weights = {0:1, 1:2, 2:0.6, 3:1.5}

RF_model_penalized = RandomForestClassifier(n_estimators = 50, 
                               class_weight=my_weights, # custom weights
                               #class_weight='balanced', # balanced
                               random_state = 42)

start = datetime.now() 
# Train the model on training data
RF_model_penalized.fit(X_train, y_train)
stop = datetime.now()

#Execution time of the model 
execution_time_RF_balanced = stop-start 
print("Random Forest balanced execution time is: ", execution_time_RF_balanced)

prediction_RF_balanced = RF_model_penalized.predict(X_test)

#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of Random Forest balanced = ", metrics.accuracy_score(y_test, prediction_RF_balanced))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_RF_balanced = MeanIoU(num_classes=num_classes)  
IOU_RF_balanced.update_state(y_test, prediction_RF_balanced)
print("Mean IoU for Random Forest balanced = ", IOU_RF_balanced.result().numpy())


#To calculate I0U for each class...
values_balanced = np.array(IOU_RF_balanced.get_weights()).reshape(num_classes, num_classes)
print(values_balanced)
class1_IoU_balanced = values_balanced[0,0]/(values_balanced[0,0] + values_balanced[0,1] + values_balanced[0,2] + values_balanced[0,3] + values_balanced[1,0]+ values_balanced[2,0]+ values_balanced[3,0])
class2_IoU_balanced = values_balanced[1,1]/(values_balanced[1,1] + values_balanced[1,0] + values_balanced[1,0] + values_balanced[1,2] + values_balanced[0,1]+ values_balanced[2,1]+ values_balanced[3,1])
class3_IoU_balanced = values_balanced[2,2]/(values_balanced[2,2] + values_balanced[2,0] + values_balanced[2,1] + values_balanced[2,3] + values_balanced[0,2]+ values_balanced[1,2]+ values_balanced[3,2])
class4_IoU_balanced = values_balanced[3,3]/(values_balanced[3,3] + values_balanced[3,0] + values_balanced[3,1] + values_balanced[3,2] + values_balanced[0,3]+ values_balanced[1,3]+ values_balanced[2,3])

print("IoU for balanced class 1 is: ", class1_IoU_balanced)
print("IoU for balanced class 2 is: ", class2_IoU_balanced)
print("IoU for balanced class 3 is: ", class3_IoU_balanced)
print("IoU for balanced class 4 is: ", class4_IoU_balanced)

#Slight improvement but not much...

########################################################################

#Can we improve by upsampling minority data and downsampling majority ?

from sklearn.utils import resample
print(dataset["Label_Value"].unique())  #Look at the labels in our dataframe
print(dataset["Label_Value"].value_counts())

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

Y_resampled = dataset_resampled['Label_Value'].values

Y_resampled = LabelEncoder().fit_transform(Y_resampled)

##Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=20)

RF_model_resampled = RandomForestClassifier(n_estimators = 50, random_state = 42)

## Train the model on training data
start = datetime.now() 

RF_model_resampled.fit(X_train_resampled, y_train_resampled)
stop = datetime.now()

#Execution time of the model 
execution_time_RF_resampled = stop-start 
print("Random Forest resampled execution time is: ", execution_time_RF_resampled)

#Predict and test accuracy
prediction_RF_resampled = RF_model_resampled.predict(X_test_resampled)
#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of Random Forest resampled = ", metrics.accuracy_score(y_test_resampled, prediction_RF_resampled))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_RF_resampled = MeanIoU(num_classes=num_classes)  
IOU_RF_resampled.update_state(y_test_resampled, prediction_RF_resampled)
print("Mean IoU for Random Forest = ", IOU_RF_resampled.result().numpy())


#To calculate I0U for each class...
values_resampled = np.array(IOU_RF_resampled.get_weights()).reshape(num_classes, num_classes)
print(values_resampled)
class1_IoU_resampled = values_resampled[0,0]/(values_resampled[0,0] + values_resampled[0,1] + values_resampled[0,2] + values_resampled[0,3] + values_resampled[1,0]+ values_resampled[2,0]+ values_resampled[3,0])
class2_IoU_resampled = values_resampled[1,1]/(values_resampled[1,1] + values_resampled[1,0] + values_resampled[1,0] + values_resampled[1,2] + values_resampled[0,1]+ values_resampled[2,1]+ values_resampled[3,1])
class3_IoU_resampled = values_resampled[2,2]/(values_resampled[2,2] + values_resampled[2,0] + values_resampled[2,1] + values_resampled[2,3] + values_resampled[0,2]+ values_resampled[1,2]+ values_resampled[3,2])
class4_IoU_resampled = values_resampled[3,3]/(values_resampled[3,3] + values_resampled[3,0] + values_resampled[3,1] + values_resampled[3,2] + values_resampled[0,3]+ values_resampled[1,3]+ values_resampled[2,3])

print("IoU for class 1 resampled is: ", class1_IoU_resampled)
print("IoU for class 2 resampled is: ", class2_IoU_resampled)
print("IoU for class 3 resampled is: ", class3_IoU_resampled)
print("IoU for class 4 resampled is: ", class4_IoU_resampled)

#######################################################################################

#SMOTE

# Generate synthetic data (SMOTE and ADASYN)
# SMOTE: Synthetic Minority Oversampling Technique
#ADASYN: Adaptive Synthetic
# https://imbalanced-learn.org/stable/over_sampling.html?highlight=smote
# pip install imblearn
# SMOTE may not be the best choice all the time. It is one of many things
#that you need to explore. 

from imblearn.over_sampling import SMOTE, ADASYN

X_smote, Y_smote = SMOTE(random_state=42).fit_resample(X, Y)  #Beware, this takes some time based on the dataset size
#X_adasyn, Y_adasyn = ADASYN().fit_resample(X, Y)

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, 
                                                                            Y_smote, 
                                                                            test_size=0.2, 
                                                                            random_state=42)


(unique, counts) = np.unique(Y, return_counts=True)
print("Original data: ", unique, counts)
(unique2, counts2) = np.unique(Y_smote, return_counts=True)
print("After SMOTE: ", unique2, counts2)

#(unique3, counts3) = np.unique(Y_adasyn, return_counts=True)
#print("After ADASYN: ", unique3, counts3)

RF_model_SMOTE = RandomForestClassifier(n_estimators = 50, random_state = 42)

start = datetime.now() 
RF_model_SMOTE.fit(X_train_smote, y_train_smote)
stop = datetime.now()

#Execution time of the model 
execution_time_RF_SMOTE = stop-start 
print("Random Forest SMOTE execution time is: ", execution_time_RF_SMOTE)

prediction_RF_smote = RF_model_SMOTE.predict(X_test_smote)

print ("Accuracy of Random Forest with SMOTE = ", metrics.accuracy_score(y_test_smote, prediction_RF_smote))


#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_RF_SMOTE = MeanIoU(num_classes=num_classes)  
IOU_RF_SMOTE.update_state(y_test_smote, prediction_RF_smote)
print("Mean IoU for Random Forest with SMOTE = ", IOU_RF_SMOTE.result().numpy())


#To calculate I0U for each class...
values_SMOTE = np.array(IOU_RF_SMOTE.get_weights()).reshape(num_classes, num_classes)
print(values_SMOTE)
class1_IoU_SMOTE = values_SMOTE[0,0]/(values_SMOTE[0,0] + values_SMOTE[0,1] + values_SMOTE[0,2] + values_SMOTE[0,3] + values_SMOTE[1,0]+ values_SMOTE[2,0]+ values_SMOTE[3,0])
class2_IoU_SMOTE = values_SMOTE[1,1]/(values_SMOTE[1,1] + values_SMOTE[1,0] + values_SMOTE[1,0] + values_SMOTE[1,2] + values_SMOTE[0,1]+ values_SMOTE[2,1]+ values_SMOTE[3,1])
class3_IoU_SMOTE = values_SMOTE[2,2]/(values_SMOTE[2,2] + values_SMOTE[2,0] + values_SMOTE[2,1] + values_SMOTE[2,3] + values_SMOTE[0,2]+ values_SMOTE[1,2]+ values_SMOTE[3,2])
class4_IoU_SMOTE = values_SMOTE[3,3]/(values_SMOTE[3,3] + values_SMOTE[3,0] + values_SMOTE[3,1] + values_SMOTE[3,2] + values_SMOTE[0,3]+ values_SMOTE[1,3]+ values_SMOTE[2,3])

print("IoU for class 1 SMOTE is: ", class1_IoU_SMOTE)
print("IoU for class 2 SMOTE is: ", class2_IoU_SMOTE)
print("IoU for class 3 SMOTE is: ", class3_IoU_SMOTE)
print("IoU for class 4 SMOTE is: ", class4_IoU_SMOTE)

