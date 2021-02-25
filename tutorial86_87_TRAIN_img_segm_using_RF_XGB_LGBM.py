# https://youtu.be/RBKKKb1GSuA
# https://youtu.be/pbdseWOKG1U
"""
Feature based segmentation using Random Forest, XGBoost and LGBM

Random Forest from sklearn

XGBoost from...
pip install xgboost

LGBM from...
pip install lightgbm

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
# Define the classifier and fit a model with our training data
###################################################################
#Random Forest
##################################
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


###########################################################################
#Now fit a model using XGBoost
#XGBoost
############################################################
import xgboost as xgb
xgb_model = xgb.XGBClassifier()

start = datetime.now() 
# Train the model on training data
xgb_model.fit(X_train, y_train) 
stop = datetime.now()

#Execution time of the model 
execution_time_xgb = stop-start 
print("XGBoost execution time is: ", execution_time_xgb)

#Predict and test accuracy
prediction_xgb =xgb_model.predict(X_test)
#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of XGBoost = ", metrics.accuracy_score(y_test, prediction_xgb))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_xgb = MeanIoU(num_classes=num_classes)  
IOU_xgb.update_state(y_test, prediction_xgb)
print("Mean IoU for XGBoost = ", IOU_xgb.result().numpy())

####################################################################

#LGBM
##############################################################
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    #Try dart for better accuracy
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':10,
              'num_class':4}  #no.of unique values in the target class not inclusive of the end value

start=datetime.now()
lgb_model = lgb.train(lgbm_params, d_train, 50) #50 iterations. Increase iterations for small learning rates
stop=datetime.now()

execution_time_lgbm = stop-start
print("LGBM execution time is: ", execution_time_lgbm)

#Prediction on test data
prediction_lgb=lgb_model.predict(X_test)  #Yields probabilities for each class
#Convert to classification
prediction_lgb = np.array([np.argmax(i) for i in prediction_lgb])

#Pixel accuracy - not a good metric for semantic segmentation
#Print overall accuracy
print ("Accuracy of LGBM = ", metrics.accuracy_score(y_test, prediction_lgb))

#IOU for each class is..
# IOU = true_positive / (true_positive + false_positive + false_negative).

#Using built in keras function
from keras.metrics import MeanIoU
num_classes = 4
IOU_lgb = MeanIoU(num_classes=num_classes)  
IOU_lgb.update_state(y_test, prediction_lgb)
print("Mean IoU for LGBM = ", IOU_lgb.result().numpy())



###############################################################

#Summary
print("___________________________________________")
print("Random Forest execution time is: ", execution_time_RF)
print("XGBoost execution time is: ", execution_time_xgb)
print("LGBM execution time is: ", execution_time_lgbm)
print("___________________________________________")

print ("Accuracy of Random Forest = ", metrics.accuracy_score(y_test, prediction_RF))
print ("Accuracy of XGBoost = ", metrics.accuracy_score(y_test, prediction_xgb))
print ("Accuracy of LGBM = ", metrics.accuracy_score(y_test, prediction_lgb))
print("___________________________________________")

print("Mean IoU for Random Forest = ", IOU_RF.result().numpy())
print("Mean IoU for XGBoost = ", IOU_xgb.result().numpy())
print("Mean IoU for LGBM = ", IOU_lgb.result().numpy())


##########################################################
# SAVE MODEL FOR FUTURE USE
###########################################################
##You can store the model for future use. In fact, this is how you do machine elarning
##Train on training images, validate on test images and deploy the model on unknown images. 
#
#
##Save the trained model as pickle string to disk for future use
# model_name = "sandstone_model_multi_image_RF"
# pickle.dump(RF_model, open(model_name, 'wb'))
#
##To test the model on future datasets
#loaded_model = pickle.load(open(model_name, 'rb'))





