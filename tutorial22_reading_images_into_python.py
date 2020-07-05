#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Reading images into Python.

Many ways but the recommended ways are using skimage or opencv.
We read images to python primarily for processing. 
Processing on multidimensional data is easy with numpy arrays.
skimage and opencv directly store imported images as numpy arrays.

Pillow library is fine but it does not convert to numpy array by default.
Need to convert as a separate step... np.asarray(img)

"""

# to install scikit-image, pip install scikit-image 
# to import the package you need to use import skimage

from skimage import io

img = io.imread("images/Osteosarcoma_01.tif")
print(img.shape)  #y,x,c

#x = Width = 1376
#y = Height = 1104
#Channels = 3 (RGB)

#Some image processing tasks in skimage require floating point image
#with values between 0 and 1

from skimage import img_as_float
img2 = img_as_float(img)

import numpy as np
img3 = img.astype(np.float)
#avoid using astype as it violates assumptions about dtype range.
#for example float should range from 0 to 1 (or -1 to 1) but if you use 
#astype to convert to float, the values do not lie between 0 and 1. 

#Convert back to 8 bit
from skimage import img_as_ubyte
img_8bit = img_as_ubyte(img2)

####################################################
#OPENCV

#################################################################################
######### Using openCV #########

"""
#to install open CV : pip install opencv-python
#to import the package you need to use import cv2
#openCV is a library of programming functions mainly aimed at computer vision.
#Very good for images and videos, especially real time videos.
#It is used extensively for facial recognition, object recognition, motion tracking,
#optical character recognition, segmentation, and even for artificial neural netwroks. 

You can import images in color, grey scale or unchanged usingindividual commands 
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

"""

import cv2

grey_img = cv2.imread("images/Osteosarcoma_01.tif", 0)
color_img = cv2.imread("images/Osteosarcoma_01.tif", 1)

#images opened using cv2 are numpy arrays
print(type(grey_img)) 
print(type(color_img)) 

#Big difference between skimage imread and opencv is that 
#opencv reads images as BGR instead of RGB.

img_opencv = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) #Should be same as skimage image







