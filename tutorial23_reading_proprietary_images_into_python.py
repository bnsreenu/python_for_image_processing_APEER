#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

#Reading standard image formats is easy, especially JPG, PNG and normal Tif files. 
from skimage import io
img = io.imread("images/Osteosarcoma_01.tif", as_gray=True)

import cv2
grey_img = cv2.imread("images/Osteosarcoma_01.tif", 0)
color_img = cv2.imread("images/Osteosarcoma_01.tif", 1)


"""
Tiff files.... especially multidimensional tiff files. 

Use tifffile
pip install tifffile
"""

import tifffile
#RGB images
img = tifffile.imread("images/Osteosarcoma_01.tif")

import numpy as np
print(np.shape(img))
#3D image
img1 = tifffile.imread("images/3d_nuclei_image_8bit.tif")

#Time series images
img2 = tifffile.imread("images/Scratch_Assay_400_289_8bit.tif")



"""
####################################################################################
#reading czi files
# pip install czifile 
# to import the package you need to use import czifile
# https://pypi.org/project/czifile/
"""

################################


import czifile

img = czifile.imread('images/Osteosarcoma_01.czi')
#img = czifile.imread('images/Scratch_Assay_400_289.czi')


print(img.shape)  #7 dimensions
#Time series, scenes, channels, y, x, z, RGB
#IN this example (Osteosarcoma) we have 1 time series, 1 scene, 3 channels and each channel grey image
#size 1376 x 1104 

#Let us extract only relevant pixels, all channels in x and y
img1=img[0, 0, :, :, :, 0]
print(img1.shape)

#Next, let us extract each channel image.
img2=img1[0,:,:]  #First channel, Red
img3=img1[1,:,:] #Second channel, Green
img4=img1[2,:,:] #Third channel, Blue DAPI

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img2, cmap='hot')
ax1.title.set_text('1st channel')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img3, cmap='hot')
ax2.title.set_text('2nd channel')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img4, cmap='hot')
ax3.title.set_text('3rd channel')
plt.show()


#Olympus images, similar way https://pypi.org/project/oiffile/

###########Reading OME-TIFF using apeer_ometiff_library ###########
# pip install apeer-ometiff-library first 
# to import the package you need to use import apeer_ometiff_library
#OME-TIFF has tiff and metada (as XML) embedded
#Image is a 5D array.


from apeer_ometiff_library import io  #Use apeer.com free platform for image processing in the cloud

(pic2, omexml) = io.read_ometiff("images/Osteosarcoma_01_8bit.ome.tiff")  #Unwrap image and embedded xml metadata
print (pic2.shape)   #to verify the shape of the array
print(pic2)

print(omexml)

#Let us extract only relevant pixels, all channels in x and y
img1=img[0, 0, :, :, :, 0]
print(img1.shape)
#Next, let us extract each channel image.
img2=img1[0,:,:]  #First channel, Red
img3=img1[1,:,:] #Second channel, Green
img4=img1[2,:,:] #Third channel, Blue

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img2, cmap='hot')
ax1.title.set_text('1st channel')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img3, cmap='hot')
ax2.title.set_text('2nd channel')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img4, cmap='hot')
ax3.title.set_text('3rd channel')
plt.show()



##################################
