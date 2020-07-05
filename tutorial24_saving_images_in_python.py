#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Saving images to local drive 

"""
#Read an image
from skimage import io
img = io.imread("images/Osteosarcoma_01.tif")

#Do something to the image
#e.g. let us apply gaussian smoothing
from skimage import filters
gaussian_img = filters.gaussian(img, sigma=1)

#Save image using skimage
#Best way as it converts float images to RGB and scales them accordingly
io.imsave("saved_using_skimage.tif", gaussian_img)

#File autoamtically gets saved to right format based on the extension.
#We can define the exact library to be used to save files but defaults work ok.
#For tiff extensions it uses tifffile library to save images, in the background.
#First, image needs to be converted to 8 bit unsigned integer.  

##############################################################################
#save image using opencv
import cv2
cv2.imwrite("saved_using_opencv.jpg", gaussian_img)

#Will succeed writing an image but rounds off flaot
#final image may not look good if saving float 
#so first convert float to 8 bit
from skimage import img_as_ubyte
gaussian_img_8bit = img_as_ubyte(gaussian_img)
cv2.imwrite("saved_using_opencv2.jpg", gaussian_img_8bit)

#This saves fine and the image should be fine but ...
#The colors may be weird, if you are saving color images.
#This is because opencv uses BGR instead of RGB.
#If scolors are important then try working fully within opencv, 
#including reading and writing images.
#Or, convert images from BGR to RGB when necessary.

gaussian_img_8bit_RGB = cv2.cvtColor(gaussian_img_8bit, cv2.COLOR_BGR2RGB)
cv2.imwrite("saved_using_opencv3.jpg", gaussian_img_8bit_RGB)


#############################################################################
#Save using Matplotlib
from matplotlib import pyplot as plt
plt.imsave("saved_using_pyplot.jpg", gaussian_img)
#For gray images you can define a colormap using cmap

#########################################################################
#Saving images into tiff files..
#USe tifffile library: pip install tifffile
#First convert images to 8 bit and then use tifffile
import tifffile
tifffile.imwrite("saved_using_tifffile.tiff", gaussian_img_8bit)

#Can also use skimage but image needs to be converted to 8 bit integer first. 
io.imsave("saved_using_skimage.tif", gaussian_img_8bit)



