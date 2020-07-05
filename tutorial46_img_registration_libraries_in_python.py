#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Learning from the astronomy guys...
A couple of ways to perform image registration
https://image-registration.readthedocs.io/en/latest/image_registration.html

pip install image_registration
"""
from skimage import io
from image_registration import chi2_shift

image = io.imread("images/Osteosarcoma_01.tif", as_gray=True)
offset_image = io.imread("images/Osteosarcoma_01_transl.tif", as_gray=True)
# offset image translated by (-17, 18.) in y and x 



#Method 1: chi squared shift
#Find the offsets between image 1 and image 2 using the DFT upsampling method
# 2D rigid

noise=0.1
xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, noise, 
                                      return_error=True, upsample_factor='auto')

print("Offset image was translated by: 18, -17")
print("Pixels shifted by: ", xoff, yoff)

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()

###########################################################################
#Method 2: Cross correlation based shift
#Use cross-correlation and a 2nd order taylor expansion to measure the shift
#2D rigid

from skimage import io
from image_registration import cross_correlation_shifts

image = io.imread("images/Osteosarcoma_01.tif", as_gray=True)
offset_image = io.imread("images/Osteosarcoma_01_transl.tif", as_gray=True)
# offset image translated by (-17, 18) in y and x 


xoff, yoff = cross_correlation_shifts(image, offset_image)


print("Offset image was translated by: 18, -17")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()

###########################################################

#Method 3: Register translation from skimage.feature
#2D rigid, same as cross correlation. 
#Depreciated in the latest skimage (0.18.0). 
#Latest version use: skimage.registration.phase_cross_correlation

from skimage import io

image = io.imread("images/Osteosarcoma_01.tif", as_gray=True)
offset_image = io.imread("images/Osteosarcoma_01_transl.tif", as_gray=True)
# offset image translated by (-17, 18) in y and x 


from skimage.feature import register_translation
shifted, error, diffphase = register_translation(image, offset_image, 100)
xoff = -shifted[1]
yoff = -shifted[0]


print("Offset image was translated by: 18, -17")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()

############################################################
#Method 4: Optical flow based shift. Best for warped images. 
#takes two images and returns a vector field. 
#For every pixel in image 1 you get a vector showing where it moved to in image 2.

from skimage import io

image = io.imread("images/Osteosarcoma_01.tif", as_gray=True)
offset_image = io.imread("images/Osteosarcoma_01_transl.tif", as_gray=True)
# offset image translated by (-17, 18) in y and x 


from skimage import registration
flow = registration.optical_flow_tvl1(image, offset_image)

# display dense optical flow
flow_x = flow[1, :, :]  #Along width
flow_y = flow[0, :, :]  #Along height


#Example 1: Simple application by just taking mean of flow in x and y
#Let us find the mean of all pixels in x and y and shift image by that amount
#ideally, you need to move each pixel by the amount from flow
import numpy as np
xoff = np.mean(flow_x)
yoff = np.mean(flow_y)


print("Offset image was translated by: 18, -17")
print("Pixels shifted by: ", xoff, yoff)

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')


#Example 2: Applying flow vectors to each pixel 

height, width = image.shape

#Use meshgrid to Return coordinate matrices from coordinate vectors.
#Extract row and column coordinates to which flow vector values will be added.
row_coords, col_coords = np.meshgrid(np.arange(height), np.arange(width),
                                     indexing='ij')   #Matrix indexing

#For each pixel coordinate add respective flow vector to transform
from skimage.transform import warp
image1_warp = warp(offset_image, np.array([(row_coords + flow_y), (col_coords + flow_x)]),
                   mode='nearest')


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(16, 16))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Pixel Corrected')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(image1_warp, cmap='gray')
ax4.title.set_text('Flow Corrected')
plt.show()


##################################################
