#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
BRISQUE calculates the no-reference image quality score for an image using the 
Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE). 

BRISQUE score is computed using a support vector regression (SVR) model trained on an 
image database with corresponding differential mean opinion score (DMOS) values. 
The database contains images with known distortion such as compression artifacts, 
blurring, and noise, and it contains pristine versions of the distorted images. 
The image to be scored must have at least one of the distortions for which the model was trained.

Mittal, A., A. K. Moorthy, and A. C. Bovik. "No-Reference Image Quality Assessment in the Spatial Domain.
" IEEE Transactions on Image Processing. Vol. 21, Number 12, December 2012, pp. 4695–4708.
https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

To install imquality
https://pypi.org/project/image-quality/
"""
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque

#img = img_as_float(io.imread('noisy_images/BSE.jpg', as_gray=True))
img = img_as_float(io.imread('images/noisy_images/sandstone_25sigma_noisy.tif', as_gray=True))

score = brisque.score(img)
print("Brisque score = ", score)


#Now let us check BRISQUE scores for a few blurred images.

img0 = img_as_float(io.imread('images/blurred_images/sandstone.tif', as_gray=True))
img2 = img_as_float(io.imread('images/blurred_images/sandstone_2sigma_blur.tif', as_gray=True))
img3 = img_as_float(io.imread('images/blurred_images/sandstone_3sigma_blur.tif', as_gray=True))
img5 = img_as_float(io.imread('images/blurred_images/sandstone_5sigma_blur.tif', as_gray=True))


score0 = brisque.score(img0)
score2 = brisque.score(img2)
score3 = brisque.score(img3)
score5 = brisque.score(img5)

print("BRISQUE Score for 0 blur = ", score0)
print("BRISQUE Score for 2 sigma blur = ", score2)
print("BRISQUE Score for 3 sigma blur = ", score3)
print("BRISQUE Score for 5 sigma blur = ", score5)


# Peak signal to noise ratio (PSNR) is Not a good metric.

from skimage.metrics import peak_signal_noise_ratio

psnr_2 = peak_signal_noise_ratio(img0, img2)
psnr_3 = peak_signal_noise_ratio(img0, img3)
psnr_5 = peak_signal_noise_ratio(img0, img5)


print("PSNR for 1 sigma blur = ", psnr_2)
print("PSNR for 2 sigma blur = ", psnr_3)
print("PSNR for 3 sigma blur = ", psnr_5)

