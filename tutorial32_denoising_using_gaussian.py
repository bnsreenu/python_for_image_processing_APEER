#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""

cv2.filter2D - https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d
cv2.GaussianBlur - https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
skimage.filters.gaussian - https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
"""
import cv2
from skimage import io, img_as_float
from skimage.filters import gaussian

img_gaussian_noise = img_as_float(io.imread('images/Osteosarcoma_01_25Sigma_noise.tif', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('images/Osteosarcoma_01_8bit_salt_pepper.tif', as_gray=True))

img = img_gaussian_noise

gaussian_using_cv2 = cv2.GaussianBlur(img, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)
#sigma defines the std dev of the gaussian kernel. SLightly different than 
#how we define in cv2


cv2.imshow("Original", img)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
cv2.imshow("Using skimage", gaussian_using_skimage)
#cv2.imshow("Using scipy2", conv_using_scipy2)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

