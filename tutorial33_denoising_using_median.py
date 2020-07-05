#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Spyder Editor

cv2.medianBlur - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
skimage.filters.median - https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median

See how median is much better at cleaning salt and pepper noise compared to Gaussian
"""
import cv2
from skimage.filters import median


#Needs 8 bit, not float.
img_gaussian_noise = cv2.imread('images/Osteosarcoma_01_25Sigma_noise.tif', 0)
img_salt_pepper_noise = cv2.imread('images/Osteosarcoma_01_8bit_salt_pepper.tif', 0)

img = img_salt_pepper_noise


median_using_cv2 = cv2.medianBlur(img, 3)

from skimage.morphology import disk  
#Disk creates a circular structuring element, similar to a mask with specific radius
median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)


cv2.imshow("Original", img)
cv2.imshow("cv2 median", median_using_cv2)
cv2.imshow("Using skimage median", median_using_skimage)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

