#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Color spaces
RGB - Red, Green, Blue 
HSV - Hue, Saturation, Value
LAB - Lightness, A (Green to red), B (Blue to Yellow)

"""

import cv2
from skimage import io


#Needs 8 bit, not float.
color_opencv = cv2.imread('images/Osteosarcoma_01.tif', 1)
gray_opencv = cv2.imread('images/Osteosarcoma_01.tif', 0)

color_skimage = io.imread('images/Osteosarcoma_01.tif', as_gray=False)
gray_skimage = io.imread('images/Osteosarcoma_01.tif', as_gray=True)

B, G, R = cv2.split(color_opencv)


cv2.imshow("Original", color_opencv)
cv2.imshow("B", B)
cv2.imshow("G", G)
cv2.imshow("R", R)


cv2.waitKey(0)          
cv2.destroyAllWindows() 


##########################################################


hsv_image = cv2.cvtColor(color_skimage, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

cv2.imshow("Original", color_opencv)
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)


cv2.waitKey(0)          
cv2.destroyAllWindows() 

#####################################


lab_image = cv2.cvtColor(color_skimage, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_image)

cv2.imshow("Original", color_opencv)
cv2.imshow("L", L)
cv2.imshow("A", A)
cv2.imshow("B", B)


cv2.waitKey(0)          
cv2.destroyAllWindows() 

############################################






