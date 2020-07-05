#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""

Canny:
The Process of Canny edge detection algorithm can be broken down to 5 different steps:
1. Apply Gaussian filter to smooth the image in order to remove the noise
2. Find the intensity gradients of the image
3. Apply non-maximum suppression to get rid of spurious response to edge detection
4. Apply double threshold to determine potential edges (supplied by the user)
5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that
are weak and not connected to strong edges.

https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
                                                 
"""



from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np


img = cv2.imread('images/sandstone.tif', 0)

#Canny
canny_edge = cv2.Canny(img, 50, 80)  #Supply Thresholds 1 and 2 

#Autocanny
sigma = 0.3
median = np.median(img)

# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * median)) 
#Lower threshold is sigma % lower than median
#If the value is below 0 then take 0 as the value

upper = int(min(255, (1.0 + sigma) * median)) 
#Upper threshold is sigma% higher than median
#If the value is larger than 255 then take 255 a the value

auto_canny = cv2.Canny(img, lower, upper)


cv2.imshow("Canny", canny_edge)
cv2.imshow("Auto Canny", auto_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

