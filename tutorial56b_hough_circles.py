#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Measure properties of labeled image regions.


https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

# The input image.
img = cv2.imread("images/cast_iron1.tif", 0)


img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
#if dp=1 , the accumulator has the same resolution as the input image. 
#If dp=2 , the accumulator has half as big width and height.
#minDist – Minimum distance between the centers of the detected circles.
#minRadius – Minimum circle radius.
#maxRadius – Maximum circle radius.

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30,
                            param1=50, param2=12, minRadius=10, maxRadius=20)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()