#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""

Sharpness Estimation for Document and Scene Images
by Jayant Kumar , Francine Chen , David Doermann

http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=33CD0038A0D2D24AE2C4F1A30B6EF1A4?doi=10.1.1.359.7002&rep=rep1&type=pdf

https://github.com/umang-singhal/pydom

pip install git+https://github.com/umang-singhal/pydom.git

#Use difference of differences in grayscale values 
of a median-filtered image as an indicator of edge sharpness
"""


from dom import DOM
import cv2

#img = cv2.imread("images/image_quality_estimation/02_2sigma_blurred.tif", 1)
img1 = cv2.imread("images/blurred_images/sandstone.tif", 1)
img2 = cv2.imread("images/blurred_images/sandstone_2sigma_blur.tif", 1)
img3 = cv2.imread("images/blurred_images/sandstone_3sigma_blur.tif", 1)
img4 = cv2.imread("images/blurred_images/sandstone_5sigma_blur.tif", 1)

# initialize DOM
iqa = DOM()

#Calculate scores
score1 = iqa.get_sharpness(img1)
score2 = iqa.get_sharpness(img2)
score3 = iqa.get_sharpness(img3)
score4 = iqa.get_sharpness(img4)

print("Sharpness for reference image:", score1)
print("Sharpness for 2 sigma blurred image:", score2)
print("Sharpness for 3 sigma blurred image:", score3)
print("Sharpness for 5 sigma blurred image:", score4)
