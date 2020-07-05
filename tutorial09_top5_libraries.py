#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
@author: Sreenivas Bhattiprolu
"""
import skimage
from skimage import io

img1 = io.imread('images/Osteosarcoma_01.tif')


import cv2
img2 = cv2.imread('images/Osteosarcoma_01.tif')

import numpy as np
a=np.ones((5,5))

import pandas as pd
df = pd.read_csv('images/image_measurements.csv')
print(df.head())

from matplotlib import pyplot as plt
plt.imshow(img1)