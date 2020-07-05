
#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG


"""
Introductory python tutorials for image processing

Tutorial 3: Appreciating the simplicity of Python code

"""

from skimage import io, filters
from matplotlib import pyplot as plt

img = io.imread('images/Osteosarcoma_01_8bit_salt_pepper_cropped.tif')
gaussian_img = filters.gaussian(img, sigma=1)

plt.imshow(gaussian_img)
