#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

#Scratch Assay on time series images
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5154238/

import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
from skimage.filters import threshold_otsu

#Use glob to extract image names and load them. 
import glob

time = 0
scale = 0.45 # microns/pixel
time_list=[]
area_list=[]
path = "images/scratch_assay/*.*"

#Put the code from single image segmentation in af for loop
# to apply segmentaion to all images
for file in glob.glob(path):
    img=io.imread(file)
    entropy_img = entropy(img, disk(3))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    scratch_area = np.sum(binary == 1)
    scratch_area = scratch_area*((scale)**2)  #Convert to microns from pixel units
    print("time=", time, "hr  ", "Scratch area=", scratch_area, "um\N{SUPERSCRIPT TWO}")
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1

#print(time_list, area_list)
plt.plot(time_list, area_list, 'bo')  #Print blue dots scatter plot

#Print slope, intercept
from scipy.stats import linregress #Linear regression
#print(linregress(time_list, area_list))

slope, intercept, r_value, p_value, std_err = linregress(time_list, area_list)
print("y = ",slope, "x", " + ", intercept  )
print("R\N{SUPERSCRIPT TWO} = ", r_value**2)
#print("r-squared: %f" % r_value**2)


