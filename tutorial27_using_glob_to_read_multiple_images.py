#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""

### Reading multiple images from a folder
#The glob module finds all the path names 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 

"""

#import the library opencv
import cv2
import glob

file_list = glob.glob('images/test_images/*.*') #Rerurns a list of file names
print(file_list)  #Prints the list containing file names

#Now let us load each file at a time...
my_list=[]  #Empty list to store images from the folder.
path = "images/test_images/*.*"
for file in glob.glob(path):   #Iterate through each file in the list using for
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    my_list.append(a)  #Create a list of images (not just file names but full images)
    
#View images from the stored list
from matplotlib import pyplot as plt
plt.imshow(my_list[2])  #View the 3rd image in the list.

########################################################################
#Now, let us load images and perform some action.
#import the opencv library so we can use it to read and process images
import cv2
import glob

#select the path
path = "images/test_images/*.*"
img_number = 1  #Start an iterator for image number.
#This number can be later added to output image file names.

for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    #print(a)  #print numpy arrays for each file

#let us look at each file
#    cv2.imshow('Original Image', a)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#process each image - change color from BGR to RGB.
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  #Change color space from BGR to RGB
    cv2.imwrite("images/test_images/Color_image"+str(img_number)+".jpg", c)
    img_number +=1 
    cv2.imshow('Color image', c)
    cv2.waitKey(1000)  #Display each image for 1 second
    cv2.destroyAllWindows()

#######################################################################################