#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Functions
"""

def my_function():
    print("Hello from inside a function")
    
#Now we are outside the function

my_function()  #When you call the function it executes it including print

#You can provide inputs to functions


def my_function(your_name="Michael"):  #Michael is the default value in case nothing is provided
    print("Your name is: ", your_name)
    
#Now we are outside the function

my_function("John")
my_function("Mary")
my_function()


#Iterate through lists from inside a function

def my_microscopes(mic):
    for x in mic:
        print(x)
        
mic = ["AxioImager", "Elyra", "LSM", "GeminiSEM", "Versa"]
my_microscopes(mic)

#Returning values
#When the function is done executing it can return values.

def add_numbers(a,b):
    return a+b

print(add_numbers(5,3))

#Let us write a function to perform Gaussian smoothing

from skimage import io, filters
from matplotlib import pyplot as plt

def gaussian_of_img(img, sigma=1):
    gaussian_img = filters.gaussian(img, sigma)
    return(gaussian_img)

my_image = io.imread('images/Osteosarcoma_01_8bit_salt_pepper_cropped.tif')
filtered = gaussian_of_img(my_image, 3)

plt.imshow(filtered)