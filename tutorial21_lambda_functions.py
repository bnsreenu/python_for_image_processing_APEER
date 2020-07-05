#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
Lambda Functions, also knoun as anonymous functions as they do not have names.

Regular python functions need to be defined with a name and can have any
number of arguments and expressions.

Lambda functions do not have names, they can have any number of arguments
but can only have one expression. These are good for one task that needs repetetion.  

"""
#For example if you want to define a function to square a number.

def squared(x):
    return(x**2)

print(squared(5))
# With lambda function you can define the same simply as...
# lambda is not the name of the function, it is a python keyword. 

f = lambda x: x**2  #Like creating a function f
print(f(5))  #Execute the function.

#Lambda functions can take multiple arguments, still one expression / statement

g = lambda x, y : 2*x**2 + 3*y
print(g(4,5))



"""
Lambda functions can be used inside other regular functions. In fact, this is 
their main advantage. 
"""

# S = ut + 1/2 a*t**2

def distance_eqn(u, a):
    return lambda t: u*t + ((1/2)*a*t**2)

#1t way of implementing
dist = distance_eqn(5, 10)  #Create a dist object/function first
print(dist(20)) #Then supply a value for t

#2nd way of implementing, give all values at once... 
print(distance_eqn(5, 10)(20))

#Makes it easy for automation...
#Create empty lists to be populated.
time=[]  #Lists covered in tutorial 12
dist=[]
for t in range(0, 22, 2):   #For loops covered in tutorial 19. 
    d = distance_eqn(5, 10)(t)
    time.append(t)
    dist.append(d)
    #print("At time = ", t, "The distance is ", dist)
from matplotlib import pyplot as plt
plt.plot(time, dist)   


# Why is this relevant for image processing?
import cv2
img=cv2.imread("images/Osteosarcoma_01.tif", 0)
#Images are arrays and applying lambda to an array is easy
#e.g. convert 8 bit image to float with values between 0 and 1.
#Common task for certain deep learning approaches. 
# Data types covered in tutorial 16
#map() - apply an operation to every element in the list/array
filtered_img = map (lambda x: x/255., img)  #Stores mapped values

#Numpy arrays covered in tutorial 15
import numpy as np
my_float_img=np.array(list(filtered_img))#Convert mapped values to a numpy array
print(my_float_img)






 