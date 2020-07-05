#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
numpy array is like a list but better suited for scientific computing.
An array is a grid of values, all of the same type.
The shape of an array is a tuple of integers reflecting the size of array
along each dimension. 

"""

#To understand why we need numpy array let us look at the limitation of lists

a = [1,2,3,4,5]
b=2*a 
print(b)   #Repeats the array twice and does not multiply each element by 2

import numpy as np
c=np.array(a)
d=2*c
print(d)  #This type of math is required for image processing.

#Also try other math... like c**2
print(c**2)

#Add 2 arrays
import numpy as np
a = np.array([1,2,3,4,5])
b= np.array([6,7,8])
c = np.array([9, 10, 11, 12, 13])

print(a+b) #Not possible as the dimensions are different for both arrays
print(a+c) #Possible

#Creating numpy arrays

x = np.array([[1,2],[3,4]])  #Integer values
y = np.array([[1,2],[3,4]], dtype=np.float64)  #Integer values


#When you import images using most libraries it automatically imports them as numpy arrays

from skimage import io
img1 = io.imread('images/Osteosarcoma_01.tif')
print(type(img1))

#You can also generate arrays using built-in functions
a=np.ones((3,3)) #3x3 array with 1s
#Similarly you can fill zeros np.zeros((3x3))

#If we already have an array we can create another array of same shape
#and fill all with ones or zeros
b=np.ones_like(img1)

c = np.random.random((3,3))  #Fill with Random values between 0 and 1

#Slicing - subset of numpy arrays
a=np.array([[1,2,3,4], [5,6,7,8],[9,10,11,12]])
print(np.shape(a))  #3x4 array. 1st value in tuple is row and the second column

b=a[:2]  #First 2 rows. Single value provided so by default rows.
c=a[:, :2]  #First colon says pick all rows, second says first 2 columns
d = a[:2, 1:3]  #First 2 rows, columns 1 and 2 (3 not included)

#Adding values in rows and columns
column_sum = np.sum(a, axis=0)
row_sum = np.sum(a, axis=1)
max_value = np.max(a)

transposed = a.T  #Transposes the arrays (rows to columns)




