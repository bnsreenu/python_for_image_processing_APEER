#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

"""
In python variables can be stored in various types based on the purpose.

Text:	str
Numbers:	int, float, complex
Arrays (lists):	list, tuple, range
Mapping:	dict
Boolean (True/False):	bool


"""

a = "Hello"
print(type(a))

b=5
b=str(5)   #Converts integer to string

c = b*2  #You will get 55 and not 10. 
#Because string multiplication is concatenating...

c = int(b)*2  #Convert string to int. 

#Float
d = 10.5
e=0.023

f=int(d)  #Rounds to integer

g = (b==c) #Boolean True/False

#Note about float.
#While float can be any number, while converting images from int to float
#using skimage library rescales values to between 0 and 1. 

"""
For image processing using skimage:
uint8: 0 to 255
uint16: 0 to 65535
uint32: 0 to 232 - 1
float: -1 to 1 or 0 to 1
int8: -128 to 127
int16: -32768 to 32767
int32: -231 to 231 - 1

"""



from skimage import io, img_as_float

img = io.imread('images/Osteosarcoma_01.tif')  #Uint8
img1 = img_as_float(img)  #All values between 0 and 1

#NOTE: if input image is dtype is int32 then the values would be scaled
#back to between -1 and 1. For uint8 it scales to 0 and 1