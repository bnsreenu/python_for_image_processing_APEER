
# https://youtu.be/-u8PHmHxJ5Q
"""
@author: Sreenivas Bhattiprolu
"""

#https://pypi.org/project/patchify/
from patchify import patchify, unpatchify
import cv2
import os

patch_size = 128
step = 128

image_path = "images/images_for_deep_learning/512_cropped_images/"
for img in os.listdir(image_path):
    #print(img)
    #Needs 8 bit, not float.
    image = cv2.imread(image_path+img, 0)
    
    image_patches = patchify(image, (patch_size, patch_size), step=step)

    m, n = image_patches.shape[0], image_patches.shape[1]

    for i in range(m):
        for j in range(n):
            print(i,j)
            patch = image_patches[i,j]
            cv2.imwrite("images/images_for_deep_learning/128_patches_images/"
                        +"_"+img+str(i)+"_"+str(j)+".png", patch)
            

mask_path = "images/images_for_deep_learning/512_cropped_masks/"
for msk in os.listdir(mask_path):
    #print(img)
    #Needs 8 bit, not float.
    mask = cv2.imread(mask_path+msk, 0)
    
    mask_patches = patchify(mask, (patch_size, patch_size), step=step)

    x, y = mask_patches.shape[0], mask_patches.shape[1]

    for i in range(x):
        for j in range(y):
            print(i,j)
            patch = mask_patches[i,j]
            cv2.imwrite("images/images_for_deep_learning/128_patches_masks/"
                        +"_"+msk+str(i)+"_"+str(j)+".png", patch)
            


#reconstructed_image = unpatchify(patches, image.shape)
