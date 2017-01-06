# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:43:25 2016

@author: Karthick Perumal
"""

import cv2
import numpy as np
import os

os.chdir("D:/Data analysis/OpenCV")

image = cv2.imread("./images/input.jpg")  ###image is loaded and stored in variable image

cv2.imshow('Karthick', image)  ### First parameter is the title of the image window and second parameter is the image variable

# 'waitkey' allows us to input information when a image window is open
# by leaving it blank, it just waits for anykey to be pressed before continuing
# by placing numbers (except 0), we can specify a delay for how long you keep the window open( in milliseconds)
cv2.waitKey()

cv2.destroyAllWindows() ## not necessarily important, but better to always close them, else this will cause your program to hang

print(image.shape)  #gives the shape of the image

## to write the images specify the filename and the image to be saved
cv2.imwrite('output.jpg', image)