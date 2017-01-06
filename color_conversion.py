# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:06:31 2016

@author: Karthick Perumal
"""

import cv2
import os
os.chdir("D:/Data analysis/OpenCV")

image = cv2.imread('./images/input.jpg')

cv2.imshow('Original', image)
cv2.waitKey()


## Use cvtColor to convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
cv2.imshow('GrayScale', gray_image)
cv2.waitKey()

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('HSVscale', hsv_image)
cv2.waitKey()

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)



for i in flags:
    color_code = 'cv2.' + i
    temp_image = cv2.cvtColor(image, color_code)
    cv2.imshow(i, temp_image)
    cv2.waitKey()

cv2.destroyAllWindows()