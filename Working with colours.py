# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:32:12 2016

@author: Karthick Perumal
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
os.chdir("D:/Data analysis/OpenCV")


"""
OpenCV stores color in BGR format

RGB - Additive color model

The integer representing a color e.g. 0x00BBGGRR will be stored as 0x00BBGGRR
Even though openCv takes color in BGR format, it stores it in RGB format

HSV colorspace

Hue - Color value (0-179)
Saturation - Vibrancy of color (0-255)  ## At low saturation everything is white
Value - Brightness or intensity (0-255) ## low value is dark and high value is bright

Color fitering is difficult in BGR format, so we use HSV

Color Range fiters: (Hue color range goes from 0 to 180 and not 360 and is mapped differently than standard)
Red - 165 to 15
Green - 45 to 75
Blue - 90 to 120

"""

image = cv2.imread('./images/input.jpg')  # IMG_3650.jpg input.jpg

"""
##BGR values for any pixel 
B, G, R = image[0, 0]
print(image[0,0])

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
print(gray_img[0,0])

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV image', hsv_image)
cv2.imshow('Hue channel', hsv_image[:,:,0])    # To display the hue channel of the image
cv2.imshow('Saturation channel', hsv_image[:,:,1])  # To display the saturation channel of the image
cv2.imshow('Value channel', hsv_image[:,:,2])  # To display the value channel of the imaage

cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
## To explore at individual channels  in an RGB image

B, G, R = cv2.split(image) #Split function splits the image into each color index

print(B.shape)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
cv2.destroyAllWindows()

### To recreate the original image
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

## To amplify the blue color
merged = cv2.merge([B+100, G, R]) ## Adds 100 to every B value. If the value reaches more than 255, it will maximize it to 255
cv2.imshow("Merged with Blue Amplified", merged)

cv2.waitKey(0)
cv2.destroyAllWindows()

"""

"""
### Let's represent the iamges in individual colors
B, G, R = cv2.split(image) # Split function to split the image into respective color index

## Let's create a matrix of zeros
# write dimensions of the image h x w
zeros = np.zeros(image.shape[:2], dtype = "uint8")  ## image shape without the color dimension as zeros
## image.shape[:2] (830, 1245)
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

cv2.waitkey()
cv2.destroyAllWindows()

"""

image = cv2.imread('./images/tobago.jpg')
cv2.imshow("Tobago", image)
## Histograms to visualize the color components
### cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
### images : it is the source image of type uint8 or float32. It should be given in square brackets
### channels: also give in square brackets. It is the index of channel for which we calculate histograms.
### If input is grayscale image, value is [0]. For color image, you can pass [0], [1], [2] to calculate histogram of blue, green or red channel respectively
### mask: mask image. To find histogram of full iamge, it is given as None. 
### histSize: this represents our BIN count. Need to be given in square brackets, for full scale, we pass [256]
# ranges: this is our range. Normally it is [0, 256]
histogram = cv2.calcHist([image], [0], None, [256], [0, 256]) 

## to plot a histogram, ravel() flattens our image array
plt.hist(image.ravel(), 256, [0,256])
plt.show()

color = ('b', 'g', 'r')
### To plot the colors separately and plot each in the histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0, 256])

plt.show()

