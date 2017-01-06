# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:01:20 2016

@author: Karthick Perumal
"""
import cv2
import numpy as np
#import cv2.cv as cv
##print(cv2.__version__)
#########################################################################################################

### Circle detection with Hough Circles
### cv2.HoughCircles(image, method, dp, MinDist, param1, param2, minRadius, MaxRadius)
### Method - cureently only cv2.HOUGH_GRADIENT available
### dp 0 inverse ratio of accumulator resolution
### MinDist - The minimum distance between the center of detected circles
### param1 - Gradient value used in the edge detection
### param2 - Accumulator threshold for the HOUGH_GRADIENT method, lower allows more circles to be detected (false positives)
### MinRadius - limits the smallest circle to this size (via radius)

#########################################################################################################

image = cv2.imread('images/bottlecaps.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 10) # dp = 1,  minRadius = 1.5, MaxRadius = 10)

for i in circles[0,:]:
    ## draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)
    
    ### draw the center of the circle
    cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)
    
cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()