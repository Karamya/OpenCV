# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:39:16 2016

@author: Karthick Perumal
"""

import cv2
import numpy as np

#########################################################################################################

#### Line detection - Hough lines and probabilistic Hough lines  
#### Imagine being used in lane detection in self driving cars, in chess boards
####   ro = x cos theta + y sin theta , where ro is the perpendicular distance from the origin and
####               theta is the angle formed by the normal of this line to the origin (in radians)
#### cv2.HoughLines(binarized image, ro accuracy, theta accuracy, threshold)
###                 threshold here is the minimum vote for it to be considered a line
###probabilistic Hough lines - idea is that it takes only a random subset of points sufficient enough for line detection
#### also returns the start and end point of the line unlike the previous function

#### cv2.HoughLinesP(binarized image, ro accuracy, theta accurancy, threshold, minimum line length, max line gap)
###    http://www.bmva.org/bmvc/1998/pdf/p176.pdf

#########################################################################################################
"""
image = cv2.imread('images/soduku.jpg')

# Grayscale and canny edges extracted
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)  ## canny edges help a bit in transforms

### Run Houghlines using a rho accuracy of 1 pixel
### theta accuracy of np.pi/180 which is 1 degree
### our line threshold is set to 240 (number of points on line)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 220)



### We iterate through each line and convert it to the format required by cv.lines (i.e., requiring end points)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""    
#########################################################################################################

#### Probabilistic Hough Lines
#### cv2.HoughLinesP(binarized image, ro accuracy, theta accurancy, threshold, minimum line length, max line gap)
###    http://www.bmva.org/bmvc/1998/pdf/p176.pdf

#########################################################################################################

### Grayscale and canny Edges extracted
image = cv2.imread('images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

### Again we use the same rho and theta accuracies
### However, we specify a minimum vote(points along line) of 100
### and min line length of 5 pixels and max gap between lines of 10 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, 5, 10)
print(lines.shape)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()