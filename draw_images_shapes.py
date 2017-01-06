# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:29:19 2016

@author: Karthick Perumal
"""

import cv2
import numpy as np


#########################################################################################################

### To create an image 

#########################################################################################################
"""
## create a black image
image = np.zeros((512, 512, 3), np.uint8)

###we could also make it in black and white
image_bw = np.zeros((512, 512), np.uint8)

cv2.imshow("Black Rectangle (color)", image)
cv2.imshow("Black Rectangle (B&W)", image_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### To draw a blue line of thickness of 5 pixels

#########################################################################################################
"""
image = np.zeros((512, 512, 3), np.uint8)
cv2.line(image, (0,0), (511, 511), (255, 127, 0), 5)
cv2.imshow("Blue Line", image)

cv2.waitKey()
cv2.destroyAllWindows()

"""

#########################################################################################################

### To draw a rectangle in the image
### cv2.rectangle(image, starting vertex, opposite vertex, color, thickness)
#########################################################################################################
"""
image = np.zeros((512, 512, 3), np.uint8)

cv2.rectangle(image, (100, 100), (300, 250), (127, 50, 127), 5) ## if instead of pixel size 5, if we use -1, it will fill the rectangle with that colour
cv2.imshow("Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### To draw a circle in the image
### cv2.circle(image, center, radius, color, fill)
#########################################################################################################
"""
image = np.zeros((512, 512, 3), np.uint8)

cv2.circle(image, (256, 256), 100, (15, 75, 50), -1)
cv2.imshow("Circle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### To draw a polygon in the image
### cv2.circle(image, center, radius, color, fill)
#########################################################################################################
"""
image = np.zeros((512, 512, 3), np.uint8)

## define the four point or the edges of the polygon
pts = np.array([[10, 50], [100, 50], [90, 200], [50, 300]], np.int32)
#print(pts)
### Let's reshape our points in form required by polylines/polygons
pts = pts.reshape((-1, 1, 2))
#print(pts)
cv2.polylines(image, [pts], True, (0, 0, 255), 3)  ## True if we want the polygon closed or not
cv2.imshow("Polygon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### To add text 
### cv2.putText(image, 'Text to Display', bottom left starting point, Font, Font size, Color, Thickness)
#########################################################################################################

image = np.zeros((512, 512, 3), np.uint8)

cv2.putText(image, "Hello Ramya", (50, 260), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 10, 0), 3)
cv2.imshow("Hello Karthick", image)
cv2.waitKey()
cv2.destroyAllWindows()
