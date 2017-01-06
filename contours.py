# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:49:28 2016

@author: Karthick Perumal
"""

#########################################################################################################

### Contours are continuous lines or curves that bound or cover the full boundary of an object in an image
### Contours are very important in Object detection and shape analysis

#########################################################################################################

import cv2
import numpy as np
import os
os.chdir("D:/Data analysis/OpenCV")


#########################################################################################################

### Finding contours

#########################################################################################################

"""
image = cv2.imread('images/shapes_donut.jpg')
cv2.imshow('Input Image', image)
cv2.waitKey(0)

##Convert to gray scale. Colored images won't work  with openCV
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### Find canny edges, it is not necessary, but good to reduce the number of unnecessary contours when doing findcontours
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)

##Finding Contours
## Use a copy of your image e.g. edged.copy(), since findCountours alters the image

#### http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
####
####mode –
####Contour retrieval mode (if you use Python see also a note below).
###
###CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
###CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
###CV_RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
###CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV contours.c demo.
######method –
###Contour approximation method (if you use Python see also a note below).
###
###CV_CHAIN_APPROX_NONE stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
###CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
###CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS applies one of the flavors of the Teh-Chin chain approximation algorithm. See [TehChin89] for details.
###offset – Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.
###

_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  ## Change cv2.RETR_EXTERNAL to cv2.RETR_LIST
cv2.imshow('Canny Edges after Contouring', edged)
cv2.waitKey(0)

print('Number of Contours found = ' + str(len(contours)))

# Draw all Contours
## use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0,255,0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
#########################################################################################################

### Drawing the contour on blank images, just not to miss the contour from a coloured image
### 
#########################################################################################################

"""
image = cv2.imread('images/bunchofshapes.jpg')
cv2.imshow("0 - Original Image", image)
cv2.waitKey(0)

## Create a black image with same dimensions as our loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))

# create a copy of our original image
original_image = image

# Grayscale our image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find canny edges
edged = cv2.Canny(gray, 50, 200)
cv2.imshow('1 - Canny Edges', edged)
cv2.waitKey(0)

## Find Contours and print how many were found
ret, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found =", len(contours))

## Draw all contours over blank image
cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 3)
cv2.imshow('2 - All Contours over blank image', blank_image)
cv2.waitKey(0)

## Draw all contours over the real image
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('3 - All Contours', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#########################################################################################################

### Sorting contours 
### Sorting by area can assist in image recognition. Eliminate small contours that maybe noise, Extract the largest contour
### sorting by spatial position (using the contour centroid) -sort characters left to right, process images in specific order
#########################################################################################################

"""
def get_contour_areas(contours):
    all_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        all_areas.append(area)
    return all_areas
    
image = cv2.imread('images/bunchofshapes.jpg')
original_image = image

# Grayscale our image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Find canny edges
edged = cv2.Canny(gray, 50, 200)

## Find Contours and print how many were found
ret, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found =", len(contours))

print("Contour areas before sorting")
print(get_contour_areas(contours))

## Sort contours large to small
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

print("Contour areas after sorting")
print(get_contour_areas(sorted_contours))

## Iterate over our contours and draw one at a time
for c in sorted_contours:
    cv2.drawContours(original_image, [c], -1, (255, 0, 0), 3)
    cv2.waitKey(0)
    cv2.imshow('Contours by area', original_image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

#########################################################################################################

#### Sorting contours from left to right (similarly can be done for top to bottom position)

#########################################################################################################

def x_cord_contour(contours):
    ## returns the x coordinate for the contour centroid
    if cv2.contourArea(contours)>10:
        M = cv2.moments(contours)
        return(int(M['m10']/M['m00']))
        
def label_contour_center(image, c, i):
    ### places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    # Draw the contour number on the image
    cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
    return image
    

# Load our image 
image = cv2.imread('images/bunchofshapes.jpg')
original_image = image.copy()

## Compute center of mass or centroids and draw them on our image
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c, i)

cv2.imshow("4 - Contour Centers", image)
cv2.waitKey(0)

### Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)

### Labeling Contours left to right
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.putText(original_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('6 - Left to Right Contour', original_image)
    cv2.waitKey(0)
    (x, y, w, h) = cv2.boundingRect(c)
    
    ### Let's now crop each contour and save these images
    cropped_contour = original_image[y:y + h, x: x + w]
    image_name = 'output_shape_number_'+str(i+1) + '.jpg'
    print(image_name)
    cv2.imwrite(image_name, cropped_contour)
    
cv2.destroyAllWindows()

"""
#########################################################################################################

#### Approximating Contours and finding their convex hull
### cv2.approxPolyDP(input_contour, approximation accuracy, closed)
#### approximation accuracy - small value gives precise approximations, large values give more generi approximation
### a good rule of thumb is less than 5% for the contour perimeter
### Closed - a Boolean value that states whether the approximate contour should be open or closed
#########################################################################################################
"""
## Load image and keep a copy
image = cv2.imread('images/house.jpg')
orig_image = image.copy()
cv2.imshow('Original Image', orig_image)
cv2.waitKey(0)

### Grayscale and binarize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

### Find Contours
ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

### Iterate through each contour and compute the bounding rectangle
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Bounding Rectangle', orig_image)
    
cv2.waitKey(0)

### Iterate through each contour and compute the approx. contour
for c in contours:
    ### Calculate accuracy as a percent of the contour perimeter
    accuracy = 0.01 * cv2.arcLength(c, True)  ### play with the decimal for accuracy
    approx = cv2.approxPolyDP(c, accuracy, True)  ### This is how we approximate contours
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow('Approx Poly DP', image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
#########################################################################################################

#### Convex Hull


#########################################################################################################
"""
image = cv2.imread('images/hand.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

### Threshold the image
ret, thresh = cv2.threshold(gray, 176, 255, 0)

### Find Contours
ret, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

## Sort contours by area and then remove the largest frame contour
n = len(contours) - 1  ### For white background, the whole are is taken as the first biggest contour
contours = sorted(contours, key = cv2.contourArea, reverse = False)[:n]

# Iterate through contours and draw the convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow('Convex Hull', image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

#### Shape Matching
### cv2.matchShape(contour template, contour, method, method parameter)
### contour template is the reference contour that we are trying to find in the new image
### contour - the indivcidual contour we are checking against
### method = type of contour matching (1, 2, 3)
### method parameter - leave alone as 0.0 (not fully utilized in python opencv)

#########################################################################################################

### Load the shape template or reference image
template = cv2.imread('images/4star.jpg', 0) ### 0 at the end loads the image as grayscale
cv2.imshow('Template', template)
cv2.waitKey()

### Load the traget image with the shapes we are trying to match
target = cv2.imread('images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

### Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

### Find contours in template
ret, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

### We need to sort the contours by area so that we can remove the largest contour which is the image outline
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

## We extract the second largest contour which will be our template contour
template_contour = contours[1]

## Extract contours from second target image
ret, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    ## Iterate through each contour in the target image and 
    ###  use cv2.matchShape to compare contour shapes
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    ## if the match value is less than 0.15 we 
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []
        
cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()

###