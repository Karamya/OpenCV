# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:39:06 2016

@author: Karthick Perumal
"""
import cv2
import numpy as np
import os
os.chdir("D:/Data analysis/OpenCV")
"""
Transformations :
Affine - scaling, rotation , translation
Non-Affine (or projective transform or homography) - does not preserve parallelish, length and angle. It however preserves collinearity and incidence


Translation Matrix

T = | 1 0 Tx |
    | 0 1 Ty |
    
    Tx represents the shift along the x-axis (horizontal)
    Ty represents the shift along the y-axis (vertical)
use openCV function cv2.warpAffine to implement these translations    
    
    
"""
#########################################################################################################

### Affine translation
### cv2.warpAffine(image, Translation matrix, image_shape)
# T = | 1 0 Tx |
#     | 0 1 Ty |
# Where T is the translation matrix
#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

## store height and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

# T = | 1 0 Tx |
#     | 0 1 Ty |
# Where T is the translation matrix

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

# we use warpAffine to transform the image using the matrix T

img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow("Translation", img_translation)
cv2.waitKey()
cv2.destroyAllWindows()

print(T)
"""
#########################################################################################################

### Affine rotation
### cv2.warpAffine(image, Translation matrix, image_shape)
# M = | cos theta   -sin theta |
#     | sin theta    cos theta |
# Where theta is the angle of rotation
#cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle_of_rotation, scale)
#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

height, width = image.shape[:2]

##divide by two to rotatate the iamge around its center
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('Rotated image', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()
"""
#########################################################################################################

### Affine rotation - But transpose to avoid the black region and resizing of the final image
### cv2.transpose(image)

#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

rotated_image = cv2.transpose(image)

cv2.imshow('Rotated Image - method 2', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()
"""

#########################################################################################################

### Re-Sizing, scaling and interpolation
### cv2.INTER_AREA - good for shrinking or down sampling
### cv2.INTER_NEAREST - fastest
### cv2.INTER_LINEAR - Good for zooming or up scaling(default)
### cv2.INTER_CUBIC - Better
### cv2.INTER_LANCZOS4 - Best

### Good comparison of interpolation methods http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
### cv2.resize(image, dsize(output image size), x scale, y scale, interpolation)
#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

# Let's make our image 3/4 of it's original size
image_scaled = cv2.resize(image, None, fx = 0.75, fy = 0.75)  ## fx and fy are the factors
cv2.imshow('Scaling - Linear interpolation', image_scaled)
cv2.waitKey()

## make the image double it's size
img_scaled = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic interpolation', img_scaled)
cv2.waitKey()

## skew the re-sizing by setting exact dimensions
img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey()

cv2.destroyAllWindows()
"""
#########################################################################################################

### Image Pyramids
### pyramiding image refers to either upscaling(enlarging)  or downscaling(shrinking) of images
### this is useful when making object detectors that scales images each time it looks for an object

#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

smaller = cv2.pyrDown(image) ## converts to half its size of the original  (for multiple smaller sizes, put it in a loop)
larger= cv2.pyrUp(smaller)  ## coverts to twice the size of the original  (becomes blurry upon resizing)
cv2.imshow('Original', image)
cv2.imshow('Smaller', smaller)
cv2.imshow('Larger', larger)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### Cropping of an image

#########################################################################################################
"""
image = cv2.imread('images/input.jpg')
height, width = image.shape[:2]

# Let's get the starting pixel coordinates (top left of cropping rectangle)
start_row, start_col = int(height*0.25), int(width*0.25)

#Let's get the ending pixel coordinates (bottom right)
end_row, end_col = int(height*0.75), int(width*0.75)

# simple use indexing to crop out the rectangle we desire
cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### Arithmetic Operations - To directly add or subtract to the color intensity
### calculates the per-element operation of two arrays. Overall effect is increasing or decreasing brightness

#########################################################################################################
"""
image = cv2.imread('images/input.jpg')

#create a matrix of one, then multiply it by a scaler of 100
# this gives a matrix with same dimensions of our image with all values being 100
M = np.ones(image.shape, dtype = 'uint8')*100

# We use this to add this matrix M to our image
added = cv2.add(image, M)
cv2.imshow('Added', added)   ### We can never exceed 255 or go below 0 value in colors

# Likewise we can also subtract
subtracted = cv2.subtract(image, M)
cv2.imshow('Subtracted', subtracted)

cv2.waitkey(10)
cv2.destroyAllWindows()
""" 
#########################################################################################################

### Bitwise operations and Masking
### 

#########################################################################################################
"""
## we use only two dimensions, as this is a grayscale image
## for coloured image, we  would use 
## rectangle = np.zeros((300, 300, 3), np.uint8)

# making a square
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow('Square', square)
cv2.waitKey(0)

# Making an ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1) #check opencv documentation for ellipse
cv2.imshow('Ellipse', ellipse)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)  # must have the same dimensions
cv2.imshow('AND', And)
cv2.waitKey(0)

# Shows only where either square or ellipse is
bitwise_or = cv2.bitwise_or(square, ellipse)
cv2.imshow('OR', bitwise_or)
cv2.waitKey(0)

# Shows 1xor1 is 0, 0xor0 is 0, 1xor0 or 0xor1 is 1 
bitwise_xor = cv2.bitwise_xor(square, ellipse)
cv2.imshow('XOR', bitwise_xor)
cv2.waitKey(0)

# Shows everything that insn't part of the square
bitwise_not_square = cv2.bitwise_not(square)
cv2.imshow('NOT Square', bitwise_not_square)
cv2.waitKey(0)

"""
#########################################################################################################

### Convolutions and Blurring
### convolution is a mathematical operation performed on two functions producing a third function 
### which is typically a modified version of one of the original functions
### Output image = image convolution Function(Kernel Size)
### in Computer vision we use Kernel's to specify the size over which we run our manipulating function over our image

### Blurring is an operation where we average the pixels within a region(kernel)
### cv2.filter2D(image, -1, kernel)

#########################################################################################################
"""
image = cv2.imread('images/elephant.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)

## Creating our 3x3 kernel
kernel_3x3 = np.ones((3, 3), np.float32)/9

## We use the cv2.filter2D to convolve the kernel with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred)
cv2.waitKey(0)

# Creating our 7x7 Kernel
kernel_7x7 = np.ones((7, 7), np.float32)/49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred2)
cv2.waitKey(0)

cv2.destroyAllWindows() 


### Averaging done by convolving the image with a normalized box filter
### This takes the pixels under the box and replaces the central element
# box size needs to be odd and positive
blur = cv2.blur(image, (3,3)) ## with in a box size of 3x3
cv2.imshow('Averaging', blur)
cv2.waitKey(0)

# Gaussian kernel instead of a box filter
gaussian = cv2.GaussianBlur(image, (7,7), 0) 
cv2.imshow('Gaussian Blurring', gaussian)
cv2.waitKey(0)

# Takes median of all the pixels under the kernel area and central element is 
# replaced with this median value
median = cv2.medianBlur(image, 5) ### Nice balance between gaussian and averaging
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)

## Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

## image de-noising - Non-local means denoising

# parameters after None are - the filter strength 'h' (5-10 is a good range)
# Next is hForColorComponents, set as same value as h again

dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
cv2.imshow('Fast means denoising', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
### Four variations of non-local means denoising
### cv2.fastNlMeansDenoising() - works with a single grayscale images
### cv2.fastNlMeansDenoisingColored() - works with a single grayscale images
### cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time(grayscale images)
### cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images
"""
#########################################################################################################

### Sharpening - opposite of blurring, it strengthens or emphasize edges in an image
###  Kernel matrix sums to one, so there is no need to normalize (ie., multiply by a factor to 
### retain the same brightness of the original)
### Kernel = | -1 -1 -1 |
###          |-1  9  -1 |
###          | -1 -1 -1 | 

#########################################################################################################
"""
image = cv2.imread('images/input.jpg')
cv2.imshow('Original', image)

# create our sharpening kernel, we don't normalize since the values in the matrix sum to 1
kernel_sharpening = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow('Image Sharpening', sharpened)

cv2.waitKey()
cv2.destroyAllWindows()
"""
#########################################################################################################

### Thresholding, Binarization and Adaptive thresholding
### Thresholding - act of converting an image to a binary form
### cv2.threshold(image, Threshold Value, Max Value, Threshold Type)

### Threshold Types:
### cv2.THRESH_BINARY  - Most common
### cv2.THRESH_BINARY_INV - Most common
### cv2.THRESH_TRUNC   ## Everything above the thresold is truncated
### cv2.THRESH_TOZERO  ##  Everything below the threshold is set to black
### cv2.THRESH_TOZERO_INV  ## inverse of before
########################Note : Image need to be converted to greyscale before thresholding#############
#########################################################################################################
"""
# load our image as greyscale
image = cv2.imread('images/gradient.jpg', 0)
cv2.imshow('Original', image)

## Values below 127 goes to 0 (black), everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1 Threshold Binary', thresh1)

## Values below 127 goes to 255 , everything above goes 127 to 0 
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('2 Threshold Binary Inverse', thresh2)

## Values above 127 are truncated(held) at 127 (the 255 argument is unused) 
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('3 Threshold Trunc', thresh3)

## Values below 127 goes to 0 , above 127 are unchanged
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('4 Threshold to Zero', thresh4)

## Values below 127 is unchanged , above 127 goes to zero
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('5 Threshold to Zero inverse', thresh5)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### Adaptive thresholding - takes the uncertainity away (while simple threshold methods require us to provide the threshold value)
### cv2.adaptiveThreshold(image, Max Value, Adaptive type, Threshold Type, Block Size, Constant that is subtracted from mean)
### NOTE: Block sizes need to be odd numbers

### Adaptive threshold types:
### ADAPTIVE_THRESH_MEAN_C - based on mean of the neighborhood of pixels
### ADAPTIVE_THRESH_GAUSSIAN_C - weighted sum of neighborhood pixels under the Gaussian Window
### THRESH_OTSU (uses cv2.threshold function) - Clever algorithm assumes there are two peaks in the gray scale histogram 
### of the image and then tries to find an optimal value to separate these two peaks to find T

#########################################################################################################
"""
image = cv2.imread('images/Origin_of_Species.jpg', 0)

cv2.imshow('Original', image)
cv2.waitKey(0)

# Values below 127 goes to 0 (black), everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

# It's good practive to blur images as it removes noise
image = cv2.GaussianBlur(image, (3,3), 0)

# using adaptiveThreshold
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
cv2.imshow('Adaptive Mean Thresholding',  thresh)
cv2.waitKey(0)

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Thresholding", th2)
cv2.waitKey(0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(image, (5,5), 0)
_,th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Gaussian Otsu's Thresholding", th3)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#########################################################################################################

### Dilation and Erosion

### Dilation - Adds pixels to the boundaries of objects in an image
### Erosion - Removes pixels at the boundaries of objects in an image
### Opening - Erosion followed by dilation
### Closing - Dilation followed by erosion
### Other methods - http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
#########################################################################################################
"""
image = cv2.imread('images/opencv_inv.png', 0)

cv2.imshow('Original', image)
cv2.waitKey(0)

# Let's define our kernel size
kernel = np.ones((5,5), np.uint8)

# Now we erode
erosion = cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)

#
dilation = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv2.waitKey(0)

# Opening - Good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey(0)

# Closing - Goood for removing noise
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#########################################################################################################

### Edge detection and image Gradients

### Edge detection is a very important area in computer vision, especially when dealing with contours
### Edges can be defined as sudden changes (discontinuities) in an image and they can encode just as much information as pixels

### Three main types of Edge detection:
### Sobel - To emphasize vertical or horizontal edges
### Laplacian - Gets all orientations
### Canny - optimal due to low error rate, well defined edges and accurate detection

### Canny Edge Detection Algorithm (developed by John F Canny in 1986)
### 1. Applies Gaussian blurring
### 2. Finds intensity gradient of the image
### 3. Applied non-maximum suppression (i.e., removes pixels that are not edges)
### 4. Hysteresis - Applies thresholds (i.e., if pixel is within the upper and lower thresholds, it is considered an edge)
### Check for research articles on edge detection algorithm comparisons
### 

#########################################################################################################
"""
image = cv2.imread('images/input.jpg', 0)

height, width = image.shape

# Extract Sobel Edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)

cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.imshow('Sobel X', sobel_x)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_OR', sobel_OR)
cv2.waitKey(0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)

## Then, we need to provide two values: Threshold1 and threshold2. Any gradient value larger than 
## threshold2 is considered to be an edge. Any value below threshold1 is considered not to be an edge.
## Value in between threshold1 and threshold2 are either classified as edges or non-edges based on how
## their intensities are 'connected'. In this case, any gradient values below 60 are considered non-edges 
### whereas any values above 120 are considered edges

## Canny edge detection uses gradient values as thresholds
## The first threshold gradient
canny = cv2.Canny(image, 60, 150)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#########################################################################################################

### Obtaining the perspective of Non-Affine transforms
### Getting perspective transform

#########################################################################################################
"""
image = cv2.imread('images/scan.jpg')

cv2.imshow('Original', image)
cv2.waitKey(0)

## Coordinates of the 4 corners of the original image

points_A = np.float32([[320, 15], [700, 215], [85, 610], [530, 780]])

### coordinates of the 4 corners of the desired output
# we use a ratio of an A4 paper 1: 1.41
points_B = np.float32([[0, 0], [420, 0], [0,594], [420, 594]])

## Use the two sets of four points to compute
## the perspective Transformation matrix, M
M = cv2.getPerspectiveTransform(points_A, points_B)

warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow('Warp Perspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#########################################################################################################

### Affine Transforms need only 3 coordinates to obtain the correct transform
### Getting perspective transform

#########################################################################################################
"""

image = cv2.imread('images/ex2.jpg')

rows, cols, ch = image.shape

cv2.imshow('Original', image)
cv2.waitKey(0)

## Coordinates of  3 corners of the original image
points_A = np.float32([[320, 15], [700, 215], [85, 610]])

### coordinates of  3 corners of the desired output
# we use a ratio of an A4 paper 1: 1.41
points_B = np.float32([[0, 0], [420, 0], [0,594]])

## Use the two sets of 3 points to compute
## the affine Transformation matrix, M
M = cv2.getAffineTransform(points_A, points_B)

warped = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow('Warp Perspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

