import cv2         
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import sys
import math


# Input file
in_file = sys.argv[1]

# Checks for valid command line arguments
prefix = in_file[:-4]
file_extension = in_file[-4:]

valid_extensions = ['.jpg','.png']
if file_extension not in valid_extensions:
    print('Invalid Filename')
    exit()

# Stores output file name
out_file = f"{prefix}_bats{file_extension}"

# Reads images
color_img = cv2.imread(in_file).astype(np.float32)
img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

# Inverts image and applies threshold
img = 255 - img
mean = np.average(img)
std = np.std(img)
thresh = mean + 2*std
retval,im_thresh = cv2.threshold(img.astype(np.uint8),thresh,255,cv2.THRESH_BINARY)

#Kernel to be used for opening and closing
kernel = np.ones((3,3),np.uint8)

#Opens the image and then closes it
after_open = cv2.morphologyEx(im_thresh,cv2.MORPH_OPEN,kernel)
im_open_first = cv2.morphologyEx(after_open,cv2.MORPH_CLOSE,kernel)

#Sets type to 8-connected for finding connected components
connectivity = 8
#Calculates connected components and corresponding statistics
#The returned connected components should correspond to each bat
num_labels, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(im_open_first, connectivity, cv2.CV_32S)

#First entry in returned statistics refers to entire image, 
#so all other indices correspond to the actual bats

#Array that stores the area of each bat
areas = stats[1:,4]
#Calculates desired radii for each circle
radii = np.sqrt(areas)
#Centroids of each bat
centroids = centroids[1:]
#Prints number of bats found
print(centroids.shape[0])
#Loops over each bat and prints its location and adds a circle to the color image around the bat
for i in range(centroids.shape[0]):
    x = round(centroids[i,1])
    y = round(centroids[i,0])
    print(x,y,areas[i])
    color_img = cv2.circle(color_img,(y,x),int(radii[i]),(0,0,255))

#Writes the final color image
cv2.imwrite(out_file,color_img)


