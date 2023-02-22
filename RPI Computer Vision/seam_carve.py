import cv2
import numpy as np
import os
import random
import sys
 
# Calculates the best vertical seam for an image
def findSeam(color_img):
    
    img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    
    # Calculates partial derivatives
    im_dx = cv2.Sobel(img,cv2.CV_32F,1,0)
    im_dy = cv2.Sobel(img,cv2.CV_32F,0,1) 
    
    # Calculates energies
    energies = abs(im_dx) + abs(im_dy)

    # Matrix for dynamic programming
    W = energies
    W[:,0] = 1000000
    W[:,-1] = 1000000
    
    for row in range(1,height):
        
        # Shifted images
        left = W[row - 1,:-2]
        right = W[row - 1,2:]
        center = W[row - 1,1:-1]
        
        # Finds the minimum W value at every location for the row above it
        mins1 = np.minimum(left,right)
        mins = np.minimum(mins1,center)
        
        # Adds in the current row energies to the minimum values
        W[row,1:-1] = energies[row,1:-1] + mins
        

    
    seam = np.zeros(height)
    
    # Calculates the final seam location
    last = int(np.argmin(W[-1,:]))
    seam[-1] = last
    
    # Creates the seam by backtracing through the image
    for row in range(height - 2,-1,-1):
        
        triple = W[row,last - 1:last + 2]
        index_shift = int(np.argmin(triple)) - 1
        seam[row] = last + index_shift
        last = int(seam[row])
        
    return seam


# Given a vertical seam on an image, colors all pixels along the seam red (0,0,255) in BGR
def colorSeam(color_img,seam):
    
    for row in range(0,color_img.shape[0]):
        seam_value = int(seam[row])
        color_img[row,seam_value] = np.array([0,0,255])
    return color_img



# Given vertical seam column locations for an image, removes all seam pixels to resize image
def removeSeam(color_img,seam):
    
    height = color_img.shape[0]
    width = color_img.shape[1]
    new_color_img = np.zeros((height,width - 1,3))
    
    for row in range(height):
        
        seam_value = int(seam[row])
        
        # Slices to the left and right of targeted pixel
        left = color_img[row,:seam_value]
        right = color_img[row,seam_value + 1:]
        
        # Puts row back together
        new_color_img[row,:] = np.concatenate((left,right),axis=0)

    return new_color_img



#Finds average energy along vertical seam
def findEnergy(color_img,seam):

    # Calculates partial derivatives
    img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    im_dx = cv2.Sobel(img,cv2.CV_32F,1,0)
    im_dy = cv2.Sobel(img,cv2.CV_32F,0,1) 
    
    # Calculates energies
    energies = abs(im_dx) + abs(im_dy)
    
    # Finds average enery along seam
    height = color_img.shape[0]
    total = 0
    for row in range(height):
        seam_value = int(seam[row])
        total += energies[row,seam_value]
    
    return total / height



if __name__ == "__main__":
    
    
    # Input file
    in_file = sys.argv[1]
    
    # Checks for valid command line arguments
    prefix = in_file[:-4]
    file_extension = in_file[-4:]

    valid_extensions = ['.jpg','.png']
    if file_extension not in valid_extensions:
        print('Invalid Filename')
        exit()

    # Output filenames
    seam_outfile = f"{prefix}_seam{file_extension}"
    final_outfile = f"{prefix}_final{file_extension}"
    
    
    # Reads in image in color
    color_img = cv2.imread(in_file).astype(np.float32)
    height = color_img.shape[0]
    width = color_img.shape[1]
    
    # Number of seams to remove
    n_iterations = np.abs(width - height) 
    
    # Stores energies of each seam
    energies = []
    
    # If image is tall, flip to remove horizontal seams
    flip = (height > width)
    if flip:
        color_img = np.transpose(color_img,(1,0,2))
                                 
    

    #Iterates until image is square
    for itr in range(n_iterations):

        # Find Seam
        seam = findSeam(color_img)

        # If on the first iteration, colors the seam red and prints the image
        if itr == 0:
            colorSeam(color_img,seam)
            if flip:
                flipped = np.transpose(color_img,(1,0,2))
                cv2.imwrite(seam_outfile,color_img)
            else:
                cv2.imwrite(seam_outfile,color_img)

        # Store energies
        energy = findEnergy(color_img,seam)
        energies.append(energy)

        # Removes the seam to create a reduced image (by one column)
        color_img = removeSeam(color_img,seam).astype(np.float32)
     
    if flip:
        color_img = np.transpose(color_img,(1,0,2))
        
    cv2.imwrite(final_outfile,color_img) 