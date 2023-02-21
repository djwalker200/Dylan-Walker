import cv2
import numpy as np

import math
import os
import random
import sys

# Creates the matching image between two images given their keypoints and the set of matches that is to be drawn
def printMatchImage(image1,keyp1,image2,keyp2,matches,output_file):
    
    out_img = np.array([])
    cv2.drawMatches(np.uint8(image1), keyp1, np.uint8(image2), keyp2, matches,outImg = out_img)
    cv2.imwrite(output_file,out_img)

# Finds the mapped locations of all 4 corners of an image with dimensions (height x width) under the projection Matrix H
# Returns a numpy array with projected corner locations in the columns and with respect to the original image,
# Stored in the order [upper-left,upper-right,lower-left,lower-right]
def findCorners(height,width,H):
    
    #Creates corner points to be mapped
    U = np.array([[0,0,1],[width,0,1],[0,height,1],[width,height,1]]).T
    
    # Maps each point and converts from homogeneous coordinates
    mapped_corners = np.matmul(H,U)
    mapped_corners[:,:] = mapped_corners[:,:] / mapped_corners[2,:]
    mapped_corners = mapped_corners[:2,:]
    mapped_corners = mapped_corners.T
    
    return mapped_corners

# Finds and returns the upper-left and lower-right corner locations of an image with dimensions (height x width)
 #Under the projection of matrix H
def findBoundingCorners(height,width,H):
    
    # Creates corner points to be mapped
    U = np.array([[0,0,1],[width,0,1],[0,height,1],[width,height,1]]).T
    
    # Maps each point and converts from homogeneous coordinates to affine
    mapped_corners = np.matmul(H,U)
    mapped_corners[:,:] = mapped_corners[:,:] / mapped_corners[2,:]
    # Stores the x and y coordinates for each projected corner
    xcoords = mapped_corners[0,:]
    ycoords = mapped_corners[1,:]
    # Finds coordinates for the upper-left corner of the bounding box
    minx = round(min(xcoords))
    miny = round(min(ycoords))
    # Finds coordinates for the lower-right corner of the bounding box
    maxx = round(max(xcoords))
    maxy = round(max(ycoords))
    # Returns a tuple of the upper-left and lower-right corner locations
    return((minx,miny),(maxx,maxy))

# Finds the matches that are inliers to the fundamental matrix F between two images
# Also produces and image of image 2 with the epipolar lines drawn for each inlier match
# Returns a list of matches that are inliers to the fundamental matrix

def findInliers(locations1,locations2,matches,F,output_file,color_img):
    
    size = len(matches)
    
    # Convert matches to homogenous coordinates
    ones = np.ones(size).reshape((size,1))
    
    Ui = np.array(locations1).reshape((size,2))
    Ui = np.concatenate((Ui,ones),axis=1)
    Ui = Ui.T
    
    Uj = np.array(locations2).reshape((size,2))
    Uj = np.concatenate((Uj,ones),axis=1)


    #Multiplies F to each of the homogeneous coordinates of image 1, and the resulting matrix A 
    #Stores the implicit forms of each epipolar line
    A = np.matmul(F,Ui)

    #Makes all c terms negative for ax + by + c for each implicit line
    A = -1 * A * (A[2,:] / abs(A[2,:]))
    # Normalize values
    A = A / np.linalg.norm(A[:2,:],axis=0)

    #Creates a color copy of image 2 to be used for the epipolar line image
    img2_lines = np.copy(color_img)
    width = img2.shape[1]
    
    
    #Stores all inliers that pass the thresholding test
    inliers = []
    #Euclidean distance threshold between the epipolar lines and keypoints
    distance_threshold = 2.0
    
    #Loops over each of the keypoints in image 2 
    for k in range(size):
        
        #Calculates the distance |ax + by - c| 
        distance = abs(np.matmul(Uj[k,:],A[:,k]))
        
        
        if(distance < distance_threshold):
            
            inliers.append(matches[k])
            #Stores the current line parameters
            line = A[:,k]
            a = line[0]
            b = line[1]
            c = line[2]
            #Sets coordinates for line endpoints
            #x0 is the left edge of the image
            x0 = int(0)
            #If x0 = 0 then y0 = -c / b
            y0 = int(-c / b)
            #x1 is the right edge of the image
            x1 = int(width)
            #If x1 = w then y1 = -(c + aw)/b
            y1 = int(-(c + a * width) / b )
            #Generates a random color
            color = tuple(np.random.randint(0,255,3).tolist())
            #Adds line to image
            img2_lines = cv2.line(img2_lines, (x0,y0), (x1,y1), color,1)

    #Writes the epipolar line image
    cv2.imwrite(output_file,img2_lines)

    #Returns a list of all inliers to the fundamental matrix
    return inliers


# Given a set of matches that are inliers to the fundamental matrix, and their corresponding pixel locations,
# finds which matches are inliers to the homography matrix 
# Returns a list of matches that are inliers to homography matrix
def findHomographyInliers(locations1,locations2,matches,H):
    
    # Converts to homogenous coordinates
    ones = np.ones(size).reshape(size,1)

    Ui = np.array(inlier_locations1).reshape(size,2)
    Ui = np.concatenate((Ui,ones),axis=1)
    Ui = Ui.T
    
    Uj = np.array(inlier_locations2).reshape((size,2))
    Uj = np.concatenate((Uj,ones),axis=1)
    Uj = Uj.T

    # Map image 1 points into image 2 space
    Mapped_points = np.matmul(H,Ui)
    Mapped_points = Mapped_points / Mapped_points[2,:]
    
    # Finds the distance between pairs in image 2 space
    distances = np.linalg.norm(Mapped_points - Uj,axis=0)


    # Distance threshold
    homography_distance_threshold = 3.0
    
    # Stores the matches that are inliers to the homography matrix
    homography_inliers = []
  
    for k in range(len(matches)):
        if(distances[k] < homography_distance_threshold):
            homography_inliers.append(matches[k])
    
    return homography_inliers

if __name__ == "__main__":
    
    # Input and output image folders
    img_folder = sys.argv[1]
    out_folder = sys.argv[2]

    # Reads each image in the input folder and sorts alphabetically
    data = []
    for img_name in os.listdir(img_folder):
        #Checks that images are of a proper file type
        if(img_name == '.DS_Store'):
            continue
            
        #Reads in each image in color and grayscale, and stores both versions 
        color_img = cv2.imread(os.path.join(img_folder,img_name))
        img = cv2.imread(os.path.join(img_folder,img_name),cv2.IMREAD_GRAYSCALE)
        
        #Smooths the grayscale image for calculations
        img_smoothed = cv2.GaussianBlur(img.astype(np.float32), (5,5), 1.0)
        
        #Stores the images and the corresponding file names
        data.append((color_img,img_smoothed,img_name))
     
    num_images = len(data)
    
    # Optional command line argument for multi-image mosaics
    # A matrix that stores which images have mosaic matches to the others

    multi_images = False
    if(len(sys.argv) == 4 and len(images) > 2):
        if(sys.argv[3] == 'multi'):
            multi_images = True
            
    mosaic_matches = np.zeros((num_images,num_images))
    mosaic_matches_homographies = np.zeros((num_images,num_images,3,3))
    
    
    # Sorts each list for consistency
    data.sort(key=lambda tup: tup[2])

    # Changes writing directory to store all produced images
    current = os.getcwd()
    os.chdir(current + '/' + out_folder)


    # Stores the keypoint and descriptor vectors for each of the images
    keypoints = []
    descriptors = []
    
    #Initializes the SIFT algorithm
    sift_alg = cv2.SIFT_create()
    
    # Finds keypoints and descriptors for each image
    for index in range(num_images):
        
        smoothed_image = data[index][1]
        sift_kp,sift_descriptors = sift_alg.detectAndCompute(smoothed_image.astype(np.uint8),None)

        keypoints.append(sift_kp)
        descriptors.append(sift_descriptors)
        
        # Prints the number of keypoints detected for each image
        #print(len(sift_kp), "keypoints detected in",names[index])

    #Prints an extra line for spacing and readability purposes
    #print()

    # Main algorithm that compares the images and produces the mosaic results
    bf = cv2.BFMatcher()
    
    # Loops over each image pair
    for i in range(num_images):
        for j in range(num_images):
            
            # Avoid self-comparison
            if i == j:
                continue
                
            # Stores the data for both images
            (color_img1,img1,name1) = data[i]
            (color_img2,img2,name2) = data[j]
            file_extension = name1[-4:]
            
            # Keypoints and descriptors 
            keyp1 = np.array(keypoints[i])
            desc1 = descriptors[i]
            keyp2 = np.array(keypoints[j])
            desc2 = descriptors[j]

            # Finds two closest matches for each keypoint with k-nearest neightbors using both image descriptors
            bf_matches = bf.knnMatch(desc1,desc2,k=2)
            
            # Stores the strong matches
            matches = []
            
            # Applies the ratio test
            threshold = 0.80
            for (m,n) in bf_matches:
                if m.distance < threshold * n.distance:
                    matches.append(m)

            #Prints the fraction of matches for each image
            fraction1 = len(matches) / len(keyp1)
            fraction2 = len(matches) / len(keyp2)
            print('Number of matches between',name1,'and',name2,':',len(matches))
            print('Fraction of matches for',name1,'%.3f' % fraction1)
            print('Fraction of matches for',name2,'%.3f' % fraction2)

            # Prints the side-by-side image with the matches that passed the ratio test drawn
            printMatchImage(color_img1,keyp1,color_img2,keyp2,matches,name1 + '_' + name2 + '_matched' + file_extension)


            # Threshold for the fraction of keypoints that must have matches in both images
            FRACTION_THRESHOLD = 0.005
            if(fraction1 < FRACTION_THRESHOLD or fraction2 < FRACTION_THRESHOLD):
                print()
                print(name1,"does not match",name2,"because of fraction of good matches found")
                continue

            # Threshold for the minimum number of matches that are required 
            ABSOLUTE_THRESHOLD = 50
            if(len(matches) < ABSOLUTE_THRESHOLD):
                print()
                print(name1,"does not match",name2,"because of number of good matches found")
                print()
                continue

            # Prints a statement signifying that significant matches were found
            print()
            print('Continuing to fundamental matrix step because sufficient matches were found')
            print()

            # Converts the ratio test matches into locations in each image
            match_locations1 = np.array([keyp1[match.queryIdx].pt for match in matches])  
            match_locations2 = np.array([keyp2[match.trainIdx].pt for match in matches])


            # Estimate fundamental matrix
            F,mask = cv2.findFundamentalMat(match_locations1,match_locations2,cv2.FM_RANSAC)

            # Calculates the inlier matches for the fundamental matrix F and produces epipolar line image
            output_file = name1 + '_' + name2 + '_lines' + file_extension
            inliers = findInliers(match_locations1,match_locations2,matches,F,output_file,color_img2)
            
            # Prints the number and fraction of inliers from the fundamental matrix step
            fraction = len(inliers) / len(matches)
            print('Number of fundamental matrix inliers:',len(inliers))
            print('Fraction of fundamental matrix inliers:','%.3f' % fraction)
            print()

            # Prints the side-by-side image with the fundamental matrix inliers drawn
            printMatchImage(color_img1,keyp1,color_img2,keyp2,inliers,name1 + '_' + name2 + '_finliers' + file_extension)

            # Threshold for the minimum fraction of inliers
            INLIER_FRACTION_THRESHOLD = 0.25
            if(fraction < INLIER_FRACTION_THRESHOLD):
                print()           
                print(name1,"does not match",name2,"because of fraction of fundamental matrix inliers")
                print()
                continue

            # Threshold for the minimum number of inliers
            INLIER_ABSOLUTE_THRESHOLD = 25
            if(len(inliers) < INLIER_ABSOLUTE_THRESHOLD ):
                print()
                print(name1,"does not match",name2,"because of number of fundamental matrix inliers")
                print()
                continue

            # Prints a statement signifying that significant fundamental matrix inliers were found
            print()
            print('Continuing to homography matrix step because sufficient inliers were found')
            print()

            #Converts the inliers to image locations
            inlier_locations1 = np.array([keyp1[match.queryIdx].pt for match in inliers])
            inlier_locations2 = np.array([keyp2[match.trainIdx].pt for match in inliers])


            # Estimates the homography matrix 
            H,mask = cv2.findHomography(inlier_locations1,inlier_locations2,cv2.RANSAC)
            
            #Finds the matches that are inliers to the homography matrix
            homography_inliers = findHomographyInliers(inlier_locations1,inlier_locations2,inliers,H)

            # Prints the number and fraction of matches that remained as inliers
            fraction = len(homography_inliers) / len(inliers)
            print('Number of homography inliers:',len(homography_inliers))
            print('Fraction of homography inliers:','%.3f' %  fraction)
            print()

            # Prints the side-by-side image with the homography matrix inliers drawn
            printMatchImage(color_img1,keyp1,color_img2,keyp2,homography_inliers,name1 + '_' + name2 + '_hinliers' + file_extension)

            # Threshold for the minimum fraction of inliers 
            HOMOGRAPHY_FRACTION_THRESHOLD = 0.10
            if(fraction < HOMOGRAPHY_FRACTION_THRESHOLD):
                print()
                print(name1,"does not match",name2,"because of fraction of homography inliers")
                print()
                continue

            # Threshold for the minimum number of matches
            HOMOGRAPHY_ABSOLUTE_THRESHOLD = 10
            if(len(homography_inliers) < HOMOGRAPHY_ABSOLUTE_THRESHOLD) :
                print()
                print(name1,"does not match",name2,"because of number of homography inliers")
                print()
                continue


            print()
            print('Forming mosaic because sufficient homography inliers were found')
            print()

            #Finds the upper-left and lower-right corner locations of the bounding box 
            #formed by mapping image 1 into image 2 via H
            upper_left,lower_right = findBoundingCorners(img1.shape[0],img1.shape[1],H)

            # Upper-left corner location of final image in image 2 space
            upper_x = min(upper_left[0],0)
            upper_y = min(upper_left[1],0)
             
            # Lower-right corner location of final image in image 2 space
            lower_x = max(lower_right[0],img2.shape[1])
            lower_y = max(lower_right[1],img2.shape[0])
            
            # Calculates the dimensions of the final image
            final_height = lower_y - upper_y
            final_width = lower_x - upper_x
            
            # Creates a completely black final image with desired dimensions
            final_img = np.tile([0,0,0],(final_height,final_width,1))

            # Offsets to convert between image 2 space and the final image space
            x_offset = upper_x
            y_offset = upper_y

            # Boundary coordiantes
            x0 = 0
            y0 = 0
            x1 = final_width
            y1 = final_height
            
            # Finds the location of the upper left corner of image 2 in the final image space
            if(x_offset < 0):
                x0 = -1 * x_offset
            if(y_offset < 0):
                y0 = -1 * y_offset

             #Finds the upper bound of the coordinates of image 2 in the final image space
            if(lower_x > img2.shape[1]):
                x1 = img2.shape[1] + x0

            if(lower_y > img2.shape[0]):
                y1 = img2.shape[0] + y0
 
            # Maps image 2 into the final image space
            final_img[y0:y1,x0:x1] = color_img2


            # Stores the height and width of image 1
            h = img1.shape[0]
            w = img1.shape[1]
            
            # Finds the mapped locations of all 4 corners of image 1 into final image space
            corners = findCorners(h,w,H)
            corners[:,0] = corners[:,0]  - x_offset
            corners[:,1] = corners[:,1] - y_offset
            
            # Stores the corner locations in the final image space
            original = np.array([[0,0],[w,0],[0,h],[w,h]],dtype=np.float32)
            
            # Finds the perspective transformation between image 1 and the final image space
            M = cv2.getPerspectiveTransform(original,corners.astype(np.float32))
            
            # Maps image 1 into the final image space
            mapped_img = cv2.warpPerspective(color_img1,M,(final_width,final_height),flags=cv2.INTER_LINEAR)
            
            # Copy of final image before any pixels from image 1 have been added
            final_copy = np.copy(final_img)
            
            # Map images onto final image
            final_img[mapped_img != 0] = 0.5 * (mapped_img[mapped_img != 0] + final_img[mapped_img != 0])
            final_img[final_copy == 0] = mapped_img[final_copy == 0]
 
            # Write final image
            output_file = name1 + '_' + name2 + file_extension
            cv2.imwrite(out_name,final_img)
            print('Wrote mosaic image:',out_name)
            print()
            mosaic_matches[j,i] = 1
            mosaic_matches_homographies[j,i,:,:] = H

        
       
    #If the multi-image command line is passed in, produces the first multi image mosaic possible
    if(multi_images):
        
        for i in range(num_images):

            row = mosaic_matches[i,:]
            number_matches = row[row > 0].shape[0]
            if(number_matches < 2):
                continue

            color_anchor,anchor,anchor_name = data[i]
            output_file = anchor_name[:-4]
            
            imgs = []
            color_imgs = []
            Homographies = []
            for j in range(num_images):
                if row[j]:
                    color_img, img, name = data[j]
                    imgs.append(img)
                    color_imgs.append(color_img)
                    Homographies.append(mosaic_matches_homographies[i,j,:,:])
                    output_file += '_' + name[:-4]

            output_file +=  file_extension
                    
                
            upper_x = 0
            upper_y = 0
            lower_x = anchor.shape[1]
            lower_y = anchor.shape[0]

            for j in range(len(imgs)):
                img = imgs[j]
                H = Homographies[j]
                upper_left,lower_right = findBoundingCorners(img.shape[0],img.shape[1],H)
                
                #Upper-left corner location of final image 
                upper_x = min(upper_left[0],upper_x)
                upper_y = min(upper_left[1],upper_y)
                
                #Lower-right corner location of final image
                lower_x = max(lower_right[0],lower_x)
                lower_y = max(lower_right[1],lower_y)

            final_height = lower_y - upper_y
            final_width = lower_x - upper_x
            
            #Creates a completely black final image with desired dimensions
            final_img = np.tile([0,0,0],(final_height,final_width,1))

            # Offsets to convert between image spaces
            x_offset = upper_x
            y_offset = upper_y

            # Boundary Coordinates
            x0 = 0
            y0 = 0
            x1 = final_width
            y1 = final_height
            
            #Finds the location of the upper left corner of projected image
            if(x_offset < 0):
                x0 = -1 * x_offset
            if(y_offset < 0):
                y0 = -1 * y_offset

            # Finds the upper bound of the coordinates of projected image
            if(lower_x > img2.shape[1]):
                x1 = img2.shape[1] + x0

            if(lower_y > img2.shape[0]):
                y1 = img2.shape[0] + y0
 
            # Maps anchor into the final image space
            final_img[y0:y1,x0:x1] = color_anchor
            factor = 1.0 / len(imgs)
            
            for j in range(len(imgs)):
                
                img = imgs[j]
                color_img = color_imgs[j]
                H = Homographies[j]
                h = img.shape[0]
                w = img.shape[1]
                
                # Finds the mapped locations of all 4 corners
                corners = findCorners(h,w,H)
                
                # Convert coordinates to the final image space
                corners[:,0] = corners[:,0]  - x_offset
                corners[:,1] = corners[:,1] - y_offset
                
                # Finds the perspective transformation between image spaces
                original = np.array([[0,0],[w,0],[0,h],[w,h]],dtype=np.float32)
                M = cv2.getPerspectiveTransform(original,corners.astype(np.float32))
                # Maps image into the final image space
                mapped_img = cv2.warpPerspective(color_img,M,(final_width,final_height),flags=cv2.INTER_LINEAR)
            
                # Copy of final image before any pixels from image 1 have been added
                final_copy = np.copy(final_img)
                
                # Map image into final image space
                final_img[mapped_img != 0] = 0.5 * (mapped_img[mapped_img != 0] + final_img[mapped_img != 0])
                final_img[final_copy == 0] = mapped_img[final_copy == 0]
               
            # Write the final mosaic image
            cv2.imwrite(output_file,final_img)
            print('Wrote mulit-image mosaic' , output_file)
            
            


    


            



