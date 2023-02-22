
import cv2
import numpy as np
import math
import os
import random
import sys
from scipy.linalg import null_space
from sklearn.cluster import AgglomerativeClustering



# Folder to read from
img_folder = 'data'
# Takes in a command line argument with the name of the file to write images to
out_folder = sys.argv[1]


# Reads each image in the folder and sorts alphabetically
data = []
# Loops over input directory and stores all images 
for name in os.listdir(img_folder):
    # Checks that images are of a proper file type
    if(name == '.DS_Store'):
        continue
    # Reads in each image in color and grayscale, and stores both versions 
    color_img = cv2.imread(os.path.join(img_folder,name))
    img = cv2.imread(os.path.join(img_folder,name),cv2.IMREAD_GRAYSCALE)
    # Smooths the grayscale image for calculations
    img_smoothed = cv2.GaussianBlur(img.astype(np.float32), (5,5), 1.0)

    # Stores the images and the corresponding file names
    data.append((color_img,img_smoothed,name))


# Sorts for consistency
data.sort(key=lambda tup: tup[2])
num_images = len(data)


# Changes writing directory to store all produced images
current = os.getcwd()
os.chdir(current + '/' + out_folder)


# Initialize the SIFT algorithm
sift_alg = cv2.SIFT_create()
# Main algorithm that compares the images and produces the mosaic results
bf = cv2.BFMatcher()
# Loops over each pair of images
for img_number in range(0,num_images,2):


    # Stores the data of both images and the file extension
    (color_img1,img1,name1) = data[img_number]
    (color_img2,img2,name2) = data[img_number + 1]
    file_extension = name1[-4:]

    print(f"Now comparing images {name1} and {name2}")

    # Converts image to uint8
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
        
    #Keypoints and descriptors for image 1
    keyp1,desc1 = sift_alg.detectAndCompute(img1,None)
    keyp1 = np.array(keyp1)
    #Keypoints and descriptors for image 2
    keyp2,desc2 = sift_alg.detectAndCompute(img2,None)
    keyp2 = np.array(keyp2)



    # Finds two closest matches for each keypoint with k-nearest neightbors using both image descriptors
    bf_matches = bf.knnMatch(desc1,desc2,k=2)

    # Applies the ratio test
    matches = []
    threshold = 0.70
    for (m,n) in bf_matches:
        if m.distance < threshold * n.distance:
            matches.append(m)


    # Converts the ratio test matches into locations in each image
    img1_points = np.array([keyp1[match.queryIdx].pt for match in matches])
    img2_points = np.array([keyp2[match.trainIdx].pt for match in matches])

    # Calculates the flow vectors
    flow_vectors = img2_points - img1_points
        

    num_points = flow_vectors.shape[0]
    print(f"Total number of flow points considered: {num_points}")


    # Compute implicit lines

    motion_lines = np.zeros((num_points,3))

    x = img2_points[:,0]
    y = img2_points[:,1]
    u = flow_vectors[:,0]
    v = flow_vectors[:,1]

    # Normalization factor
    normalizer = u * u + v * v
    normalizer = np.sqrt(normalizer)

    # Calculates the implicit line parameters
    motion_lines[:,0] = - v / normalizer
    motion_lines[:,1] = u / normalizer
    c = v * x - u * y
    c = c / normalizer
    motion_lines[:,2] = c

    # Asserts that for all implicit lines the c value is negative
    motion_lines[c > 0] = - motion_lines[c > 0]


    # Parameters for RANSAC
    FOE = None
    best_inliers1 = None
    best_outliers1 = None
    best_inliers2 = None
    best_outliers2 = None
    best_flows = None
    most_inliers = -1
    NUM_ITERATIONS = 200

    for itr in range(NUM_ITERATIONS):

        sample = np.random.randint(0,num_points,2)
        if(sample[0] == sample[1]):
            continue

        index1 = sample[0]
        index2 = sample[1]

        # Calculates the focus of expansion
        foe_matrix = np.concatenate((motion_lines[index1],motion_lines[index2])).reshape((2,3))
        focus = null_space(foe_matrix)

        # If the result found was not a singular point then skip to next iteration
        if(focus.shape[1] > 1):
                continue

        # Converts from homogeneous coordinates back to image locations
        focus = focus / focus[2]


        # Compute error of all motion lines
        distances = np.matmul(motion_lines,focus)
        distances = np.abs(distances)
        distances = np.repeat(distances,2,axis=1)

        # Calculates the points that are inliers to the estimated FOE
        INLIER_THRESHOLD = 8.0
        inliers = img1_points[distances < INLIER_THRESHOLD].reshape(-1,2)
        num_inliers = inliers.shape[0]
            
        if num_inliers > most_inliers:
            
            FOE = focus
           
            best_inliers1 = inliers
            best_outliers1 = img1_points[ distances >= INLIER_THRESHOLD].reshape(-1,2)
            
            best_inliers2 = img2_points[distances < INLIER_THRESHOLD].reshape(-1,2)
            best_outliers2 = img2_points[ distances >= INLIER_THRESHOLD].reshape(-1,2)


            most_inliers = num_inliers


    # Stores the coordinates of the focus of expansion
    x_FOE = int(FOE[0])
    y_FOE = int(FOE[1])
    print(f"Found FOE at ({x_FOE},{y_FOE})")


    # Checks that the calculated FOE is within the range of the image dimensions
    VALID_FOE = True
    if(x_FOE < 0 or x_FOE > img2.shape[1] or y_FOE < 0 or y_FOE > img2.shape[0]):
        print("FOE is OUTSIDE of the image")
        VALID_FOE = False

    print(f"Number of inliers to the FOE: {most_inliers}")

    # Creates a copy of image 2 that will be used to create the output image
    out_img1 = np.copy(color_img2)

    # Draw flow vectors
    for i in range(num_points):

        x1 = int(img1_points[i,0])
        y1 = int(img1_points[i,1])
        x2 = int(img2_points[i,0])
        y2 = int(img2_points[i,1])

        p1 = (x1,y1)
        p2 = (x2,y2)
        out_img1 = cv2.line(out_img1,p1,p2,(255,0,0),2)
        

    # Check minimum number of inliers met
    MOVING_THRESHOLD = 100
    MOVING = True
    if(most_inliers < MOVING_THRESHOLD):
        print("Insufficient Inliers Found to FOE: No Camera Motion Detected")
        MOVING = False
            

    # Add FOE to image
    if VALID_FOE and MOVING:

        # 10x10 green square to denote FOE
        region = np.zeros((10,10,3))
        region[:,:,1] = 255

        # Boundary coordinates
        min_x = max(0,x_FOE - 5)
        min_y = max(0,y_FOE - 5)
        max_x = min(out_img1.shape[1],x_FOE + 5)
        max_y = min(out_img1.shape[0],y_FOE + 5)

        out_img1[min_y: max_y, min_x : max_x] = region

    # Write flow image
    out_name = name2 + '_FOE' + file_extension
    cv2.imwrite(out_name,out_img1)


    # Flow vectors for outlier points
    outlier_flows = best_outliers2 - best_outliers1

    # Use FOE to eliminate the noisy data points that are super outliers 
    if MOVING:
            
        # Compute normalized flows
        norm = np.linalg.norm(outlier_flows,axis=1)
        normalized_flows = outlier_flows / norm

        # Finds the vector from every outlier point to the focus of expansion
        FOE_point = np.array([x_FOE,y_FOE]).reshape((1,2))
        FOE_vectors = best_outliers2 - FOE_point

        # Normalizes each of the vectors 
        norm = np.linalg.norm(FOE_vectors,axis=1)
        FOE_vectors = FOE_vectors / norm


        #At this point, there are normalized vectors representing the flows and the vectors from every outlier
        #point to the focus of expansion. Taking the dot product of these vectors gives the cosine measure between the
        #two vectors. From this, we can eliminate all points that point in very different directions, since these 
        #points are likely noise

        # Stores the dot product values
        dots = np.sum(normalized_flows * FOE_vectors,axis=1)
        dots = np.abs(dots)

        # Eliminate points where the flow is over 30 degrees
        dots = np.abs(dots)
        outlier_flows = outlier_flows[dots > 0.86]
        best_outliers2 = best_outliers2[dots > 0.86]



    # Creates a copy of previous output image
    out_img2 = np.copy(out_img1)
    height = out_img2.shape[0]
    width = out_img2.shape[1]

    # Distance threshold
    DISTANCE_THRESHOLD = 600
    # Minimum number of points in a cluster
    CLUSTER_THRESHOLD = 10  

    # Minimum average flow magnitude of a cluster 
    MOVEMENT_THRESHOLD = 10.0

    # Number of independently moving objects detected
    moving_objects = 0


    cluster_coordinates = np.copy(best_outliers2)
    cluster_flows = np.copy(outlier_flows)
    minimum = np.amin(cluster_flows,axis=0)
    maximum = np.amax(cluster_flows,axis=0)

    # Compute scale factor
    scale = (maximum - minimum)
    scale[0] = scale[0] / width 
    scale[1] = scale[1] / height 

    # Converts flow coordinates to proper range
    cluster_flows = (cluster_flows - minimum) / scale

    # Stores the outlier points and flows together
    cluster_data = np.concatenate((cluster_coordinates,cluster_flows),axis=1)

    # Agglomerative clustering
    clusters = AgglomerativeClustering(None, distance_threshold=DISTANCE_THRESHOLD)
    clusters.fit_predict(cluster_data)
    cluster_labels = np.array(clusters.labels_)


    num_clusters = clusters.n_clusters_

    # Loops over each cluster
    for i in range(num_clusters):


        # Generate a random color tuple
        colors = np.random.randint(0,256,3)
        color = (int(colors[0]),int(colors[1]),int(colors[2]))

        # Finds all of the points in the current cluster
        C = best_outliers2[cluster_labels == i]

        if(C.shape[0] < CLUSTER_THRESHOLD):
            continue

        # Finds the motion vector at each point in the cluster
        motion = outlier_flows[cluster_labels == i]

        # Finds the average magnitude of the flow in the cluster
        motion_magnitudes = np.linalg.norm(motion,axis=1)
        average_motion = np.average(motion_magnitudes)

        
        if average_motion < MOVEMENT_THRESHOLD:
            continue
              
        # If the cluster passes both thresholds, then  it is determined to be an independent object

        # Compute the bounding box of the cluster 
        ul = np.amin(C,axis=0)
        lr = np.amax(C,axis=0)
        upper_left = (int(ul[0]),int(ul[1]))
        lower_right = (int(lr[0]),int(lr[1]))

        out_img2 = cv2.rectangle(out_img2,upper_left,lower_right,color,2)
        moving_objects += 1

    print(f"Number of moving objects detected: {moving_objects}")
    print()

    # Write final image
    out_name = name2 + '_Clusters' + file_extension
    cv2.imwrite(out_name,out_img2)


        



      

        