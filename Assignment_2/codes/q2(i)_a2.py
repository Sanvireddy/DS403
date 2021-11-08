# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:22:37 2021

@author: jahna
"""
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
  
#Reading the image 
image = cv2.imread(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg2\Image.jpg") 
  
#Change color from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
  

#3D to 2D ==>each row would now be a 3D spaced of RGB Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3)) 
  
#Convert to float type inorder for kmeans to read in opencv 
pixel_vals = np.float32(pixel_vals)


#the below line of code defines the criteria for the algorithm to stop running,  
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)  
#becomes 85% 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
  
plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()
fig = plt.figure(figsize=(25,25))
rows,columns = 1, 4
# then perform k-means clustering with number of clusters defined as 8
#also random centres are initally chosed for k-means clustering
K = [2, 4, 8, 15]
n = 1
print("K-MEANS CLUSTERING-BASED SEGMENTATION OF IMAGE ")
print("2(i)WHEN USING ONLY PIXEL COLOR AS FEATURES")
for k in K:
    attempts = 10
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS) 
  
    # convert data into 8-bit values 
    centers = np.uint8(centers) 

    #regenerate the image
    segmented_data = centers[labels.flatten()] 
  
    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((image.shape)) 

    #plotting and comparing original and segmented image
    
    fig.add_subplot(rows, columns, n)
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
    n += 1
plt.show()