# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:07:41 2021

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

#taking the image's shape to create a array that contains both pixel values and locations
pixel_width, pixel_height, dim = image.shape
count =0
l = []
#to create a new array when we are using both pixel values and normalised locations(for detailed image, if not normalised we only get the image with colors and not clustered image)
for i in range(pixel_width):
    for j in range(pixel_height):
        pix1, pix2, pix3 = pixel_vals[count]
        #appending int of pix values and location
        l.append([int(pix1),int(pix2),int(pix3),int(i)/pixel_width,int(j)/pixel_height])
        count += 1

#list form of array converting to array
pixel_vals = np.array(l)

#Convert to float type inorder for kmeans to read in opencv 
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,  
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)  
#becomes 85% 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 


fig = plt.figure(figsize=(25,25))
rows,columns = 1, 4
# then perform k-means clustering with number of clusters defined as 8
#also random centres are initally chosed for k-means clustering
K = [2, 4, 8, 15]
n = 1
print("2(ii)WHEN USING BOTH PIXEL COLOR AND LOCATION VALUES AS FEATURES")
#both pixel colors and location values
for k in K:
    attempts = 10
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS) 
  
    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    
    #regenerate the image
    segmented_data = centers[labels.flatten()] 
    
    #kmeans clustering segmentation is based on both pixels and location
    #for image to reshape it only needs the pixel values and not the location
    #running for loop to eliminate location values
    l1=[]
    for i in segmented_data:
        l1.append(i[:-2])
    segmented_data=np.array(l1)
        
    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((image.shape)) 
    
    #plotting and comparing original and segmented image
    
    fig.add_subplot(rows, columns, n)
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
    n += 1
plt.show()
        