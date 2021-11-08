# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:52:25 2021

@author: jahna
"""
#################### CLASS - 1 (Linearly separable data) #################
#importing the required modules
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#reading the text file
ls1=open(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\ls_data\class1.txt")
ls2=open(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\ls_data\class2.txt")
ls_c1 = ls1.readlines()
ls_c2 = ls2.readlines()

#f1_c1 represents feature 1 of class 1
#d2_c2 represents feature 2 of class 2
f1_c1=[]
f2_c1=[]
f1_c2=[]
f2_c2=[]

# we are appending the data given in text fie to respective lists
for x in ls_c1:
    b = x.split(",")
    f1_c1.append(b[0])
    f2_c1.append(b[1])
    
# taking float values in list
f1_c1 = list(map(float,f1_c1))
f2_c1=list(map(float,f2_c1))
for y in ls_c2:
    a = y.split(",")
    f1_c2.append(a[0])
    f2_c2.append(a[1])
    
    
f1_c2 = list(map(float,f1_c2))
f2_c2=list(map(float,f2_c2))

# converting our list to data frame
df1=pd.DataFrame(list(zip(f1_c1,f2_c1)),columns=['x1','y1'])
df2=pd.DataFrame(list(zip(f1_c2,f2_c2)),columns=['x2','y2'])


# splitting data into train and test data 
xt1,xtt1=train_test_split(df1,test_size=0.2,random_state=42)
xt2,xtt2=train_test_split(df2,test_size=0.2,random_state=42)
 
# performing min-max normalisation for class 1
a1 = xt1['x1'].max()
b1 =xt1['x1'].min()
a2 =xt1['y1'].max()
b2 =xt1['y1'].min()
xt1['x1'] = (xt1['x1']-b1)/(a1-b1)
xt1['y1'] = (xt1['y1']-b2)/(a2-b2)


# since the features are independent we can multiply the probalities of 
# two different attributes to get the probability of the class

pb_cl1 = [0]*len(xt1)
xt1['Overall']=pb_cl1
   
xt1['Overall'] = xt1['x1']*xt1['y1']

# plotting the histogram
p=xt1['Overall']
p.hist()
plt.title("Histogram of class 1 (Train data)")
plt.xlabel('X-axis - class 1(LS)')
plt.ylabel('Y-axis - frequency')
plt.show()

# calculating the probabilites of test data 
xtt1['x1'] = (xtt1['x1']-b1)/(a1-b1)
xtt1['y1']= (xtt1['y1']-b2)/(a2-b2)

# since the data is independent
pb_cl1 = [0]*len(xtt1)
xtt1['Overall']=pb_cl1

  
# plotting the histogram
xtt1['Overall'] = xtt1['x1']*xtt1['y1']
p=xtt1['Overall']
p.hist()
plt.title("Histogram of class 1 (test data)")
plt.xlabel('X-axis - class 1(LS)')
plt.ylabel('Y-axis - frequency')
plt.show()


#################### CLASS - 2 (Linearly separable data) #################

# =performing min-max normalisation for each of the features of class 2
# so that the data lies between 0 and 1
a1 = xt2['x2'].max()
b1 =xt2['x2'].min()
a2 =xt2['y2'].max()
b2 =xt2['y2'].min()
xt2['x2'] = (xt2['x2']-b1)/(a1-b1)
xt2['y2'] = (xt2['y2']-b2)/(a2-b2)


# since the features are independent we can multiply the probalities of 
# two different attributes to get the probability of the class

pb_cl2 = [0]*len(xt2)
xt2['Overall']=pb_cl2
    
xt2['Overall'] = xt2['x2']*xt2['y2']

# plotting the histogram
p=xt2['Overall']
p.hist()
plt.title("Histogram of class 2 (train data) ")
plt.xlabel('X-axis - class 2 (LS)')
plt.ylabel('Y-axis - frequency')
plt.show()

# calculating the probabilites of test data 
xtt2['x2'] = (xtt2['x2']-b1)/(a1-b1)
xtt2['y2']= (xtt2['y2']-b2)/(a2-b2)


pb_cl2 = [0]*len(xtt2)
xtt2['Overall']=pb_cl2

  
xtt2['Overall'] = xtt2['x2']*xtt2['y2']
p=xtt2['Overall']
p.hist()
plt.title("Histogram of class 2(test data)")
plt.xlabel('X-axis - class 2(LS)')
plt.ylabel('Y-axis - frequency')
plt.show()


############################  NON -LINEARLY SEPARABLE DATA ######################




#################### CLASS - 1 (Non linearly separable data) #################
#reading the text file
nls1=open(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class1.txt")
nls2=open(r"C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class2.txt")

nls_c1 = nls1.readlines()
nls_c2 = nls2.readlines()

#f1_c1 represents feature 1 of class 1
#f2_c2 represents feature 2 of class 2
f1_nc1=[]
f2_nc1=[]
f1_nc2=[]
f2_nc2=[]

# we are appending the data given in text fie to respective lists
for x in nls_c1:
    nb = x.split(",")
    f1_nc1.append(nb[0])
    f2_nc1.append(nb[1])
    
# taking float values in list
f1_nc1 = list(map(float,f1_nc1))
f2_nc1=list(map(float,f2_nc1))
for y in nls_c2:
    na = y.split(",")
    f1_nc2.append(na[0])
    f2_nc2.append(na[1])
    
    
f1_nc2 = list(map(float,f1_nc2))
f2_nc2=list(map(float,f2_nc2))

# converting our list to data frame
ndf1=pd.DataFrame(list(zip(f1_nc1,f2_nc1)),columns=['x1','y1'])
ndf2=pd.DataFrame(list(zip(f1_nc2,f2_nc2)),columns=['x2','y2'])


# converting our data into train and test data 
nxt1,nxtt1=train_test_split(ndf1,test_size=0.2,random_state=42)
nxt2,nxtt2=train_test_split(ndf2,test_size=0.2,random_state=42)
 
# doibg min-max normalisation
na1 = nxt1['x1'].max()
nb1 =nxt1['x1'].min()
na2 =nxt1['y1'].max()
nb2 =nxt1['y1'].min()
nxt1['x1'] = (nxt1['x1']-nb1)/(na1-nb1)
nxt1['y1'] = (nxt1['y1']-nb2)/(na2-nb2)


# since the features are independent we can multiply the probalities of 
# two different attributes to get the probability of the class

pb_cl1 = [0]*len(nxt1)
nxt1['Overall']=pb_cl1


    
nxt1['Overall'] = nxt1['x1']*nxt1['y1']

# plotting the histogram
np=nxt1['Overall']
np.hist()
plt.title("Histogram oF class 1(train data)")
plt.xlabel('X-axis - class 1 (NLS)')
plt.ylabel('Y-axis - frequency')
plt.show()

# calculating the probabilites of test data 
nxtt1['x1'] = (nxtt1['x1']-nb1)/(na1-nb1)
nxtt1['y1']= (nxtt1['y1']-nb2)/(na2-nb2)


pb_cl1 = [0]*len(nxtt1)
nxtt1['Overall']=pb_cl1

  
# since the data is independent
# histogram of class 1
# plotting the histogram
nxtt1['Overall'] = nxtt1['x1']*nxtt1['y1']
np=nxtt1['Overall']
np.hist()
plt.title("Histogram of class 1(test data)")
plt.xlabel('X-axis - class 1 (NLS)')
plt.ylabel('Y-axis - frequency')
plt.show()


#################### CLASS - 2 (Non linearly separable data) #################


# doibg min-max normalisation
na1 = nxt2['x2'].max()
nb1 =nxt2['x2'].min()
na2 =nxt2['y2'].max()
nb2 =nxt2['y2'].min()
nxt2['x2'] = (nxt2['x2']-nb1)/(na1-nb1)
nxt2['y2'] = (nxt2['y2']-nb2)/(na2-nb2)


# since the features are independent we can multiply the probalities of 
# two different attributes to get the probability of the class

pb_cl2 = [0]*len(nxt2)
nxt2['Overall']=pb_cl2
    
nxt2['Overall'] = nxt2['x2']*nxt2['y2']

# plotting the histogram
np=nxt2['Overall']
np.hist()
plt.title("Histogram oF class 2(train data)")
plt.xlabel('X-axis - class 2 (NLS)')
plt.ylabel('Y-axis - frequency')
plt.show()

# calculating the probabilites of test data 
nxtt2['x2'] = (nxtt2['x2']-nb1)/(na1-nb1)
nxtt2['y2']= (nxtt2['y2']-nb2)/(na2-nb2)


pb_cl2 = [0]*len(nxtt2)
nxtt2['Overall']=pb_cl2

  
nxtt2['Overall'] = nxtt2['x2']*nxtt2['y2']
np=nxtt2['Overall']
np.hist()
plt.title("Histogram of class 2(test data)")
plt.xlabel('X-axis - class 2 (NLS)')
plt.ylabel('Y-axis - frequency')
plt.show()