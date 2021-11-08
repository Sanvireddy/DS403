# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 17:18:07 2021

@author: jahna
"""
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

print("*****************************ls***************************")

f1 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\ls_data\class1.txt'
f2 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\ls_data\class2.txt'

cls1 = np.loadtxt(f1, delimiter=',')
cls2 = np.loadtxt(f2, delimiter=',')


cls1_label = [1]*len(cls1)
cls2_label = [2]*len(cls2)

cls1_train,cls1_test,cls1_label_train,cls1_label_test=train_test_split(cls1,cls1_label,test_size=0.2,random_state=42)
cls2_train,cls2_test,cls2_label_train,cls2_label_test=train_test_split(cls2,cls2_label,test_size=0.2,random_state=42)

test = np.concatenate((cls1_test, cls2_test))
train = np.concatenate((cls1_train, cls2_train))

for i in range(len(cls1)):
    plt.scatter(cls1[i][0],cls1[i][1], color = 'r')
    plt.scatter(cls2[i][0],cls2[i][1], color = 'b')
plt.title("Linearly Separable data", fontsize=15)
plt.show()
for i in range(len(train)):
    plt.scatter(train[i][0],train[i][1], color = 'm')
for i in range(len(test)):
    plt.scatter(test[i][0],test[i][1],color='k')
    
plt.title("Linearly Separable data", fontsize=15)
plt.show()

label_train=np.concatenate((cls1_label_train,cls2_label_train))
label_test=np.concatenate((cls1_label_test,cls2_label_test))

#mean vectors
mean1 = cls1.mean(axis=0)
mean2 = cls2.mean(axis=0)
#covariance vectors
covar1=np.cov(cls1.T)
covar2=(np.cov(cls2.T))

xpredl=[]
print(mean1, "is the mean of class1 of linearly separable data")
print(mean2, "is the mean of class2 of linearly separable data")
print("Covariance of class1 of linearly separable data")
print(covar1)
print("Covariance of class2 of linearly separable data")
print(covar2)

#decision
#prior
pcls1 = 0.5
pcls2 = 0.5
#since prior is equal for both the classes we use minimum distance classifier for classification
predontest = []
def euclidean_distance(x):
    dist1 = (((x[0]-mean1[0])**2)+((x[1]-mean1[1])**2))**0.5
    dist2 = (((x[0]-mean2[0])**2)+((x[1]-mean2[1])**2))**0.5
    return dist1, dist2
for i in range(len(test)):
    a,b = euclidean_distance(test[i])
    if a > b:
        predontest.append(2)
    elif a < b:
        predontest.append(1)
    

#confusion-matrix
print("Confusion Matrix after building bayes classifier is \n",confusion_matrix(label_test,predontest))




print("******************************nls***************************")




fnls1 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class1.txt'
fnls2 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\nls_data\class2.txt'

nlscls1 = np.loadtxt(fnls1, delimiter=',')
nlscls2 = np.loadtxt(fnls2, delimiter=',')


nlscls1_label = [1]*len(nlscls1)
nlscls2_label = [2]*len(nlscls2)

nlscls1_train,nlscls1_test,nlscls1_label_train,nlscls1_label_test=train_test_split(nlscls1,nlscls1_label,test_size=0.2,random_state=42)
nlscls2_train,nlscls2_test,nlscls2_label_train,nlscls2_label_test=train_test_split(nlscls2,nlscls2_label,test_size=0.2,random_state=42)

nlstest = np.concatenate((nlscls1_test, nlscls2_test))
nlstrain = np.concatenate((nlscls1_train, nlscls2_train))

for i in range(len(nlscls1)):
    plt.scatter(nlscls1[i][0],nlscls1[i][1], color='r')
    plt.scatter(nlscls2[i][0],nlscls2[i][1], color='b')
plt.title("Non-linearly Separable data", fontsize=15)
plt.show()

for i in range(len(nlstrain)):
    plt.scatter(nlstrain[i][0],nlstrain[i][1], color='m')
for i in range(len(nlstest)):
    plt.scatter(nlstest[i][0],nlstest[i][1], color='k')
plt.title("Non-linearly Separable data", fontsize=15)
plt.show()

nlslabel_train=np.concatenate((nlscls1_label_train,nlscls2_label_train))
nlslabel_test=np.concatenate((nlscls1_label_test,nlscls2_label_test))

#if priors are equal for non-linear separable we use squared mahanbolis
def squared_mahanbolis(x,mean,covar):
    x_mu=x-mean
    #applying formula for likelihood
    p1=np.linalg.multi_dot([x_mu.T,np.linalg.inv(covar),x_mu])
    return p1

#mean vectors
nlsmean1 = nlscls1.mean(axis=0)
nlsmean2 = nlscls2.mean(axis=0)
#covariance vectors
nlscovar1=np.cov(nlscls1.T)
nlscovar2=(np.cov(nlscls2.T))


print(nlsmean1, "is the mean of class1 of non-linearly separable data")
print(nlsmean2, "is the mean of class2 of non-linearly separable data")
print("Covariance of class1 of non-linearly separable data")
print(nlscovar1)
print("Covariance of class1 of non-linearly separable data")
print(nlscovar2)

#decision
#prior
nlspcls1 = 0.5
nlspcls2 = 0.5
#since prior is equal for both the classes we use squared mahanbolis for classification
predontestnls = []

#prior
pcls1 = 0.5
pcls2= 0.5
#finding probabilities through bayes classifier and assigning the class labels for test data
for i in range(len(nlstest)):
    x=nlstest[i]
    #assigning to classes
    lh1=squared_mahanbolis(x,nlsmean1,nlscovar1)
    lh2=squared_mahanbolis(x,nlsmean2,nlscovar2)
    
    #posterior probability of a class
    prob1=(lh1*pcls1)/(lh1*pcls1+lh2*pcls2)         #evidence
    prob2=(lh2*pcls2)/(lh1*pcls2+lh2*pcls2)         #evidence
    #assigning class label according to the maximum of posterior probabilty
    if lh2>lh1:
        predontestnls.append(1)
    else:
        predontestnls.append(2)
        

#confusion matrix between predicted label and actual class labels
print("Confusion Matrix after building bayes classifier is \n",confusion_matrix(nlslabel_test,predontestnls))



print("*****************************Real-world data********************************")




fr1 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\real_world_data\class1.txt'
fr2 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\real_world_data\class2.txt'
fr3 = r'C:\Users\jahna\Desktop\Academics\Sem-5\PR\assg1\real_world_data\class3.txt'

rcls1 = np.loadtxt(fr1)
rcls2 = np.loadtxt(fr2)
rcls3 = np.loadtxt(fr3)

for i in range(len(rcls1)):
    plt.scatter(rcls1[i][0],rcls1[i][1], color = 'r')
for i in range(len(rcls2)):
    plt.scatter(rcls2[i][0],rcls2[i][1], color = 'b')
for i in range(len(rcls3)):
    plt.scatter(rcls3[i][0],rcls3[i][1], color = 'g')
plt.title("Real-World data", fontsize=15)
plt.show()

rcls1_label = [1]*len(rcls1)
rcls2_label = [2]*len(rcls2)
rcls3_label = [3]*len(rcls3)

rcls1_train,rcls1_test,rcls1_label_train,rcls1_label_test=train_test_split(rcls1,rcls1_label,test_size=0.2,random_state=42)
rcls2_train,rcls2_test,rcls2_label_train,rcls2_label_test=train_test_split(rcls2,rcls2_label,test_size=0.2,random_state=42)
rcls3_train,rcls3_test,rcls3_label_train,rcls3_label_test=train_test_split(rcls3,rcls3_label,test_size=0.2,random_state=42)

rtest = np.concatenate((rcls1_test, rcls2_test, rcls3_test))
rtrain = np.concatenate((rcls1_train, rcls2_train, rcls3_train))

rlabel_train=np.concatenate((rcls1_label_train,rcls2_label_train,rcls3_label_train))
rlabel_test=np.concatenate((rcls1_label_test,rcls2_label_test,rcls3_label_test))

#calling the function for finding the likelihood of data vector
def calculate_probability(x, mean, covar):
    x_mu=x-mean
    #applying formula for likelihood
    p1=np.linalg.multi_dot([x_mu.T,np.linalg.inv(covar),x_mu])
    exponent=math.exp(-p1/2)
    return (1/(((2*math.pi)**5)*(np.linalg.det(covar))**0.5))*exponent


#mean vectors
rmean1 = rcls1.mean(axis=0)
rmean2 = rcls2.mean(axis=0)
rmean3 = rcls3.mean(axis=0)
#covariance vectors
rcovar1=np.cov(rcls1.T)
rcovar2=(np.cov(rcls2.T))
rcovar3=np.cov(rcls3.T)


print(rmean1,"is the mean of class1 of real-world data")
print(rmean2,"is the mean of class2 of real-world data")
print(rmean3,"is the mean of class3 of real-world data")
print("Covariance of class1 data of real-world data")
print(rcovar1)
print("Covariance of class2 data of real-world data")
print(rcovar2)
print("Covariance of class3 data of real-world data")
print(rcovar3)

#prior

rpcls1 = len(rcls1)/(len(rcls1)+len(rcls2)+len(rcls3))
rpcls2 = len(rcls2)/(len(rcls1)+len(rcls2)+len(rcls3))
rpcls3 = len(rcls3)/(len(rcls1)+len(rcls2)+len(rcls3))

predontestr=[]

#finding probabilities through bayes classifier and assigning the class labels for test data
for i in range(len(rtest)):
    x=rtest[i]
    #assigning to classes
    rlh1=calculate_probability(x,rmean1,rcovar1)
    rlh2=calculate_probability(x,rmean2,rcovar2)
    rlh3=calculate_probability(x,rmean3,rcovar3)
    
    #posterior probability of a class
    prob1=(rlh1*rpcls1)/(rlh1*rpcls1+rlh2*rpcls2+rlh3*rpcls3)#evidence
    prob2=(rlh2*rpcls2)/(rlh1*rpcls1+rlh2*rpcls2+rlh3*rpcls3)#evidence
    prob3=(rlh3*rpcls3)/(rlh1*rpcls1+rlh2*rpcls2+rlh3*rpcls3)#evidence
    #assigning class label according to the maximum of posterior probabilty

    if prob1>max(prob2,prob3):
        predontestr.append(1)
    else:
        if prob2 > prob3:
            predontestr.append(2)
        else:
            predontestr.append(3)

#confusion matrix between predicted label and actual class labels
print("Confusion Matrix after building bayes classifier is \n",confusion_matrix(rlabel_test,predontestr))



