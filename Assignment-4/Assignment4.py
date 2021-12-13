#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 08:47:51 2021

@author: sanvireddy
"""

#importing the required modules
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# giving the parameters of showing the plot
plt.rcParams["figure.figsize"] = (10,7)

# using grid style
plt.style.use("ggplot")


def dataFrame(path,sep,titles):
    data = pd.DataFrame()
    color = ["r","g"]
    marker = ['*','v']
    for i in range(len(path)):
        X = pd.read_csv(path[i], sep=sep, header=None)  
        data11 = pd.read_csv(path[i]).to_numpy()
        
        plt.scatter(data11[:,0],data11[:,1],c=color[i],marker = marker[i])
        
        X[len(X.columns)] = i
        data = pd.concat([data,X],ignore_index=True)
    plt.xlabel("Attribute 1")
    plt.ylabel("Attribute 2")
    plt.title(titles)
    plt.show()
    data.columns = ['X','Y','label']
    return data 

df_nls = dataFrame(['/Users/sanvireddy/Downloads/Assignment_04/nls_data/class1.txt','/Users/sanvireddy/Downloads/Assignment_04/nls_data/class2.txt'],","," Visualisation of Non Linearly separable Data")
actuallabel_nls = df_nls['label']

# dropping the column with class values
X_nls = df_nls.drop('label', axis=1)
Y_nls = df_nls['label']

#splitting the data into train and test
X_train_nls, X_test_nls, Y_train_nls, Y_test_nls = train_test_split(X_nls, Y_nls, test_size=0.3, stratify = Y_nls, random_state=42)
X_train_nls = X_train_nls.values
X_test_nls = X_test_nls.values

df_ls = dataFrame(['/Users/sanvireddy/Downloads/Assignment_04/ls_data/class1.txt','/Users/sanvireddy/Downloads/Assignment_04/ls_data/class2.txt'],",","Visualisation of Linearly separable Data")
actuallabel_ls = df_ls['label']

X_ls = df_ls.drop('label', axis=1)
Y_ls = df_ls['label']


X_train_ls, X_test_ls, Y_train_ls, Y_test_ls = train_test_split(X_ls, Y_ls, test_size=0.3, stratify = Y_ls, random_state=42)
X_train_ls = X_train_ls.values
X_test_ls = X_test_ls.values



class Perceptron:

   def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y,titles):
      """Fit training data
      X : Training vectors, X.shape : [#samples, #features]
      y : Target values, y.shape : [#samples]
      """

      # weights
      self.weight = np.zeros(1 + X.shape[1])
      
      # Number of misclassifications
      self.errors = []  # Number of misclassifications
      # each iteration the weights (w) are updated using the equation:
      # updated_weight = w + learning_rate * (expected - predicted) * x
      for i in range(self.niter):
         err = 0
         for xi, target in zip(X, y):
            delta_w = self.rate * (target - self.predict(xi))
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            err += int(delta_w != 0.0)
         self.errors.append(err)
         
      #Graph For misclassification vs iteration. 
      plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
      plt.xlabel('Epochs')
      plt.ylabel('Number of misclassifications')
      plt.title("No. of misclassification vs iteration for " + titles)
      plt.show()
      
      #Graph for the separating hyperplane.
      w0 = self.weight[0]/self.weight[2]
      w1 = self.weight[1]/self.weight[2]
      x11 = np.linspace(-10,15,100)
      
       # Straight Line eqn of the separating hyperplane.
       
      y11 = -w1*x11-w0  
      
      #plot of hyperplane
      plt.plot(x11, y11, 'blue', label='y=w0+wx')
      plt.title('Graph of y=wx+w0')
      plt.xlabel('x', color='#1C2833')
      plt.ylabel('y', color='#1C2833')
      plt.legend(loc='upper left')
      plt.grid()
      return self
      
   def net_input(self, X):
      """Calculate net input"""
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def predict(self, X):
      """Return class label after unit step"""
      return np.where(self.net_input(X) >= 0.0, 1, 0)
    

def plot_decision_regions(X, Y, X_test, Y_test, classifier, resolution=0.02):
   # setup marker generator and color map
   
   markers = ('*', 'v', '*', 'v', '^')
   colors = ('red', 'green', 'blue', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(Y))])
   colors1 = ('yellow','magenta')
   cmap1 = ListedColormap(colors1[:len(np.unique(Y))])
   
   
   # plot the decision surface
   x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot class samples
   for idx, cl in enumerate(np.unique(Y)):
      plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1], alpha=1, color=cmap(idx),marker=markers[idx], label=cl)

   for idx, cl in enumerate(np.unique(Y_test)):
      plt.scatter(x=X_test[Y_test == cl, 0], y=X_test[Y_test == cl, 1], alpha=0.5, color=cmap1(idx),marker=markers[idx+2], label=cl)
    

def Perceptron_(x_train,y_train,x_test,y_test, titles):
    print()
    print()
    print("=========++++++++++++  Q1). Perceptron for",titles,"Data  +++++++++++++========")  
    perceptron = Perceptron(rate = 0.5,niter = 20)
    perceptron.fit(x_train, y_train,titles)
    y_pred = perceptron.predict(x_test)
    #print(y_pred)
    
    print("Accuracy For Perceptron: ",accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix For Perceptron: ",conf_mat[0])
    print("                                 ",conf_mat[1])
    plot_decision_regions(x_train, y_train,x_test,y_test, classifier=perceptron)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Decision Boundry for "+titles)
    plt.legend()
    plt.show()



def MultiLayerPerceptron(x_train, y_train, x_test, y_test, titles, hidden_layer):
    print()
    print()
    print("=========++++++++++++  Q2). MLP for",titles,"Data  +++++++++++++========")  
    
    MLP_classifier = MLPClassifier(hidden_layer_sizes = hidden_layer, solver ='sgd',learning_rate_init = 0.001, max_iter = 1000,random_state = 42)
    
    MLP_classifier.fit(x_train, y_train)
    plot_decision_regions(x_train,y_train,x_test,y_test, classifier=MLP_classifier)
    plt.title(titles)
    
    plt.legend()
    plt.show()
    y_pred_mlp = MLP_classifier.predict(x_test)
    print("Accuracy For MLP: ",accuracy_score(y_test, y_pred_mlp))
    conf_mat_mlp = confusion_matrix(y_test,y_pred_mlp)
    
    print("Confusion Matrix For MLP: ",conf_mat_mlp[0])
    print("                          ",conf_mat_mlp[1])
  
    

def SVM(x_train, y_train, x_test, y_test, titles, kernals):
    print()
    print()
    print("=====+++++++  Q3). SVM for",titles,"Data with",kernals,"Kernal  +++++++=====")  
    SVM_classifier = SVC(gamma='auto', kernel =kernals)
    # fit the data in SVC
    SVM_classifier.fit(x_train,y_train)
    plot_decision_regions(x_train,y_train,x_test,y_test, classifier=SVM_classifier)
    plt.title(titles)
    
    y_pred = SVM_classifier.predict(x_test)
    print("Accuracy For MLP: ",accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test,y_pred)
    
    print("Confusion Matrix For MLP: ",conf_mat[0])
    print("                          ",conf_mat[1])
    if(kernals != "linear"):
        plt.show()

    # plot the decision boundary ,data points,support vector etcv

    if (kernals == "linear"):
        w = SVM_classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-10,15,100)
        yy = a * xx - SVM_classifier.intercept_[0] / w[1]
        y_neg = a * xx - SVM_classifier.intercept_[0] / w[1] + 1
        y_pos = a * xx - SVM_classifier.intercept_[0] / w[1] - 1
    
        plt.plot(xx, yy, 'k') #label=f"Decision Boundary (y ={w[0]}x1  + {w[1]}x2  {SVM_classifier.intercept_[0] })"
        plt.plot(xx, y_neg, 'b-.')#,label=f"Neg Decision Boundary (-1 ={w[0]}x1  + {w[1]}x2  {SVM_classifier.intercept_[0] })"
        plt.plot(xx, y_pos, 'r-.')#,label=f"Pos Decision Boundary (1 ={w[0]}x1  + {w[1]}x2  {SVM_classifier.intercept_[0] })"
        plt.legend()
        plt.show()
        # calculate margin
        print(f'Margin : {2.0 /np.sqrt(np.sum(SVM_classifier.coef_ ** 2)) }')

    
    
Perceptron_(X_train_ls,Y_train_ls,X_test_ls, Y_test_ls,"Linearly Separable") 
Perceptron_(X_train_nls,Y_train_nls,X_test_nls, Y_test_nls,"Non-Linearly Separable")


MultiLayerPerceptron(X_train_ls,Y_train_ls, X_test_ls,Y_test_ls, "Linearly Separable",(5,2))
MultiLayerPerceptron(X_train_nls,Y_train_nls, X_test_nls,Y_test_nls, "Non Linearly Separable",(8,6,4,2)) 

SVM(X_train_ls,Y_train_ls,X_test_ls, Y_test_ls,"linearly Separable","linear") 
SVM(X_train_nls,Y_train_nls,X_test_nls, Y_test_nls,"Non Linearly Separable","linear")
SVM(X_train_ls,Y_train_ls,X_test_ls, Y_test_ls,"linearly Separable","poly") 
SVM(X_train_nls,Y_train_nls,X_test_nls, Y_test_nls,"Non Linearly Separable","poly")


