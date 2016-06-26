# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:07:59 2016

@author: oneinchman
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from scipy import append

# Read data and see statistics
data = pd.read_csv('/Users/oneinchman/Documents/Coursera/MPHTI/advertising.csv')
print(data.head())
sns.pairplot(data)

# Error function definition
def errorfunc(w):
    y = data['Sales'].values
    x1 = data['TV'].values
    x2 = data['Radio'].values
    x3 = data['Newspaper'].values
    
    func = (1/200 * sum((y-(w[0]+w[1]*x1+w[2]*x2+w[3]*x3))**2))
    return func
  
# Data preprocessing, standartization and adding a column of ones for w_0  
X = np.array(data[['TV','Radio','Newspaper']].values)
y = np.array(data['Sales'].values)
X = preprocessing.scale(X)
X = np.hstack((np.ones([len(y), 1]), X))

# Prediction error function definition
def mserror(y,y_pred):
    func = (sum((y-y_pred)**2))
    return func* 1/200
    
# Median prediction for sales error
medianPred = np.full((len(y),), np.median(y))
medianError = mserror(y, medianPred)

# Matrix equation function definition
def normal_equation(X,y):
    return np.linalg.solve(X, y)
    
# Calculate w's vector
norm_eq_weights = normal_equation(np.dot(X.T,X), np.dot(X.T, y))



# Linear prediction definition
def linear_prediction(X,w):
    return np.dot(X,w)

# Prediction for norm eq weights
norm_eq_error = mserror(y, linear_prediction(X,norm_eq_weights))


# Stochastic gradient descent realization
from numpy.linalg import norm

def stochastic_gradient_step(X,y,w,train_ind,eta):
    z = np.empty(4)
    z[0] = w[0] + 2*eta/200. * (y[train_ind]-(w[0]+w[1]*X[train_ind,1]+w[2]*X[train_ind,2]+w[3]*X[train_ind,3]))
    z[1] = w[1] + 2*eta/200. * X[train_ind,1] * (y[train_ind]-(w[0]+w[1]*X[train_ind,1]+w[2]*X[train_ind,2]+w[3]*X[train_ind,3]))
    z[2] = w[2] + 2*eta/200. * X[train_ind,2] * (y[train_ind]-(w[0]+w[1]*X[train_ind,1]+w[2]*X[train_ind,2]+w[3]*X[train_ind,3]))
    z[3] = w[3] + 2*eta/200. * X[train_ind,3] * (y[train_ind]-(w[0]+w[1]*X[train_ind,1]+w[2]*X[train_ind,2]+w[3]*X[train_ind,3]))

        
    return z

def stochastic_gradient_descent(X,y,w_init,eta=1e-2,max_iter=1e4,min_weight_dist=1e-8,seed=42,verbose=False):
    weight_dist = np.inf
    w = w_init
    print(w)
    errors = []
    iter_num = 1
    np.random.seed(seed)
    k = 1
    
    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        #print(random_ind)
        w_iter = stochastic_gradient_step(X,y,w,random_ind, eta)
        #print(w_iter)
        weight_dist = norm(np.array(w)-np.array(w_iter))
        #weight_dist = np.sqrt(sum((np.array(w)-np.array(w_iter))**2))
        #print(weight_dist)
        #print np.array(w)
        #print np.array(w_iter)
        w = w_iter
        errors.append(mserror(y,linear_prediction(X,w)))
        iter_num = iter_num + 1
    
    return [w, errors]
    
[stoch_grad_desc_weights, stoch_errors_by_iter] = stochastic_gradient_descent(X,y,[0.0,0.0,0.0,0.0], max_iter = 1e5, seed = 42)
#S = stochastic_gradient_descent(X,y,[0.0,0.0,0.0,0.0], max_iter = 1e5)

plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iter num')
plt.ylabel('MSE')
plt.show()
