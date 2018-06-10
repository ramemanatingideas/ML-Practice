#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:33:37 2018

@author: rando51
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset  = pd.read_csv('abaloneData.csv')
dataset.describe()

y = dataset.iloc[:,-1].values

#encoding the variables manually 
dataset['Male'] = (dataset['Gender']=='M').astype(int)
dataset['Female'] = (dataset['Gender']=='F').astype(int)
dataset['Infant'] = (dataset['Gender']=='I').astype(int)
dataset.drop(['Gender'],axis=1,inplace=True)
#created for shifting the varibales 
col = list(dataset)
#swap the columns fully as needed 
col[0] , col[8] = col[8],col[0]
col[1] , col[9] = col[9],col[1]
col[2] , col[10] = col[10],col[2]
col[8] , col[10] = col[10] , col[8]

#analyse the data
dataset.describe()
#create the matrix of features of independent variables
dataset = dataset.loc[:,col]
X = dataset.iloc[:,:-1].values

#remove those rows whihc have height as 0
dataset = dataset[dataset['Height']>0]


#split the train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state=0)


#applying simple regression linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the result of test data
y_pred = regressor.predict(X_test)

#plot for scatter points
plt.scatter(X_train,y_train,color = 'Green')
plt.plot(X_train,regressor.predict(X_test),color='blue')
plt.show()



