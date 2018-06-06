#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 00:14:59 2018

@author: rando51
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('iris.csv')
print(dataset.head(20))
dataset.describe()


#split into X and y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#label encoding the variables of categroiacal featurues
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
y_label = LabelEncoder()
y = y_label.fit_transform(y)



#box plot and whisker along wiht histograms to understandthe spread and the dep of var
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

#scatter matrix to see the variation
pd.scatter_matrix(dataset)
plt.show()

#splitting to train and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,ytest = train_test_split(X,y,test_size = 0.3,random_state=0)

#decding the model selection using k-folds method
# Spot Check Algorithms
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
scoring = 'accuracy'
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=0)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


#apply feature scaling to the model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#apply the classifier 

knn = KNeighborsClassifier(n_neighbors = 5,p=2,metric = 'minkowski')
knn.fit(X_train,y_train)

#predicting
y_pred = knn.predict(X_test)

#cnfusion matrix to find the accuracy metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)

from sklearn.metrics import accuracy_score , classification_report
print(accuracy_score(ytest,y_pred))
print(classification_report(ytest,y_pred))



