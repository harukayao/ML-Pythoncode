#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:12:42 2018

@author: haruka
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],
                                                 iris_dataset['target'])
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print('Test set score:\n{:.2f}'.format(knn.score(X_test,y_test)))