#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:43:28 2018

@author: Niklas
"""
#GridSerachSV testing script

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import loadMushroom

trainingX,testingX,trainingY,testingY = loadMushroom.load_data()

clf = svm.SVC()
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
               'C': [7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
               'degree': [1,2,3,4,5]}]

grid=GridSearchCV(clf,param_grid,cv=10,scoring='accuracy')
print("Tuning hyper-parameters")
grid.fit(trainingX,trainingY)
print(grid.best_params_)
print(np.round(grid.best_score_,3))