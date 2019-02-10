#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:40:29 2018

@author: Niklas
"""
#Final SVM with tuned kernel function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import svm
import loadMushroom

def svm_baseline():
    
    trainingX,testingX,trainingY,testingY = loadMushroom.load_data()
    
    #print(trainingX.head())
    
    #clf = svm.SVC(C=1000,gamma=0.001,kernel='rbf', degree=3)
    clf = svm.SVC(kernel='rbf', C=15, gamma=0.01, degree=1)

    print "Building the module using RBF kernel using C=15..."
    clf.fit(trainingX, trainingY)
    print "Module is built!"
    
    predictions = [int(a) for a in clf.predict(testingX)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, testingY))
    print "Final classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(testingY))
    
    Ypreds=clf.predict(testingX)
    cm = confusion_matrix(testingY,Ypreds)
    xy=np.array([0,1])
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',
                xticklabels=xy,yticklabels=xy, fmt='g')

if __name__ == "__main__":
    svm_baseline()