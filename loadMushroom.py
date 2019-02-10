#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 14:23:49 2018

@author: Niklas
"""
#Load dataset into training an testing set

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():
    
    df = pd.read_csv("mushroom.csv")
    df.dropna(inplace=True)
    df1 = df.drop(columns=['veil-type'], axis=1)
    new_df = df1.copy()
    for i in df1.columns:
        new_df[i] = LabelEncoder().fit_transform(df1[i])
    
    X=new_df.drop(['Class'],axis=1)
    
    y=new_df['Class']
    
    trainingX,testingX,trainingY,testingY=train_test_split(X,y,test_size=0.30)
    
    return (trainingX,testingX,trainingY,testingY)
