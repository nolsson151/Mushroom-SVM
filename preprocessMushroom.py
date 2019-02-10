#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:22:59 2018

@author: Niklas
"""
#Preprocessing dataset script

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("mushroom.csv")

df.dropna(inplace=True)

print("\nHead of dataset after removeing mssing values")
print("##################################################\n")
print(df.head(10))
print("\nSum of of entires with missing data")
print (sum((df.isnull().sum())))

print("\nRemove veil-type from dataset")
print("##################################################\n")
df1 = df.drop(columns=['veil-type'], axis=1)
print(df1.describe().transpose())

 
print("\nHead of dataset after encoding values")
print("##################################################\n")
new_df = df1.copy()
for i in df1.columns:
    new_df[i] = LabelEncoder().fit_transform(df1[i])

print(new_df.head(10))
