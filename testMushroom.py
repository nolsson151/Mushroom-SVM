#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:30:25 2018

@author: Niklas
"""

import pandas as pd
import numpy as np

df = pd.read_csv('mushroom.csv', header=None, na_values="?")

pd.set_option('display.max_columns', 21)
print(df.describe())
print("##################################################\n")
print(df.head(10))
print("##################################################\n")
print(df.shape)
print("##################################################\n")
print (sum((df.isnull().sum())))
df.dropna(inplace=True)
print("##################################################\n")
print(df.shape)
print("##################################################\n")
print(df.isnull().sum())
print("##################################################\n")
print(df.head())




#df.fillna(df.mean(), inplace=True) 
#print(df.isnull().sum())