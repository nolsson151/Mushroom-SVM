#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:18:10 2018

@author: Niklas
"""
#Exploring dataset script

import pandas as pd

df = pd.read_csv("mushroom.csv")

print("\nHead of dataset")
print("##################################################\n")
print(df.head(5))
print(df.shape)

print("\nUnique values within attributes")
print("##################################################\n")
print(df.nunique()) 
print("\nSum of unique values")
print(df.nunique().sum())

print("\nAttributes with missing data")
print("##################################################\n")
print(df.isnull().sum())
print("\nSum of of entires with missing data")
print (sum((df.isnull().sum())))

