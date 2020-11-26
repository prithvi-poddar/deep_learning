#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:31:58 2020

@author: prithvi
"""

import numpy as np
import pandas as pd

def load_full_data_cross_validate():
    data = pd.read_csv('data/17191.csv').to_numpy()
    X = data[:,:-1]
    y = data[:,-1]
    y_ = []
    
    for i in y:
        a = np.zeros((10,1))
        a[i] = 1.0
        y_.append(a)
        
    training_data = []
    for i in range(len(data)):
        training_data.append((X[i].reshape(784,1).astype('float32')/255.0, y_[i]))
        
    return training_data


def load_full_data():
    data = pd.read_csv('data/17191.csv').to_numpy()
    X = data[:,:-1]
    y = data[:,-1]
    y_ = []
    
    for i in y:
        a = np.zeros((10,1))
        a[i] = 1.0
        y_.append(a)
        
    training_data = []
    for i in range(2500):
        training_data.append((X[i].reshape(784,1).astype('float32')/255.0, y_[i]))
    
    test_data = []
    for i in range(2500, 3000):
        test_data.append((X[i].reshape(784,1).astype('float32')/255.0, y[i]))
        
    return training_data, test_data

def load_PCA():
    data = pd.read_csv('data/17191_pca.csv').to_numpy()
    X = data[:,:-1]
    y = data[:,-1]
    y_ = []

    for i in y:
        a = np.zeros((10,1))
        a[int(i)] = 1.0
        y_.append(a)
        
    training_data = []
    for i in range(len(data)):
        training_data.append((X[i].reshape(25,1).astype('float64'), y_[i]))
        
    return training_data