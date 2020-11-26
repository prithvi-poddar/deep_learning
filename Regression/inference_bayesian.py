#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:27:44 2020

@author: prithvi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CsvModifier import (TrainCsvModifier, TestCsvModifier)
from regressors import BayesianLinearRegression
from polynomial import PolynomialFeature

path_train= 'train.csv'
new_train = TrainCsvModifier(path_train)
new_train.create_new_csv()

path_predict = 'test.csv'
new_predict = TestCsvModifier(path_predict)
new_predict.create_new_csv()

data = pd.read_csv('train_new.csv')

X = data.iloc[:,:2].to_numpy()
y = data.iloc[:,-1:].to_numpy()
y = np.reshape(y, (110))

n = 100
X_train = X[:n,:]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]

y_train = y[:n]

X_test = X[n:,:] 
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]

y_test = y[n:]

feature = PolynomialFeature(6)
X_train = feature.transform(X_train)
X_test = feature.transform(X_test)

regressor = BayesianLinearRegression(0,10)
error1, error2 = regressor.fit(X_train,y_train,X_test,y_test)

pred = pd.read_csv('test_new.csv')
X_pred = pred.iloc[:,:].to_numpy()
X_pred = np.c_[np.ones(len(X_pred),dtype='int64'),X_pred]
X_pred = feature.transform(X_pred)
prediction = regressor.predict(X_pred)

result = pd.read_csv('test.csv')
ids = result['id']
final = {'id':ids, 'value':prediction}
df = pd.DataFrame(final)
df.to_csv (r'Prithvi_17191_prediction3.csv', index = False, header=True)
