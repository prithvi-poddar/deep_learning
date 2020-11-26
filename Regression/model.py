#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:07:03 2020

@author: prithvi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CsvModifier import (TrainCsvModifier, TestCsvModifier)
from regressors import LinearRegression, BayesianLinearRegression
from polynomial import PolynomialFeature

path_train= 'train.csv'
new_train = TrainCsvModifier(path_train)
new_train.create_new_csv()

path_predict = 'test.csv'
new_predict = TestCsvModifier(path_predict)
new_predict.create_new_csv()

#Train and test splits
data = pd.read_csv('train_new.csv')

X = data.iloc[:,:2].to_numpy()
y = data.iloc[:,-1:].to_numpy()
y = np.reshape(y, (110))

#linear regression linear basis
n = 100
X_train = X[:n,:]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train = y[:n]
X_test = X[n:,:]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y[n:]
regressor = LinearRegression()
train_error, test_error = regressor.fit(X_train,y_train,X_test,y_test)
errors = {"Linear regression_linear basis":[train_error,test_error]}


#linear regression polynomial basis
training_errors = []
test_errors = []
n = 100
X_train = X[:n,:]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]

y_train = y[:n]

X_test = X[n:,:] 
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]

y_test = y[n:]

regressor = LinearRegression()

for i in range(1,15):
    feature = PolynomialFeature(i)
    X_train = feature.transform(X_train)
    X_test = feature.transform(X_test)
    error1, error2 = regressor.fit(X_train,y_train,X_test,y_test)
    training_errors.append(error1)
    test_errors.append(error2)
    
plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("MSE")
plt.show()


