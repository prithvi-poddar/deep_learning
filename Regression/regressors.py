#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:29:05 2020

@author: prithvi
"""

import numpy as np
import math

class LinearRegression(object):
    
    def fit(self,X_train,y_train,X_test,y_test):
        self.w = np.linalg.pinv(X_train).dot(y_train)
        test = X_test.dot(self.w)
        test_error = 0
        for i in range(len(y_test)):
            test_error = test_error + (test[i]-y_test[i])**2
        test_mse = test_error/len(test)
        
        train = X_train.dot(self.w)
        train_error = 0
        for i in range(len(y_train)):
            train_error = train_error + (train[i]-y_train[i])**2
        train_mse = train_error/len(train)
        
        print("Linear Regression errors-")
        print("Training set error: "+str(train_mse))
        print("Test set error: "+str(test_mse))
        return train_mse, test_mse
    
    def predict(self,test_data):
        prediction = test_data.dot(self.w)
        return prediction
        
    
class BayesianLinearRegression:
    
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        
    def fit(self,X_train,y_train,X_test,y_test):
        n = np.shape(X_train)[1]
        self.Sn = np.linalg.inv(self.alpha*np.eye(n)+(self.beta*(np.transpose(X_train).dot(X_train))))
        self.Mn = self.beta*((self.Sn.dot(np.transpose(X_train))).dot(y_train))
        
        phi_2_train = (1/self.beta)+((X_train.dot(self.Sn)).dot(np.transpose(X_train)))
        mean_train = self.Mn.dot(np.transpose(X_train))
        
        phi_2_test = (1/self.beta)+((X_test.dot(self.Sn)).dot(np.transpose(X_test)))
        mean_test = self.Mn.dot(np.transpose(X_test))
        
        train_predictions = np.random.multivariate_normal(mean_train,phi_2_train)
        test_predictions = np.random.multivariate_normal(mean_test,phi_2_test)
        
        train_error = 0
        for i in range(len(y_train)):
            train_error = train_error + (train_predictions[i]-y_train[i])**2
        train_mse = train_error/len(train_predictions)
        
        test_error = 0
        for i in range(len(y_test)):
            test_error = test_error + (test_predictions[i]-y_test[i])**2
        test_mse = test_error/len(test_predictions)
        
        print("Bayesian Linear Regression errors-")
        print("Training set error: "+str(train_mse))
        print("Test set error: "+str(test_mse))
        return train_mse, test_mse
    
    def predict(self,test_data):
        phi_2 = (1/self.beta)+((test_data.dot(self.Sn)).dot(np.transpose(test_data)))
        mean = self.Mn.dot(np.transpose(test_data))
        predictions = np.random.multivariate_normal(mean,phi_2)
        return predictions




        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
