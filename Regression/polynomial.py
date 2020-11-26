#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:29:05 2020

@author: prithvi
"""

import math
import numpy as np


class PolynomialFeature(object):
    """
    polynomial features

    transforms input array with polynomial features

    Example
    =======
    x =
    [[a, b],
    [c, d]]

    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, b^2],
    [1, c, d, c^2, d^2]]
    """

    def __init__(self, degree=2):
       
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if (self.degree==1):
            return np.asarray(x,dtype=np.int64)
        elif(self.degree > 1):
            a=[]
            b=[]
            for i in range (2,self.degree+1):
                for j in range(len(x)):
                    a.append([math.pow(x[j][1],i)])
                    b.append([math.pow(x[j][2],i)])
                a = np.asarray(a,dtype=np.int64)
                b = np.asarray(b,dtype=np.int64)
                x = np.hstack((x,a))
                x = np.hstack((x,b))
                a = []
                b = []
            return np.asarray(x,dtype=np.int64)
        