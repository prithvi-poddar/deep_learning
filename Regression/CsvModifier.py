#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:36:34 2020

@author: prithvi
"""
# Modifies the csv files and separates the date into month and year  
import pandas as pd
import numpy as np

class TrainCsvModifier(object):
    def __init__(self,path):
        self.path = path
        
    def create_new_csv(self):
        data = pd.read_csv(self.path)
        ids = data['id']
        month=[]
        year=[]
        for value in range(len(ids)):
            month.append(int(ids[value][: len(ids[value])-5 ]))
            year.append(int(ids[value][-2:]))
        
        out = data['value']
        res = []
        for i in range(len(out)):
            res.append(out[i])
            
        data_final = {'month':month, 'year':year, 'value':res}
        df = pd.DataFrame(data_final)
        df.to_csv (r'train_new.csv', index = False, header=True)


class TestCsvModifier(object):
    def __init__(self,path):
        self.path = path
        
    def create_new_csv(self):
        data = pd.read_csv(self.path)
        ids = data['id']
        month=[]
        year=[]
        for value in range(len(ids)):
            month.append(int(ids[value][: len(ids[value])-5 ]))
            year.append(int(ids[value][-2:]))
            
        data_final = {'month':month, 'year':year}
        df = pd.DataFrame(data_final)
        df.to_csv (r'test_new.csv', index = False, header=True)