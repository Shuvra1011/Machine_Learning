# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

## Import Dataset ##
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 3].values

## Dealing with missing values ##
from sklearn.impute import SimpleImputer 
simple_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
simple_imputer = simple_imputer.fit(X[:, 1:3])
X[:, 1:3] = simple_imputer.transform(X[:, 1:3])

## Encoding Categorical Data ##
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_X.fit_transform(y)

## Split dataset ##

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling ##
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
