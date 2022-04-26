from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Bagging on a chosen classifier. We choose Decision Tree as an example.
"""

import pandas as pd
# import features
dataset = pd.read_csv("comp.csv")
# separate features from data
X = dataset.iloc[:,1:]
Y = dataset.iloc[:, 0]

#Separate data into n pieces 
kfold = model_selection.KFold(n_splits=10)
#Choose a classifier
cart = DecisionTreeClassifier()
#Choose a number of estimators
num_trees = 100
#Implement bagging classifier
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees)
#Average results
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())








