#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:35:02 2022

@author: ryanweightman
"""

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:53:54 2022

@author: ryanweightman
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


#print(dataset.head())



dataset = pd.read_csv("comp.csv")
X = dataset.iloc[:,1:]
Y = dataset.iloc[:, 0]


kfold = model_selection.KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees)

results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())








