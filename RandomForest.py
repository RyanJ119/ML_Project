import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
"""
This module implements the Random Foest Learner on a set of features read in from a csv. 
"""
#Read in CSV
dataset = pd.read_csv("comp.csv")
#Separate labels from data
X = dataset.iloc[:,1:]
y = dataset.iloc[:, 0]
#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#Initialize the Classifier with n-trees
clf=RandomForestClassifier(n_estimators=10)
#Apply classifier to training data
clf.fit(X_train,y_train)
#input test data to check model validity
y_pred=clf.predict(X_test)

#print results
print("confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Matthew Corr: ", matthews_corrcoef(y_test, y_pred))

