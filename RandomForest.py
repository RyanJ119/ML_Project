import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv("comp.csv")
X = dataset.iloc[:,1:]
y = dataset.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Matthew Corr: ", matthews_corrcoef(y_test, y_pred))

