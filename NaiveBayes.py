import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

####IMPORTING CSV####
gdata = pd.read_csv("comp.csv")

####FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:,1:20].values ##LABELS
y = gdata.iloc[:,0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0], (y_test != y_pred).sum()))