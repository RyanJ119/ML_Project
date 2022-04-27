"""
This code runs the NaiveBayes algorithm on our feature extraction. In order to get the best information to form our conculsion,
it was decided that we should find how the algoritm performs over 50 seperate test. The average will be computed to ensure our
conclusions are correct
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import sensitivitySpecificity
####IMPORTING CSV####
gdata = pd.read_csv("data/g_data_features.csv")
####FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:,1:455].values ##LABELS 455
y = gdata.iloc[:,0].values 
loops = 50
kdata = []
accuracy = []
####Run the Naive Bayes Classification multiple times
for loop in range(loops):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train,y_train).predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    MCC = matthews_corrcoef(y_test,y_pred)
    score = gnb.score(X_test,y_test)
    kdata.append(loop+1)
    accuracy.append(score)
    print("LOOP",loop,"SCORE", score,"MCC",MCC,sensitivitySpecificity.printSpecifictySensitivity(cm))
###Plot Section
plt.plot(kdata,accuracy)
plt.title("Accuracy")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.savefig("NaiveBayesgraph")
plt.clf()