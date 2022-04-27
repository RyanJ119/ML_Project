"""
This code runs the KNN algorithm on our feature extraction. In order to get the best information to form our conculsion,
it was decided that we should find how the algoritm performs over 50 seperate test. The average will be computed to ensure our
conclusions are correct
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import sensitivitySpecificity
####IMPORTING CSV####
gdata = pd.read_csv("data/g_data_features.csv")
####FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:,1:455].values ##LABELS
y = gdata.iloc[:,0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kdata = []
accuracy = []
MCCfinal = []
spec = []
sen = []
#####LOOP RUNS THROUGH K VALUES 1 TO 18####
for k in range(18):
    A0 = 0
    A1 = 0
    A2 = 0
    A3 = 0   
    B0 = 0
    B1 = 0
    B2 = 0
    B3 = 0    
    C0 = 0
    C1 = 0
    C2 = 0
    C3 = 0   
    D0 = 0
    D1 = 0
    D2 = 0
    D3 = 0 
    loops = 50  
    MCC = 0
    score = 0
    for loop in range(loops):
####SECTION ALLOWS FOR THE KNN MODEL TO BE FORMED AS WELL AS KEEP TRACK OF DATA FOR THE LINE GRAPH###    
        classifier = KNeighborsClassifier(n_neighbors=k+1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        score = score + (classifier.score(X_test,y_test))*100
        MCC = MCC + matthews_corrcoef(y_test,y_pred)      
####KEEPS TRACK OF ALL CELLS IN CONFUSION MATRIX IN ORDER TO CREATE AVERAGE####
        A0 = cm[0][0] + A0
        A1 = cm[0][1] + A1
        A2 = cm[0][2] + A2
        A3 = cm[0][3] + A3
        B0 = cm[1][0] + B0
        B1 = cm[1][1] + B1
        B2 = cm[1][2] + B2
        B3 = cm[1][3] + B3
        C0 = cm[2][0] + C0
        C1 = cm[2][1] + C1
        C2 = cm[2][2] + C2
        C3 = cm[2][3] + C3
        D0 = cm[3][0] + D0
        D1 = cm[3][1] + D1
        D2 = cm[3][2] + D2
        D3 = cm[3][3] + D3
####PRINTING THE CONFUSION MATRIX####
    cm =[[A0/loops,B0/loops,C0/loops,D0/loops],[A1/loops,B1/loops,C1/loops,D1/loops],[A2/loops,B2/loops,C2/loops,D2/loops],[A3/loops,B3/loops,C3/loops,D3/loops]]
    plt.figure(figsize=(7,5))
    sn.heatmap(cm,annot=True)
    plt.title("Confusion Matrix for K = " + str(k+1))
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig("KNNgraphs\Confusionmatrix\CMK="+str(k+1)+".png")
    plt.clf()
####SECTION RECORDS RESULTS OF EACH K####
    kdata.append(k+1)
    accuracy.append(score/loops)
    MCCfinal.append(MCC/loops)
    print("K "+str(k+1)+"................DONE"+str(sensitivitySpecificity.printSpecifictySensitivity(cm)))
####SECTION PRINTS AVERAGES OF THE 50 TESTS####
c= 0
for k in kdata:
    print("K=",k,"Accuracy:",accuracy[c],"MCC:",MCCfinal[c])
    c = c+1
####LINE GRAPH####
plt.plot(kdata,accuracy)
plt.title("Accuracy's relationship to K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.savefig(fname="KNNgraphs\Linegraph.png")
    