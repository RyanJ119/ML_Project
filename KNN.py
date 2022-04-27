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


# ###IMPORTING CSV####
gdata = pd.read_csv("data/g_data_features.csv")
num_classes = 4

# ###FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:, 1:]  # features
y = gdata.iloc[:, 0]  # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kdata = []
accuracy = []
MCCfinal = []
spec = []
sen = []

# ####LOOP RUNS THROUGH K VALUES 1 TO 18####
for k in range(18):
    cm_total = np.zeros([num_classes, num_classes])
    loops = 50  
    MCC = 0
    score = 0
    for loop in range(loops):
        # ###SECTION ALLOWS FOR THE KNN MODEL TO BE FORMED AS WELL AS KEEP TRACK OF DATA FOR THE LINE GRAPH###    
        classifier = KNeighborsClassifier(n_neighbors=k+1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        score = score + (classifier.score(X_test, y_test))*100
        MCC = MCC + matthews_corrcoef(y_test, y_pred)      
        # ###KEEPS TRACK OF ALL CELLS IN CONFUSION MATRIX IN ORDER TO CREATE AVERAGE####
        cm_total += cm
    
    # ###PRINTING THE CONFUSION MATRIX####
    cm_average = cm_total / loops
    plt.figure(figsize=(7, 5))
    sn.heatmap(cm_average, annot=True)
    plt.title("Confusion Matrix for K = " + str(k+1))
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig("KNNgraphs\Confusionmatrix\CMK="+str(k+1)+".png")
    plt.clf()
    
    # ###SECTION RECORDS RESULTS OF EACH K####
    kdata.append(k+1)
    accuracy.append(score/loops)
    MCCfinal.append(MCC/loops)
    print("K "+str(k+1)+"................DONE"+str(
        sensitivitySpecificity.printSpecifictySensitivity(cm)))

# ###SECTION PRINTS AVERAGES OF THE 50 TESTS####
c = 0
for k in kdata:
    print("K=", k, "Accuracy:", accuracy[c], "MCC:", MCCfinal[c])
    c = c+1

# ###LINE GRAPH####
plt.plot(kdata, accuracy)
plt.title("Accuracy's relationship to K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.savefig(fname="KNNgraphs\Linegraph.png")
