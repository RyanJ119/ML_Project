"""
This code runs the SVM algorithm on our feature extraction. In order to get the best information to form our conculsion,
it was decided that we should find how the algoritm performs over 50 seperate test. The average will be computed to ensure our
conclusions are correct
"""
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
import sensitivitySpecificity


# ###IMPORTING CSV####
gdata = pd.read_csv("data/n_data_features.csv")
num_classes = 8

# ###FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:, 1:]  # #features
y = gdata.iloc[:, 0]  # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kernel = ["rbf", "linear", "poly"]
loops = 2

# ###LOOP THAT ALLOWS ALL KERNELS TO BE RUN####
for ktype in kernel:
    print("\nCURRENT KERNEL:", ktype)
    kdata = []
    accuracy = []
    MCCfinal = []
    spec = []
    sen = []    
    for k in range(15):
        # ###KEEPS TRACK OF ALL CELLS IN CONFUSION MATRIX IN ORDER TO CREATE AVERAGE####
        cm_total = np.zeros([num_classes, num_classes])
        score = 0 
        MCC = 0
        kdata.append(k+1)
        for loop in range(loops):
            # ###SECTION ALLOWS FOR THE SVM MODEL TO BE FORMED AS WELL AS KEEP TRACK OF DATA FOR THE LINE GRAPH###
            model = SVC(C=k+1, kernel=ktype)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            current_cm = confusion_matrix(y_test, y_pred)
            score = score + (model.score(X_test, y_test))*100
            MCC = MCC + matthews_corrcoef(y_test, y_pred)

            # ##SECTION KEEPS TRACK OF TOTAL VALUE IN EACH CELL TO CALCULATE DATA###
            cm_total += current_cm

        # ###SECTION RECORDS RESULTS OF EACH C####
        MCCfinal.append(MCC/loops)
        accuracy.append(score/loops)
        print("C "+str(k+1)+"................DONE"+str(
            sensitivitySpecificity.printSpecifictySensitivity(current_cm)))

        # ###PRINTING THE CONFUSION MATRIX FOR EACH CTYPE####
        plt.figure(figsize=(7, 5))
        sn.heatmap(current_cm, annot=True)
        plt.title("Confusion Matrix for" + ktype + " RBF Kernel and C = "+str(k+1))
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig("SVMgraphs\\"+ktype+"\CMC="+str(k+1)+".png")
        plt.clf()

    # ###SECTION PRINTS AVERAGES OF THE 50 TESTS####
    c = 0
    for k in kdata:
        print("K=", k, "Accuracy:", accuracy[c], "MCC:", MCCfinal[c])
        c = c+1

    # ###SECTION OUTPUTS THE LINEGRAPH REPRESENTING THE ACCURACY (OVERALL) OF EACH KERNEL TYPE WITH DIFFERENT C VALUES####
    plt.plot(kdata,accuracy)
    plt.title("Accuracy's relationship to C ("+ktype+" Kernel)")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.savefig("SVMgraphs\\"+ktype+"\RBFSVMLinegraph.png")
    plt.clf()
