"""
This code runs the SVM algorithm on our feature extraction. In order to get the best information to form our conculsion,
it was decided that we should find how the algoritm performs over 50 seperate test. The average will be computed to ensure our
conclusions are correct
"""
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
import sensitivitySpecificity
####IMPORTING CSV####
gdata = pd.read_csv("data/n_data_features.csv")
####FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:,1:455].values ##LABELS
y = gdata.iloc[:,0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kernel = ["rbf","linear","poly"]
loops = 2
####LOOP THAT ALLOWS ALL KERNELS TO BE RUN####
for ktype in kernel:
    print("\nCURRENT KERNEL:",ktype)
    kdata = []
    accuracy = []
    MCCfinal = []
    spec = []
    sen = []    
    for k in range(15):
####KEEPS TRACK OF ALL CELLS IN CONFUSION MATRIX IN ORDER TO CREATE AVERAGE####
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
        score = 0 
        MCC = 0
        kdata.append(k+1)
        for loop in range(loops):
####SECTION ALLOWS FOR THE SVM MODEL TO BE FORMED AS WELL AS KEEP TRACK OF DATA FOR THE LINE GRAPH###
            model = SVC(C=k+1, kernel=ktype)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            score = score + (model.score(X_test,y_test))*100
            MCC = MCC + matthews_corrcoef(y_test,y_pred)
###SECTION KEEPS TRACK OF TOTAL VALUE IN EACH CELL TO CALCULATE DATA###
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
####SECTION RECORDS RESULTS OF EACH C####
        MCCfinal.append(MCC/loops)
        accuracy.append(score/loops)
        print("C "+str(k+1)+"................DONE"+str(sensitivitySpecificity.printSpecifictySensitivity(cm)))
####PRINTING THE CONFUSION MATRIX FOR EACH CTYPE####
        plt.figure(figsize=(7,5))
        sn.heatmap(cm,annot=True)
        plt.title("Confusion Matrix for" + ktype +" RBF Kernel and C = "+str(k+1))
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig("SVMgraphs\\"+ktype+"\CMC="+str(k+1)+".png")
        plt.clf()
####SECTION PRINTS AVERAGES OF THE 50 TESTS####
    c= 0
    for k in kdata:
        print("K=",k,"Accuracy:",accuracy[c],"MCC:",MCCfinal[c])
        c = c+1
####SECTION OUTPUTS THE LINEGRAPH REPRESENTING THE ACCURACY (OVERALL) OF EACH KERNEL TYPE WITH DIFFERENT C VALUES####
    plt.plot(kdata,accuracy)
    plt.title("Accuracy's relationship to C ("+ktype+" Kernel)")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.savefig("SVMgraphs\\"+ktype+"\RBFSVMLinegraph.png")
    plt.clf()

    

