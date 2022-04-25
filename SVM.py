import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

####IMPORTING CSV####
gdata = pd.read_csv("comp.csv")

####FORMATING DATA TO BE PASSED INTO THE MODEL####
X = gdata.iloc[:,1:20].values ##LABELS
y = gdata.iloc[:,0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
kdata = []
accuracy = []
kernel = ["rbf","linear","poly"]
#####CONSIDERING THERE ARE MANY KERNELS I WILL PRINT IN SEPERATE SECTIONS####
for ktype in kernel:
    print("\nCURRENT KERNEL:",ktype)
    for k in range(15):
        ####SECTION ALLOWS FOR THE SVM MODEL TO BE FORMED AS WELL AS KEEP TRACK OF DATA FOR THE LINE GRAPH###
        kdata.append(k+1)
        model = SVC(C=k+1, kernel=ktype)
        model.fit(X_train, y_train)
        accuracy.append(model.score(X_test, y_test))
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ####SECTION CALCULATES THE STATS REQUESTED (AS PER ASSIGNMENT) USING THE COORDS. FROM THE CONFUSION MATRIX####
        print("Statistics for c=",k)

        #Fold 1#
        TP = cm[0][0]
        FP = cm[1][0]+cm[2][0]+cm[3][0]
        FN = cm[0][1]+cm[0][2]+cm[0][3]
        TN = (cm[1][1]+cm[1][2]+cm[1][3])+(cm[2][1]+cm[2][2]+cm[2][3])+(cm[3][1]+cm[3][2]+cm[3][3])
        Accuracy = (TP+TN)/(TP+TN+FP+FN) 
        Specificity = TN/(TN+FP)
        Sensitivity = TP/(TP+FN)
        MCC = ((TP*TN)-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))*.5)
        print("FOLD 1 DATA    Accuracy:",Accuracy,"Specificity:",Specificity,"Sensitivity:",Sensitivity,"MCC:",MCC)
        #Fold 2#
        TP = cm[1][1]
        FP = cm[0][1]+cm[2][1]+cm[3][1]
        FN = cm[1][0]+cm[1][2]+cm[1][3]
        TN = (cm[0][0]+cm[0][2]+cm[0][3])+(cm[2][0]+cm[2][2]+cm[2][3])+(cm[3][0]+cm[3][2]+cm[3][3])
        Accuracy = (TP+TN)/(TP+TN+FP+FN) 
        Specificity = TN/(TN+FP)
        Sensitivity = TP/(TP+FN)
        MCC = ((TP*TN)-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))*.5)
        print("FOLD 2 DATA    Accuracy:",Accuracy,"Specificity:",Specificity,"Sensitivity:",Sensitivity,"MCC:",MCC)
        #Fold 3#
        TP = cm[2][2]
        FP = cm[0][2]+cm[1][2]+cm[3][2]
        FN = cm[2][0]+cm[2][1]+cm[2][3]
        TN = (cm[0][0]+cm[0][1]+cm[0][3])+(cm[1][0]+cm[1][1]+cm[1][3])+(cm[3][0]+cm[3][1]+cm[3][3])
        Accuracy = (TP+TN)/(TP+TN+FP+FN) 
        Specificity = TN/(TN+FP)
        Sensitivity = TP/(TP+FN)
        MCC = ((TP*TN)-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))*.5)
        print("FOLD 3 DATA    Accuracy:",Accuracy,"Specificity:",Specificity,"Sensitivity:",Sensitivity,"MCC:",MCC)
        #Fold 4#
        TP = cm[3][3]
        FP = cm[2][3]+cm[1][3]+cm[0][3]
        FN = cm[3][2]+cm[3][1]+cm[3][0]
        TN = (cm[0][0]+cm[0][1]+cm[0][2])+(cm[1][0]+cm[1][1]+cm[1][2])+(cm[2][0]+cm[2][1]+cm[2][2])
        Accuracy = (TP+TN)/(TP+TN+FP+FN) 
        Specificity = TN/(TN+FP)
        Sensitivity = TP/(TP+FN)
        MCC = ((TP*TN)-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))*.5)
        print("FOLD 4 DATA    Accuracy:",Accuracy,"Specificity:",Specificity,"Sensitivity:",Sensitivity,"MCC:",MCC)

        ####PRINTING THE CONFUSION MATRIX FOR EACH KTYPE####
        plt.figure(figsize=(7,5))
        sn.heatmap(cm,annot=True)
        plt.title("Confusion Matrix for" + ktype +" RBF Kernel and C = "+str(k+1))
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig("SVMgraphs\\"+ktype+"\CMC="+str(k+1)+".png")
        plt.clf()

    ####SECTION OUTPUTS THE LINEGRAPH REPRESENTING THE ACCURACY (OVERALL) OF EACH KERNEL TYPE WITH DIFFERENT C VALUES####    
    plt.plot(kdata,accuracy)
    plt.title("Accuracy's relationship to C ("+ktype+" Kernel)")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.savefig("SVMgraphs\\"+ktype+"\RBFSVMLinegraph.png")

    

