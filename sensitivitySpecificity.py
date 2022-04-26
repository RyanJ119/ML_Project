
def sensitivitySpecificity(mat, fold):
    """
    Input: a confusion matrix and a fold(scalar)
    
    Output: Sensitivity and Specificity values for the given fold
    """
    # initialize values for computing Sensitivity and Specificity
    truePositive = mat[fold][fold]
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    #calculate true negative, true positive, false negative and false positive values
    for i in range(len(mat)):
        for j in range(len(mat)):
            if i != j and i == fold:
                falsePositive = falsePositive+mat[i][j]
            if i != j and j == fold:
                falseNegative = falseNegative+mat[i][j]
            if i!=fold and j!=fold:
                trueNegative= trueNegative+mat[i][j]
     # Calculate Sensitivity and Specificity        
    sensitivity =      truePositive / (truePositive+falseNegative)  
    specificity =    trueNegative / (trueNegative + falsePositive)   
                
    return [sensitivity, specificity]


                
def printSpecifictySensitivity(mat):
    """
    Input: a confusion matrix 
    
    Output: Prints sensitivity and specificity in order of the columns of the confusion matrix
    """
    # initialize collection list
    vals = []
    #run sensitivitySpecificity function on each of the labels in order, collecting them in vals
    for i in range(len(mat)):
        vals.append( sensitivitySpecificity(mat, i)) 
#        print("the sensitivity of fold {} is {}".format(i+1, vals[0]))
#        print("the specificity of fold {} is {}".format(i+1, vals[1]))
#    for i in vals:
#        print(i)

    print("sensitivity for folds in order:")
    #Print all sensitivity values in order of confusion matrix columns 
    for i in vals:
        print("{:.2f}".format(i[0]), end = ' , ')
    print()
    #print all specificity values in order of confusion matrix columns
    print("specificity for folds in order:")
    for i in vals:
        print("{:.2f}".format(i[1]),  end = ' , ')
