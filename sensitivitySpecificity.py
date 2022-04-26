#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:14:21 2022

@author: ryanweightman
"""

def sensitivitySpecificity(mat, fold):
    """
    Input: a confusion matrix and a fold(scalar)
    
    Output: truePositive, trueNegative,falsePositive,falseNegative rates
    """

    truePositive = mat[fold][fold]
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if i != j and i == fold:
                falsePositive = falsePositive+mat[i][j]
            if i != j and j == fold:
                falseNegative = falseNegative+mat[i][j]
            if i!=fold and j!=fold:
                
                trueNegative= trueNegative+mat[i][j]
    sensitivity =      truePositive / (truePositive+falseNegative)  
    specificity =    trueNegative / (trueNegative + falsePositive)   
                
    return [sensitivity, specificity]


                
def printSpecifictySensitivity(mat):
    """
    Input: a confusion matrix 
    
    Output: Prints sensitivity and specificity in order of the columns of the confusion matrix
    """
    vals = []
    for i in range(len(mat)):
        vals.append( sensitivitySpecificity(mat, i)) 
#        print("the sensitivity of fold {} is {}".format(i+1, vals[0]))
#        print("the specificity of fold {} is {}".format(i+1, vals[1]))
#    for i in vals:
#        print(i)
    print("sensitivity for folds in order:")
    for i in vals:
        print("{:.2f}".format(i[0]), end = ' , ')
    print()
    print("specificity for folds in order:")
    for i in vals:
        print("{:.2f}".format(i[0]),  end = ' , ')