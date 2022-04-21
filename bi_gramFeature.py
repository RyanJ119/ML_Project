#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:45:14 2022

@author: ryanweightman
"""

#Import dataframe 
import numpy as np
import pandas as pd
import file_io



def initialize_feature_dataframes(data_dict):
    """This function will create a dictionary that holds the dataframe for features
        data_dict and feature_dict will be parallel
    
    Parameters
    ----------
    data_dict : dictionary
        dictionary where key is file name/data source and value is pd dataframe
        of sequence data
    
    Returns
    -------
    feature_dict : dicitonary
        dictionary of pd dataframes that hold features
    """
    feature_dict = {}
    # iterate over each key in raw dictionary
    for key in data_dict:
        feature_key = key + '_features'

        # the second col holds fold information - these are the labels
        feature_dict[feature_key] = data_dict[key].iloc[:, 1]

    return feature_dict


def countpairs(string, twoLetters):
    counter = 0
    string = list(string)

    for i in range(len(string)-1):
        g = ''.join(string[i:i+2])
        if g == twoLetters:
            counter+=1
    return counter  


def bi_gram(data_list):
    
   # print(data_dict.head())
    pairsOfLetters = create_pairs()
    
    rows = len(data_list)
    cols = len(pairsOfLetters)
    mat = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(len(data_list)):
        for j in range(len(pairsOfLetters)):
            mat[i][j] = countpairs(data_list[i], pairsOfLetters[j])
    return mat


def create_pairs():
    
    amino_acid_alphabet = 'ACDEFGHIKLMNPQRSTVWY' #had  o and u 
    combinations = []
    for i in amino_acid_alphabet:      
        for j in amino_acid_alphabet:    
                combinations.append(i+j)
    return    combinations 
        
     
def convert_feature_list_to_df(feature_list):
    """converts list of lists into a dataframe
    
    Parameters
    ----------
    feature_list : list
        list of lists. First index corresponds to which AA sequence we are at
        second index corresponds to a specific feature
    
    Returns
    -------
    pandas dataframe
        convert to pandas dataframe for easier use in the future
    """
    return pd.DataFrame(feature_list, columns=None)


        


if __name__ == "__main__":
    # testing the above code
    file_name_list = ['g_data.csv', 'n_data.csv']
    test_dict = file_io.read_seq_files(file_name_list)

    cleaned_data = file_io.clean_input(test_dict)

    feature_dict = initialize_feature_dataframes(cleaned_data)

    # sequence data is held in the final col of the dataframe
    g_seq_data = cleaned_data['g_data'].iloc[:, 3].tolist()
    n_seq_data = cleaned_data['n_data'].iloc[:, 3].tolist()

    # print(g_seq_data)
    test3 = bi_gram(g_seq_data)
    # print(sum(test1[0]))
    # print(test1[0])
    # print(sum(test2[0]))
    # print(test2[0])

    # this section shows how to combine the different list of lists
    thing3 = convert_feature_list_to_df(test3)

    print(thing3)

    # file_io.write_feature_info(feature_dict)



        
#check if "xy" pairs are in each string 



#write to csv 

