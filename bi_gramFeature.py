import numpy as np
import pandas as pd
import file_io
from feature_extraction import initialize_feature_dataframes, convert_feature_list_to_df, calc_occur_comp





def countpairs(string, twoLetters):
    """This function will count the number of occurrences of an input two letters in the input string
    
    Parameters
    ----------
    string : string
        input string
    twoLetters: a string consistning of two letters
    
    Returns
    -------
    counter : int
        integer number of times the two letters appear in the string
    """
    # counter = 0
    # string = list(string)

    # for i in range(len(string)-1):
    #     g = ''.join(string[i:i+2])
    #     if g == twoLetters:
    #         counter+=1
    counter = string.count(twoLetters)
    return counter  


def bi_gram(seq_data_list):
    """This function will count the number of occurrences of an input two letters in the input string
    
    Parameters
    ----------
    string : string
        input string
    twoLetters: a string consistning of two letters
    
    Returns
    -------
    mat : 2d matrix
        matrix where first index is which protein sequence and second is feature
        where the features are bigram occurrances
    """
    # print(data_dict.head())
    pairsOfLetters = create_pairs()
    
    rows = len(seq_data_list)
    cols = len(pairsOfLetters)
    mat = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(len(seq_data_list)):
        for j in range(len(pairsOfLetters)):
            mat[i][j] = countpairs(seq_data_list[i], pairsOfLetters[j])
    return mat


def create_pairs():
    """This function will simply create all combinations of two letters in a given alphabet

    Returns
    -------
    combinations : list
       returns a list of all sets of two letters in a given alphabet
    """
    amino_acid_alphabet = 'ACDEFGHIKLMNPQRSTVWY'  # had  o and u 
    combinations = []
    for i in amino_acid_alphabet:      
        for j in amino_acid_alphabet:    
            combinations.append(i+j)
    return combinations 


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
    g_bigram_df = convert_feature_list_to_df(test3)


    g_comp, g_occur = calc_occur_comp(g_seq_data)
    g_comp_df = convert_feature_list_to_df(g_comp)
    g_occur_df = convert_feature_list_to_df(g_occur)

    existing_features = feature_dict['g_data_features']
    updated_features = pd.concat([existing_features, g_comp_df, g_occur_df, g_bigram_df], axis=1)

    print(updated_features)

        
