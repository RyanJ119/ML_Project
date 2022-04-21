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


def calc_occur_comp(seq_data_list):
    """This function takes in a list of AA sequences and returns an occurnece list
    that is parallel to the input list. The occurence list holds 'sub-lists' that
    correspond to the counts of the letter in the AA alphabet
    
    Parameters
    ----------
    seq_data_list : list
        list of AA sequences
    
    Returns
    -------
    occur_list : list (of lists)
        list that holds counts of letters from the the AA seqs

    comp_list : list (of lists)
        list that holds the composition from each of the AA seqs
    """
    # we count the occurences of the letters in this alphabet
    amino_acid_alphabet = 'ACDEFGHIKLMNOPQRSTUVWY'
    occur_list = []  # this will store the occurences for the seqs
    comp_list = []  # this will store the composition for the seqs

    for seq in seq_data_list:
        # init list of 0s, corresponding to counts of AA in seq
        temp_occur = [0] * 22
        seq_len = len(seq)  # get the len of seq for composition calc
        for i in range(0, len(amino_acid_alphabet)):  # for each letter in alphabet
            char = amino_acid_alphabet[i]  # recover the letter
            counts = seq.count(char)  # get the occurence for the letter
            temp_occur[i] = counts

        temp_comp = [x / seq_len for x in temp_occur]

        comp_list.append(temp_comp)  # store the composition list
        occur_list.append(temp_occur)  # store the occurence list

    return occur_list, comp_list


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
    test1, test2 = calc_occur_comp(g_seq_data)
    # print(sum(test1[0]))
    # print(test1[0])
    # print(sum(test2[0]))
    # print(test2[0])

    # this section shows how to combine the different list of lists
    thing1 = convert_feature_list_to_df(test1)
    thing2 = convert_feature_list_to_df(test2)
    combine = pd.concat([thing1, thing2], axis=1)
    combine.columns = [i for i in range(0, 44)]
    print(combine)

    # file_io.write_feature_info(feature_dict)
