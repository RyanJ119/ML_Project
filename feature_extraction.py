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


def special_trigram_occurance(seq_data_list):
    special_trigrams = [
        "MPT", "SRL", "GVWT", "AEF", "TGF", "HHV", "KNF", "LAK", "AGL", "SYN", "DAL"]

    rows = len(seq_data_list)
    cols = len(special_trigrams)
    special_trigram_mat = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            special_trigram_mat[i][j] = seq_data_list[i].count(special_trigrams[j])
    return special_trigram_mat


if __name__ == "__main__":
    # testing the above code
    file_name_list = ['g_data.csv', 'n_data.csv']
    test_dict = file_io.read_seq_files(file_name_list)

    cleaned_data = file_io.clean_input(test_dict)

    feature_dict = initialize_feature_dataframes(cleaned_data)

    # sequence data is held in the final col of the dataframe
    g_seq_data = cleaned_data['g_data'].iloc[:, 3].tolist()
    n_seq_data = cleaned_data['n_data'].iloc[:, 3].tolist()

    # ###### gram positive feature extraction section ############
    g_bigram = bi_gram(g_seq_data)
    g_special_trigrams = special_trigram_occurance(g_seq_data)
    g_occur, g_comp = calc_occur_comp(g_seq_data)

    g_bigram_df = convert_feature_list_to_df(g_bigram)
    g_special_trigrams = convert_feature_list_to_df(g_special_trigrams)
    g_occur_df = convert_feature_list_to_df(g_occur)
    g_comp_df = convert_feature_list_to_df(g_comp)

    g_existing_features = feature_dict['g_data_features']
    g_updated_features = pd.concat(
        [g_existing_features, g_occur_df, g_comp_df, g_bigram_df, g_special_trigrams], axis=1)

    # print(g_updated_features)
    #######################################################
    
    # #### gram negative feature extraction ###################
    n_bigram = bi_gram(n_seq_data)
    n_special_trigrams = special_trigram_occurance(n_seq_data)
    n_occur, n_comp = calc_occur_comp(n_seq_data)

    n_bigram_df = convert_feature_list_to_df(n_bigram)
    n_special_trigrams = convert_feature_list_to_df(n_special_trigrams)
    n_occur_df = convert_feature_list_to_df(n_occur)
    n_comp_df = convert_feature_list_to_df(n_comp)

    n_existing_features = feature_dict['n_data_features']
    n_updated_features = pd.concat(
        [n_existing_features, n_occur_df, n_comp_df, n_bigram_df, n_special_trigrams], axis=1)

    # print(n_updated_features)
    ###################################################
    
    # add headers to the file to correspond to provided files
    num_features = g_updated_features.shape[1]  # shape returns tuple, we want # cols
    header_list = ['Fold']
    # we start at 1 because first entry is the fold!
    for i in range(1, num_features):
        header_list.append('V'+str(i))

    # add headers to the dataframes:
    g_updated_features.columns = header_list
    n_updated_features.columns = header_list
    # update the dictionary of features:
    feature_dict['g_data_features'] = g_updated_features
    feature_dict['n_data_features'] = n_updated_features

    # write the features to files
    file_io.write_feature_info(feature_dict)