import numpy as np
import pandas as pd
import file_io
from feature_extraction import initialize_feature_dataframes, convert_feature_list_to_df


def noccurance(data_list, sequences):
    """This function will count the number of occurrences of a given sequence within the sequences list
    
    Parameters
    ----------
    sequences : list
    data_list: pandas dataframe containing protein sequences 
    
    Returns data frame containing counts of sequence occurance
    -------
    """
    ndata = {'protein':[]}
    #Creates dataframe for count records
    for sequence in sequences:
        ndata[str(sequence)]=[]
    dataframe = pd.DataFrame(ndata)
    #Iterates through dataframe to cound the amount of occurances 
    for sequence in sequences:
        ph = 0
        for protein in cleaned_data['n_data'][3]:
            count = protein.count(sequence)
            dataframe.loc[ph,"protein"] = protein
            dataframe.loc[ph,sequence] = count
            ph = ph+1
    
    return dataframe
    
        
def goccurance(data_list, sequences):
    """This function will count the number of occurrences of a given sequence within the sequences list
    
    Parameters
    ----------
    sequences : list
    data_list: pandas dataframe containing protein sequences 
    
    Returns data frame containing counts of sequence occurance
    -------
    """
    #Creates dataframe for count records
    gdata = {'protein':[]}
    for sequence in sequences:
        gdata[str(sequence)]=[]
    dataframe = pd.DataFrame(gdata)
    #Iterates through dataframe to cound the amount of occurances 
    for sequence in sequences:
        ph = 0
        for protein in cleaned_data['g_data'][3]:
            #print(protein)
            count = protein.count(sequence)
            dataframe.loc[ph,"protein"] = protein
            dataframe.loc[ph,sequence] = count
            ph = ph+1
    return dataframe

  
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

    
   
    # sequences = ["MPT","SRL","GVWT","AEF","TGF","HHV","KNF","LAK","AGL","SYN","DAL"]
    
    print(noccurance(cleaned_data,sequences))
    #print(goccurance(cleaned_data,sequences))
    