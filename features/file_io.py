import pandas as pd


def read_seq_files(file_name_list):
    """Read in each csv file in file_name_list, returns dictionary of file_name:df of csv
    
    Parameters
    ----------
    file_name_list : list of strings
        holds a list of strings of the files that we want to
        read from data/
    
    Returns
    -------
    seq_dict : dictionary
        key is file name and value is pd dataframe of csv
        data of bacteria sequences
    """

    directory_of_files = '../data/'

    seq_dict = {}
    for file_name in file_name_list:
        # read csv from directory
        data = pd.read_csv(directory_of_files + file_name, header=None)
        
        # key for dictionary will be everything before the extension
        file_no_extension = file_name.split('.')[0]

        # add an entry to the dictionary
        seq_dict[file_no_extension] = data

    return seq_dict


def read_feature_files(file_name_list):
    """This function will read in each file provided in file_name_list
    and create a dictionary that holds the data stored in those files
    keys for the dictionary will be the filename (no extension)
    values will be pandas dataframes
    
    Parameters
    ----------
    file_name_list : list of strings
        holds the files to read in from the data subfolder
    
    Returns
    -------
    dictionary
        dictionary has keys of file names with values of pandas
        dataframes that store the data
    """
    directory_of_files = '../data/'
    feature_dict = {}

    for file_name in file_name_list:
        data = pd.read_csv(directory_of_files + file_name)
        file_no_extension = file_name.split('.')[0]

        feature_dict[file_no_extension] = data

    return feature_dict
    

def clean_input(seq_dict):
    """Some of the excel files have extra cols at the end, we will remove them
    
    Parameters
    ----------
    seq_dict : dictionary
        keys are names of files and values are pd dataframes of .csv data from key file

    Returns
    ---------
    cleaned_dictionary : dictionary
        we truncate each dataframe to only the first four columns
    """
    cleaned_dictionary = {}
    for key in seq_dict.keys():
        # truncate each dataframe to only the first 4 columns
        cleaned_dictionary[key] = seq_dict[key].iloc[:, 0:4]

    return cleaned_dictionary


def write_feature_info(feature_dict):
    """This function will write the features to a csv file
        
    Parameters
    ----------
    feature_dict : dictionary
        keys are names of files that the features came from, values are the
        features of the data
    """
    directory_of_files = '../data/'

    for key in feature_dict:
        save_loc = directory_of_files + key + '.csv'
        feature_dict[key].to_csv(save_loc, index=False)


if __name__ == "__main__":
    # testing the above code
    file_name_list = ['g_data.csv', 'n_data.csv']
    test_dict = read_seq_files(file_name_list)

    print(test_dict)

    cleaned_data = clean_input(test_dict)
    print(cleaned_data)
