import pandas as pd
from gensim.models import word2vec
import numpy as np
import pickle
from random import sample


def data_summary(data):

    """Summary of Data

      Parameters:
      -----------
      data : pandas DataFrame with all variables
      response: print column names, dimensions, first 20 lines, statistics (mean, std., min, max) of data
      Returns:
      --------
      Null
    """
    print("Columns of Dataset")
    print(data.columns)

    print("Shape of Data")
    print(data.shape)

    print("First eight lines")
    print(data.tail(8))

    print("Statistics of Data")
    print(data.describe())


def create_array_of_vectors(model,data):

    """
    create array of unique word vectors of training data from 'gensim' model of GoogleNews-vectors,
    and position of each vector is stored as 'id' in 'word_to_id' and 'id_to_word' dictionaries.
    Each word has a vector of (1,300) dimension

    Parameters:
    -----------
    :param model: gensim model of word vectors created from GoogleNews-vectors
    :param train_data: train data for which embedding has to be created

    Returns:
    --------
    :return: array of vectors (embedding array) of unique words of the train dataset, word to id dictionary, id to word dictionary
    """

    embedding_array = []
    word_to_id_dict = {}
    id_to_word_dict = {}

    counter = 0

    for index, row in data.iterrows():
        question1_words = str(row['question1']).lower().replace('?', '').split(' ')
        question2_words = str(row['question2']).lower().replace('?', '').split(' ')

        question_words_concat = question1_words + question2_words

        for j in range(len(question_words_concat)):
            if question_words_concat[j] not in word_to_id_dict.keys():
                if question_words_concat[j] in model:
                    embedding_array.append(model[question_words_concat[j]])
                    word_to_id_dict[question_words_concat[j]] = counter
                    id_to_word_dict[counter] = question_words_concat[j]
                    counter += 1

    embedding_array.append(model['ukn'])
    word_to_id_dict['ukn'] = counter
    id_to_word_dict[counter] = 'ukn'
    counter += 1


    embedding_array = np.array(embedding_array)

    return embedding_array,id_to_word_dict,word_to_id_dict


def data_preparation(data,word_to_id_dict):
    """
    preparation of data for modeling: replace each word in question by position of it's vector position
    in vector aaray (embedding array)

    Parameters:
    -----------
    :param data: data for modeling
    :param word_to_id_dict: position of words in vector array (embedding array)

    Returns:
    --------
    :return: position of word vectors in place of each words in question1 and question2 fields
    """
    processed_data = []

    for index, row in data.iterrows():
        print(index)
        question1_words = str(row['question1']).lower().replace('?', '').split(' ')
        question2_words = str(row['question2']).lower().replace('?', '').split(' ')

        question1_words_list = []
        question2_words_list = []
        question_word_concat_list = []

        for i in range(len(question1_words)):
            if question1_words[i] in word_to_id_dict.keys():
                question1_words_list.append(word_to_id_dict[question1_words[i]])
            else:
                question1_words_list.append(word_to_id_dict['ukn'])

        for i in range(len(question2_words)):
            if question2_words[i] in word_to_id_dict.keys():
                question2_words_list.append(word_to_id_dict[question2_words[i]])
            else:
                question2_words_list.append(word_to_id_dict['ukn'])

        question_word_concat_list.append(question1_words_list)
        question_word_concat_list.append(question2_words_list)
        #question_word_concat_list.append(row['is_duplicate'])
        question_word_concat_list.append(row['test_id'])

        question_word_concat_list_tuple = tuple(question_word_concat_list)
        processed_data.append(question_word_concat_list_tuple)


    return processed_data

def split_data_to_train_dev(data):
    """
    split datasets into train and dev data

    Parameters:
    -----------
    :param data: dataset


    Returns:
    --------
    :return: train and dev data
    """

    train_data = data[:int((len(data) + 1) * .90)]  # Remaining 90% to training set
    dev_data = data[int(len(data) * .90 + 1):]

    return train_data,dev_data


def distribution_of_dev_data(data):
    """
    check distrbution of duplicate and non duplicate classes

     Parameters:
    -----------
    :param data: dev dataset

    Returns:
    --------
    :return: NULL

    """
    zero=0
    non_zero=0

    print(len(data))
    for i in range(len(data)):
        if data[i][2]==0:
            zero+=1
        else:
            non_zero+=1

    print("% of non duplicate classes : "+zero/len(data))
    print("% of duplicate classes :"+ (non_zero/len(data)))


def main():

    ##### Load Dataset
    train_data = pd.read_csv('train.csv', encoding='utf-8')
    test_data = pd.read_csv('./data/test.csv', encoding='utf-8')
    print(test_data.columns)

    #### Visualize and Undestand data
    data_summary(train_data)
    data_summary(test_data)


    ###### create a model of word vectors from GoogleNews corpus
    model=word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)

    ##### Create embedding lookup
    embedding_array,id_to_word_dict,word_to_id_dict=create_array_of_vectors(model,train_data)

    with open('embedding_array.p', 'wb') as fp:
        pickle.dump(embedding_array, fp)

    with open('id_to_word_dict.p', 'wb') as fp:
        pickle.dump(id_to_word_dict, fp)

    with open('word_to_id_dict.p', 'wb') as fp:
        pickle.dump(word_to_id_dict, fp)



    ####### Data Preparation

    with open('./data/word_to_id_dict.p','rb') as fp:
        word_to_id_dict=pickle.load(fp)

    ### Training Data
    processed_data=data_preparation(train_data,word_to_id_dict)
    with open('filename.p','wb') as fp:
        pickle.dump(processed_data,fp)

    ### Test Data
    processed_data = data_preparation(test_data, word_to_id_dict)
    with open('filename_new_test_data.p', 'wb') as fp:
        pickle.dump(processed_data, fp)


    ### split data into train and dev data
    with open('train_data.p', 'rb') as fp:
        train_data = pickle.load(fp)

    #### Shuffle data for unbiased distribution of train and dev data
    train_data=sample(train_data, len(train_data))

    ### Split training data into dev and train data
    train_data,dev_data=split_data_to_train_dev(train_data)

    with open('train_data_modelling.p', 'wb') as fp:
         pickle.dump(train_data,fp)

    with open('dev_data.p', 'wb') as fp:
        pickle.dump(dev_data, fp)


    ## Check distribution of classes in dev data
    distribution_of_dev_data(dev_data)



if __name__ == '__main__':
    main()