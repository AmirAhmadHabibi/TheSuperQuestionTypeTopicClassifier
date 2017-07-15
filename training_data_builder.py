import numpy
import pandas as pd
import sys


def build_train_data_for_subjs():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    words_vector = pd.read_csv('words_vector.csv')
    subjs = pd.read_csv('./1_combine_tags/subjs-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, subj in subjs.iterrows():
        train['sub' + str(subj['id'])] = 0
    for word in words_vector['term'].as_matrix():
        train[word] = 0

    # remove questions without subjects
    for i, qrow in questions.iterrows():
        if str(qrow['subject1']) == 'nan' or qrow['subject1'] == numpy.nan:
            questions = questions[questions['id'] != qrow['id']]

    # build the train data
    for i, qrow in questions.iterrows():
        sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, subj in subjs.iterrows():
            train.loc[i, 'sub' + str(subj['id'])] = 0
            if qrow['subject1'] == subj['tag']:
                train.loc[i, 'sub' + str(subj['id'])] = 1
            if qrow['subject2'] == subj['tag']:
                train.loc[i, 'sub' + str(subj['id'])] = 1
            if qrow['subject2'] == subj['tag']:
                train.loc[i, 'sub' + str(subj['id'])] = 1
        # set occurrence values
        for word in words_vector['term'].as_matrix():
            if word in qrow['sentence']:
                train.loc[i, word] = 1
            else:
                train.loc[i, word] = 0
        print train.loc[i]

    # rename columns
    number = 0
    for col in train:
        if 'sub' not in col:
            train = train.rename(columns={col: 'a' + str(number)})
            number += 1

    train.to_csv('train.csv', index=False)


def create_arff_header():
    header = '@relation \'CQA\'\n\n'
    for i in range(0, 23):
        header += '@attribute s' + str(i)+' {0,1}\n'
    for i in range(23, 984):
        header += '@attribute a' + str(i)+' numeric\n'

    header+='\n@data\n'
    print header


# build_train_data_for_subjs()
create_arff_header()