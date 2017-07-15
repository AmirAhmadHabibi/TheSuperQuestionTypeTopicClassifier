import numpy
import pandas as pd
import sys


def build_train_data_for_subjs():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    words_vector = pd.read_csv('words_vector.csv')
    subjs = pd.read_csv('./1_combine_tags/subjs-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame()
    for word in words_vector['term'].as_matrix():
        train[word] = 0
    train['subject'] = 0

    # remove questions without subjects
    for i, qrow in questions.iterrows():
        if str(qrow['subject1']) == 'nan' or qrow['subject1'] == numpy.nan:
            questions = questions[questions['id'] != qrow['id']]

    # build the train data
    for i, qrow in questions.iterrows():
        sys.stdout.write('\r'+'processed question '+str(i))
        # set the subject id from subjs file as the subject value
        train.loc[i, 'subject'] = subjs['id'][subjs['tag'] == qrow['subject1']].as_matrix()[0]
        # set occurrence values
        for word in words_vector['term'].as_matrix():
            if word in qrow['sentence']:
                train.loc[i, word] = 1
            else:
                train.loc[i, word] = 0

    # rename columns
    number = 0
    for col in train:
        if col != 'subject':
            train = train.rename(columns={col: 'a' + str(number)})
            number += 1

    train.to_csv('train.csv', index=False)


build_train_data_for_subjs()
