# coding=utf-8
import numpy
import pandas as pd
import sys


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    # TODO: use a better method
    return qstn.split()


def concat_word2vec_subjs():
    w2v = pd.read_csv('questions-word2vec.txt', header=None)
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    subjs = pd.read_csv('./1_combine_tags/subjs-result.csv', delimiter=';')
    trn = pd.DataFrame()
    for i, subj in subjs.iterrows():
        trn['sub' + str(subj['id'])] = 0
    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, subj in subjs.iterrows():
            sub_id = str(subj['id'])
            trn.loc[i, 'sub' + sub_id] = 0
            if qrow['subject1'] == subj['tag'] or qrow['subject2'] == subj['tag'] or qrow['subject3'] == subj['tag']:
                trn.loc[i, 'sub' + sub_id] = 1
                # print i, '----',trn.loc[i]

    result = pd.concat([trn, w2v], axis=1)
    result.to_csv('subj-word2vec.csv', index=False, header=None)


def concat_word2vec_types():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    w2v = pd.read_csv('questions-word2vec.txt', header=None)
    types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, typ in types.iterrows():
        train['typ' + str(typ['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, typ in types.iterrows():
            typ_id = str(typ['id'])
            train.loc[i, 'typ' + typ_id] = 0
            if qrow['type1'] == typ['tag'] or qrow['type2'] == typ['tag'] or qrow['type3'] == typ['tag']:
                train.loc[i, 'typ' + typ_id] = 1

    result = pd.concat([train, w2v], axis=1)
    result.to_csv('type-word2vec.csv', index=False)


def create_1000word_vector():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    words_vector = pd.read_csv('words_vector.csv')

    # create dataframe
    train = pd.DataFrame(dtype=object)
    for wrd in words_vector['term'].as_matrix():
        train[wrd] = 0
    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        train.loc[i, words_vector['term'][0]] = 0
        # set occurrence values
        for word in tokenise(qrow['sentence']):
            if word in train:
                train.loc[i, word] = 1

    print train.shape
    train = train.fillna(0)

    # rename columns
    number = 0
    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')
        train = train.rename(columns={col: 'wrd' + str(number)})
        number += 1
    print train.shape
    train.to_csv('1000word_vector_Q.csv', index=False)


def create_type_vector():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, typ in types.iterrows():
        train['typ' + str(typ['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, typ in types.iterrows():
            typ_id = str(typ['id'])
            train.loc[i, 'typ' + typ_id] = 0
            if qrow['type1'] == typ['tag'] or qrow['type2'] == typ['tag'] or qrow['type3'] == typ['tag']:
                train.loc[i, 'typ' + typ_id] = 1

    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')

    train.to_csv('type_vector_Q.csv', index=False)


def create_subject_vector():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    subjs = pd.read_csv('./1_combine_tags/subjs-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, sub in subjs.iterrows():
        train['sub' + str(sub['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, subj in subjs.iterrows():
            sub_id = str(subj['id'])
            train.loc[i, 'sub' + sub_id] = 0
            if qrow['subject1'] == subj['tag'] or qrow['subject2'] == subj['tag'] or qrow['subject3'] == subj['tag']:
                train.loc[i, 'sub' + sub_id] = 1

    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')

    train.to_csv('subject_vector_Q.csv', index=False)


def concat_1000vec_type_cat():
    wrd = pd.read_csv('1000word_vector_Q.csv')
    subj = pd.read_csv('subject_vector_Q.csv')
    type = pd.read_csv('type_vector_Q.csv')
    result = pd.concat([type, subj, wrd], axis=1)
    result.to_csv('type-subj-wrd_Q.csv', index=False)


def create_arff_header():
    header = '@relation \'CQA\'\n\n'
    for i in range(0, 12):
        header += '@attribute typ' + str(i) + ' numeric\n'
    for i in range(0, 24):
        header += '@attribute sub' + str(i) + ' {0,1}\n'
    for i in range(0, 1000):
        header += '@attribute wrd' + str(i) + ' numeric\n'
    header += '\n@data\n'
    print header


# create_arff_header()
# concat_word2vec_subjs()
# concat_word2vec_types()

# create_1000word_vector()
# create_type_vector()
# create_subject_vector()
# concat_1000vec_type_cat()
