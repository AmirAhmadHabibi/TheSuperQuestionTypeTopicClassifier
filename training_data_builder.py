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
            sub_id = str(subj['id'])
            train.loc[i, 'sub' + sub_id] = 0
            if qrow['subject1'] == subj['tag'] or qrow['subject2'] == subj['tag'] or qrow['subject3'] == subj['tag']:
                train.loc[i, 'sub' + sub_id] = 1
        # set occurrence values
        for word in words_vector['term'].as_matrix():
            if word in qrow['sentence']:
                train.loc[i, word] = 1
            else:
                train.loc[i, word] = 0
                # print train.loc[i]

    # rename columns
    number = 0
    for col in train:
        if 'sub' not in col:
            train = train.rename(columns={col: 'a' + str(number)})
            number += 1

    train.to_csv('subject.csv', index=False)


def build_train_data_for_types():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    qvectors = pd.read_csv('vector1000.csv')
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

    result = pd.concat([train, qvectors], axis=1)
    result.to_csv('train-type.csv', index=False)


def create_arff_header():
    header = '@relation \'CQA\'\n\n'
    for i in range(0, 12):
        header += '@attribute s' + str(i) + ' {0,1}\n'
    for i in range(12, 313):
        header += '@attribute a' + str(i) + ' numeric\n'
    header += '\n@data\n'
    print header


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
    for word in words_vector['term'].as_matrix():
        train[word] = 0
    # remove questions without subjects
    for i, qrow in questions.iterrows():
        if str(qrow['subject1']) == 'nan' or qrow['subject1'] == numpy.nan:
            questions = questions[questions['id'] != qrow['id']]
    # build the train data
    for i, qrow in questions.iterrows():
        sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence values
        for word in words_vector['term'].as_matrix():
            if word in qrow['sentence']:
                train.loc[i, word] = 1
            else:
                train.loc[i, word] = 0
    # rename columns
    number = 0
    for col in train:
        train = train.rename(columns={col: 'wrd' + str(number)})
        number += 1

    train.to_csv('1000word_vector_Q.csv', index=False)

# build_train_data_for_subjs()
# create_arff_header()
# concat_word2vec_subjs()
# build_train_data_for_types()
# concat_word2vec_types()
# concat_1000vec_type_cat()
create_1000word_vector()
