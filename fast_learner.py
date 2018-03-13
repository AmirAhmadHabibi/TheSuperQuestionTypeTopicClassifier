# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, label_ranking_loss
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
import sys
from sklearn.svm import SVC
import pickle


def learn_svm():
    wrd = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
    tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')
    typ = pd.read_csv('./Primary_data/type_vector_Q.csv')

    # for col in tpc:
    #     print(col, np.unique(tpc[col]))
    # for col in typ:
    #     print(col, np.unique(typ[col]))
    tpc.drop('tpc41', axis=1, inplace=True)
    tpc.drop('tpc42', axis=1, inplace=True)

    # wrd = pd.read_csv('./StackExchange_data/data_1000word.csv')
    # tpc = pd.read_csv('./StackExchange_data/data_tags.csv')
    # topics = pd.read_csv('./StackExchange_data/tags.csv')
    # words_vector = pd.read_csv('./StackExchange_data/1000words.csv', header=None, names={'term'})
    # tpc_cols_list = list(topics['term'])
    # wrd_cols_list = list(words_vector['term'])
    # tpc = tpc[tpc_cols_list].values
    # wrd = wrd[wrd_cols_list].values
    print('read')

    print(wrd.shape)
    print(tpc.shape)
    print(typ.shape)

    topic_classifier = BinaryRelevance(classifier=SVC(kernel='linear', probability=True), require_dense=[True, True])
    topic_classifier.fit(wrd.values, tpc.values)

    # with open('./StackExchange_data/topic_classifier.pkl', 'wb') as sub_class_file:
    #     pickle.dump(topic_classifier, sub_class_file)
    with open('./Primary_data/topic_classifier.pkl', 'wb') as outfile:
        pickle.dump(topic_classifier, outfile)

    type_classifier = BinaryRelevance(classifier=SVC(kernel='linear', probability=True), require_dense=[False, True])
    type_classifier.fit(wrd.values, typ.values)

    with open('./Primary_data/type_classifier.pkl', 'wb') as outfile:
        pickle.dump(type_classifier, outfile)

    print('Saved the pickles!')




learn_svm()
