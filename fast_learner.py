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


def tokenise(question):
    question = question.replace('ي', 'ی')
    question = question.replace('ى', 'ی')
    question = question.replace('ك', 'ک')
    question = question.replace(' ', ' ')
    question = question.replace('‌', ' ')
    question = question.replace('‏', ' ')
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=']:
        question = question.replace(ch, ' ')
    return question.split()


def question_word_vector(question, words_vector):
    question_vector = pd.DataFrame(dtype=object)
    for wrd in words_vector['term'].as_matrix():
        question_vector[wrd] = 0
    # build the question_vector
    question_vector.loc[0, words_vector['term'][0]] = 0
    # set occurrence values
    for word in tokenise(question):
        if word in question_vector:
            question_vector.loc[0, word] = 1
    question_vector = question_vector.fillna(0)
    return question_vector


def predict_subject_type(classifier, question_vector, labels):
    # predict
    prediction_probabilities = np.array(classifier.predict_proba(question_vector).todense())[0]
    # add labels to probabilities
    prediction = pd.concat([labels, pd.DataFrame(prediction_probabilities)], axis=1)
    # sort by probability
    prediction = prediction.rename(columns={0: 'prob'})
    prediction = prediction.sort_values('prob', ascending=False)
    prediction = prediction.reset_index(drop=True)

    if not prediction[prediction['prob'] >= 0.1].empty:
        return prediction[prediction['prob'] >= 0.1]
    return prediction[0:5]


def do_questions():
    topics = pd.read_csv('./Porsak_data/topic_list.csv')
    # types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')
    words_vector = pd.read_csv('words_vector.csv')
    # load the classifiers
    with open('tpc_class_file.pkl', 'rb') as tpc_class_file:
        topic_classifier = pickle.load(tpc_class_file)
    # typ_class_file = open('typ_class_file.pkl', 'rb')
    # type_classifier = pickle.load(typ_class_file)
    # get a question and predict its subject
    while True:
        question = input('Enter a question:')
        question_vector = question_word_vector(question, words_vector)

        predictions = predict_subject_type(topic_classifier, question_vector, topics['topic'])
        print('topic: ')
        print(predictions)

        # predictions = predict_subject_type(type_classifier, question_vector, types['tag'])
        # print('type: ')
        # print predictions


def eval_porsak_questions():
    topics = pd.read_csv('./Porsak_data/topic_list.csv')
    words_vector = pd.read_csv('words_vector.csv')

    with open('tpc_class_file.pkl', 'rb') as tpc_class_file:
        topic_classifier = pickle.load(tpc_class_file)

    questions = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')

    total = 0
    TP = 0
    TP_5 = 0
    for i, qrow in questions.iterrows():
        if i % 10 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        try:
            question_vector = question_word_vector(qrow['content'], words_vector)
            predictions = predict_subject_type(topic_classifier, question_vector, topics['topic'])

            total += 1
            if qrow['topic'] == predictions['topic'][0]:
                TP += 1
            for p in predictions['topic']:
                if qrow['topic'] == p:
                    TP_5 += 1
                    break
        except Exception as e:
            print('\n', e)
            print('--', qrow['content'], qrow)

    print('res: ', TP, total, TP / total)
    print('res3: ', TP_5, total, TP_5 / total)


learn_svm()
# evaluate_model()
# eval_porsak_questions()
# do_questions()
