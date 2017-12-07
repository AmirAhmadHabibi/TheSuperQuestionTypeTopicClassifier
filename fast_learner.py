# coding=utf-8
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import pickle


def learn_svm():
    wrd = pd.read_csv('1000word_vector_Q.csv')
    tpc = pd.read_csv('topic_vector_Q.csv')
    # typ = pd.read_csv('type_vector_Q.csv')

    subject_classifier = BinaryRelevance(classifier=SVC(probability=True), require_dense=[False, True])
    subject_classifier.fit(wrd, tpc)

    sub_class_file = open('tpc_class_file.pkl', 'wb')
    pickle.dump(subject_classifier, sub_class_file)
    sub_class_file.close()

    # type_classifier = BinaryRelevance(classifier=SVC(probability=True), require_dense=[False, True])
    # type_classifier.fit(wrd, typ)
    #
    # typ_class_file = open('typ_class_file.pkl', 'wb')
    # pickle.dump(type_classifier, typ_class_file)
    # typ_class_file.close()

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
    return prediction[0:3]


def do_questions():
    topics = pd.read_csv('./Porsak_data/topic_list.csv')
    # types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')
    words_vector = pd.read_csv('words_vector.csv')
    # load the classifiers
    tpc_class_file = open('tpc_class_file.pkl', 'rb')
    topic_classifier = pickle.load(tpc_class_file)
    # typ_class_file = open('typ_class_file.pkl', 'rb')
    # type_classifier = pickle.load(typ_class_file)
    # get a question and predict its subject
    while True:
        question = raw_input('Enter a question:')
        question_vector = question_word_vector(question, words_vector)

        predictions = predict_subject_type(topic_classifier, question_vector, topics['topic'])
        print('topic: ')
        print predictions

        # predictions = predict_subject_type(type_classifier, question_vector, types['tag'])
        # print('type: ')
        # print predictions


# learn_svm()
do_questions()
