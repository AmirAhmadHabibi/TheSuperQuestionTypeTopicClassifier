# coding=utf-8
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import pickle

words_vector = pd.read_csv('words_vector.csv')
subjs = pd.read_csv('subjs-result.csv', delimiter=';')
# types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')


def learn_svm():
    wrd = pd.read_csv('1000word_vector_Q.csv')
    sub = pd.read_csv('subject_vector_Q.csv')
    # typ = pd.read_csv('type_vector_Q.csv')

    subject_classifier = BinaryRelevance(classifier=SVC(probability=True), require_dense=[False, True])
    subject_classifier.fit(wrd, sub)

    sub_class_file = open('sub_class_file.pkl', 'wb')
    pickle.dump(subject_classifier, sub_class_file)
    sub_class_file.close()


def tokenise(question):
    question = question.replace('ي', 'ی')
    question = question.replace('ى', 'ی')
    question = question.replace('ك', 'ک')
    question = question.replace(' ', ' ')
    question = question.replace('‌', ' ')
    question = question.replace('‏', ' ')
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        question = question.replace(ch, ' ')
    return question.split()


def predict_subject(subject_classifier, question):
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

    # predict
    prediction_probabilities = np.array(subject_classifier.predict_proba(question_vector).todense())[0]

    prediction = pd.concat([subjs['tag'], pd.DataFrame(prediction_probabilities)], axis=1)
    prediction = prediction.rename(columns={0: 'prob'})
    prediction = prediction.sort_values('prob', ascending=False)
    prediction = prediction.reset_index(drop=True)

    print prediction[prediction['prob'] >= 0.1]


def go():
    sub_class_file = open('sub_class_file.pkl', 'rb')
    subject_classifier = pickle.load(sub_class_file)
    question = raw_input('Enter a question:')
    predict_subject(subject_classifier, question)


# learn_svm()
go()
