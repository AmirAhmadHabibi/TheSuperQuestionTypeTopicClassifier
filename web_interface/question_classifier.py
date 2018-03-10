import pandas as pd
import numpy as np
import pickle


class QuestionClassifier:
    def __init__(self):
        with open('./Primary_data/topic_classifier.pkl', 'rb') as infile:
            self.topic_classifier = pickle.load(infile)
        with open('./Primary_data/type_classifier.pkl', 'rb') as infile:
            self.type_classifier = pickle.load(infile)
        self.topics = pd.read_csv('./Porsak_data/topic_list.csv')
        self.types = pd.read_csv('./1_combine_tags/types-result.csv', delimiter=';')
        self.words_vector = pd.read_csv('./Primary_data/words_vector.csv')

    def do_questions(self, question):
        question_vector = self.__question_word_vector(question)

        topic_predictions = self.__predict(self.topic_classifier, question_vector, self.topics['topic'])
        type_predictions = self.__predict(self.type_classifier, question_vector, self.types['tag'])

        return topic_predictions, type_predictions

    def __question_word_vector(self, question):
        question_vector = pd.DataFrame(dtype=object)
        for wrd in self.words_vector['term'].as_matrix():
            question_vector[wrd] = 0
        # build the question_vector
        question_vector.loc[0, self.words_vector['term'][0]] = 0
        # set occurrence values
        for word in self.__tokenise(question):
            if word in question_vector:
                question_vector.loc[0, word] = 1
        question_vector = question_vector.fillna(0)
        return question_vector

    @staticmethod
    def __tokenise(question):
        question = question.replace('ي', 'ی')
        question = question.replace('ى', 'ی')
        question = question.replace('ك', 'ک')
        question = question.replace(' ', ' ')
        question = question.replace('‌', ' ')
        question = question.replace('‏', ' ')
        for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>',
                   '؟', '،', '.', '­', '«', '»', '_', '+', '=']:
            question = question.replace(ch, ' ')
        return question.split()

    def __predict(self, classifier, question_vector, labels):
        # predict
        prediction_probabilities = np.array(classifier.predict_proba(question_vector).todense())[0]
        # add labels to probabilities
        prediction = pd.concat([labels, pd.DataFrame(prediction_probabilities)], axis=1)
        # sort by probability
        prediction = prediction.rename(columns={0: 'prob'})
        prediction = prediction.sort_values('prob', ascending=False)
        prediction = prediction.reset_index(drop=True)
        return prediction
