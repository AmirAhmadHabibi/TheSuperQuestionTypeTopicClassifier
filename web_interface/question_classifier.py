import pandas as pd
import numpy as np
import pickle


class QuestionClassifier:
    def __init__(self):
        with open('../Primary_data/topic_classifier.pkl', 'rb') as infile:
            self.topic_classifier_bow = pickle.load(infile)
        with open('../Primary_data/type_classifier.pkl', 'rb') as infile:
            self.type_classifier_bow = pickle.load(infile)

        with open('../Primary_data/topic_classifier-w2v.pkl', 'rb') as infile:
            self.topic_classifier_w2v = pickle.load(infile)
        with open('../Primary_data/type_classifier-w2v.pkl', 'rb') as infile:
            self.type_classifier_w2v = pickle.load(infile)

        with open('../word2vec/Mixed/w2v_per.pkl', 'rb') as infile:
            self.w2v = pickle.load(infile)
            self.w2v_length = 100

        self.topics = pd.read_csv('../Porsak_data/topic_list.csv')['topic']
        self.types = pd.read_csv('../1_combine_tags/types-result.csv', delimiter=';')['tag']
        self.bag_of_words = pd.read_csv('../Primary_data/words_vector.csv')
        self.stop_words = set(pd.read_csv('../Primary_data/PersianStopWordList.txt', header=None)[0])

    def bow_classify(self, question):
        question_vector = np.array(self.__question_bow_vector(question)).reshape(1, -1)

        topic_predictions = self.__predict(self.topic_classifier_bow, question_vector, self.topics)
        type_predictions = self.__predict(self.type_classifier_bow, question_vector, self.types)

        return topic_predictions, type_predictions

    def w2v_classify(self, question):
        question_vector = np.array(self.__question_w2v_vector(question)).reshape(1, -1)

        topic_predictions = self.__predict(self.topic_classifier_w2v, question_vector, self.topics)
        type_predictions = self.__predict(self.type_classifier_w2v, question_vector, self.types)

        return topic_predictions, type_predictions

    def __question_bow_vector(self, question):
        question_vector = pd.DataFrame(dtype=object)
        for wrd in self.bag_of_words['term'].as_matrix():
            question_vector[wrd] = 0
        # build the question_vector
        question_vector.loc[0, self.bag_of_words['term'][0]] = 0
        # set occurrence values
        for word in self.__tokenise(question):
            if word in question_vector:
                question_vector.loc[0, word] = 1
        question_vector = question_vector.fillna(0)
        return question_vector

    def __question_w2v_vector(self, question):
        sum_array = np.zeros(self.w2v_length)
        number_of_words = 0
        for word in self.__tokenise(question):
            if word not in self.stop_words and word in self.w2v:
                number_of_words += 1
                sum_array += self.w2v[word]
        if number_of_words > 0:
            return list(sum_array / number_of_words)
        return list(sum_array)

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

    @staticmethod
    def __predict(classifier, question_vector, labels):
        # predict
        prediction_probabilities = np.array(classifier.predict_proba(question_vector).todense())[0]
        # add labels to probabilities
        prediction = pd.concat([labels, pd.DataFrame(prediction_probabilities)], axis=1)
        # sort by probability
        prediction = prediction.rename(columns={0: 'prob'})
        prediction = prediction.sort_values('prob', ascending=False)
        prediction = prediction.reset_index(drop=True)
        prediction = prediction.fillna(0.0)
        # if not prediction[prediction['prob'] >= 0.1].empty:
        #     return prediction[prediction['prob'] >= 0.1]
        # return prediction[0:5]
        return prediction
