# coding=utf-8
import pandas as pd
from math import log


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    # TODO: use a better method
    return qstn.split()


def get_word_list(qstn):
    w_list = []
    flag = 0
    for word in tokenise(qstn):
        for tpl in w_list:
            if tpl[0] == word:
                tpl[1] = tpl[1] + 1
                flag = 1
                break
        if flag == 0:
            w_list.append([word, 1])
        flag = 0
    return w_list


def extract_words():
    questions = pd.read_csv('result.csv', delimiter=';')
    # num_questions_containing - frequency_in_question
    words = pd.DataFrame(columns=('term', 'TF-IDF', 'num_q', 'f_in_q'))
    q_words_count = []
    # counting the frequency of words in each question and questions containing each word
    for q_id, qstn in questions.iterrows():
        if q_id % 100 == 0: print q_id

        qstn = qstn['sentence']
        word_list = get_word_list(qstn)
        # q_words_count.append(len(word_list))
        for word, f in word_list:
            if words[words['term'] == word].empty:
                words = words.append({'term': word, 'TF-IDF': 0, 'num_q': 1, 'f_in_q': [[q_id, f]]},
                                     ignore_index=True)
            else:
                words.loc[words['term'] == word, 'f_in_q'].iloc[0].append([q_id, f])
                words.loc[words['term'] == word, 'num_q'] = words.loc[words['term'] == word, 'num_q'] + 1

    return q_words_count, words


def go():
    # q_words_count, words = extract_words()
    words = pd.read_csv('words.csv', delimiter=';')
    words = words[['term', 'TF-IDF', 'num_q', 'f_in_q']]
    questions = pd.read_csv('result.csv', delimiter=';')
    q_words_count = []
    for q_id, qstn in questions.iterrows():
        q_words_count.append(len(tokenise(qstn['sentence'])))

    # calculating TF-IDF
    for index, word in words.iterrows():
        if index % 1000 == 0: print 'word', index
        idf = log(len(q_words_count) / word['num_q'])
        tf = []
        tf_sum = 0
        for f in eval(word['f_in_q']):
            tf_tmp = float(f[1]) / q_words_count[f[0]]
            tf.append([f[0], tf_tmp])
            tf_sum += tf_tmp
        words.loc[index, 'TF-IDF'] = (float(tf_sum) / len(q_words_count)) * idf * 1000000
        # print word['TF-IDF']

    words.to_csv("words.csv", sep=';')
    print words


def do_stuff():
    words = pd.read_csv('words.csv', delimiter=';')
    print words.describe()
    # words.drop('Unnamed: 0', axis='columns')
    words = words[['term', 'TF-IDF']]
    print '-----------'
    print words.describe()


go()
# do_stuff()
