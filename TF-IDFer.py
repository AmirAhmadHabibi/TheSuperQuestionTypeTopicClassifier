# coding=utf-8
import pandas as pd


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    # TODO: replace unnecessary characters
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


def go():
    questions = pd.read_csv('result.csv', delimiter=';')
    questions = questions['sentence']
    # num_questions_containing - frequency_in_question
    words = pd.DataFrame(columns=('term', 'IDF', 'TF', 'num_q', 'f_in_q'), index='word')

    for q_id, qstn in questions.iterrows():
        for word, f in get_word_list(qstn):
            if words[words['word'] == word].empty:
                words.append({'term': word, 'IDF': 0, 'TF': 0, 'num_q': 0, 'f_in_q': []})
            words.loc[word, 'f_in_q'] = f
            words.loc[word, 'num_q'] = words.loc[word, 'num_q'] + 1


go()