# coding=utf-8
import pandas as pd
import numpy as np


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


def count_words():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    words = pd.DataFrame(columns=('term', 'num'))

    # iterate on questions
    for q_id, qstn in questions.iterrows():
        if q_id % 100 == 0: print q_id  # printing progress
        word_list = get_word_list(qstn['sentence'])

        # iterate on words of each question
        for word, f in word_list:
            if words[words['term'] == word].empty:
                words = words.append({'term': word, 'num': 1},
                                     ignore_index=True)
            else:
                words.loc[words['term'] == word, 'num'] = words.loc[words['term'] == word, 'num'] + 1
    return words


def save_vector():
    # count and write to file
    # words = count_words()
    # words.to_csv('words_with_count.csv', sep=';', index=False)

    # pick the first 1000
    words = pd.read_csv('words_with_count.csv', delimiter=';')
    stop_words = pd.read_csv('PersianStopWordList.txt', delimiter=';', header=None)
    words = words.dropna(subset={'term'})
    # remove the stop words
    for stop_word in stop_words.as_matrix():
        words = words[words.term != stop_word[0]]
    # remove numbers
    for i, word in words.iterrows():
        if word['term'].isdigit():
            words = words[words.term != word['term']]
    # sort by frequency number
    words = words.sort_values(by='num', ascending=False)
    words = words.reset_index(drop=True)
    # save first 1000 to file
    words = words[:1000]
    words = words[['term']]
    words.to_csv('words_vector.csv', sep=';', index=False)


def replace_bad_characters():
    items = pd.read_csv('result_filtered.csv', delimiter=';')
    for i, row in items.iterrows():
        term = items.loc[i, 'title']
        if str(term) == 'nan':
            continue
        term = term.replace('ي', 'ی')
        term = term.replace('ى', 'ی')
        term = term.replace('ك', 'ک')
        term = term.replace(' ', ' ')
        term = term.replace('‌', ' ')
        term = term.replace('‏', ' ')
        items.loc[i, 'title'] = term
    items.to_csv('result_filtered.csv', sep=';', index=False)


save_vector()
