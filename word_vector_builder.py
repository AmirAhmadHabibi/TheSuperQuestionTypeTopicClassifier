# coding=utf-8
import pandas as pd
import numpy as np


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    # TODO: use a better method
    return qstn.split()


def count_words():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    questions_2 = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')
    # words = pd.DataFrame(columns=('term', 'num'))
    words = dict()

    # iterate on questions
    for q_id, qstn in questions.iterrows():
        for word in set(tokenise(qstn['sentence'])):
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    # iterate on questions
    for q_id, qstn in questions_2.iterrows():
        for word in set(tokenise(str(qstn['content']) + ' ' + qstn['title'])):
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    words_df = pd.DataFrame(words.items(), columns=['term', 'num'])
    # sort by frequency number
    words_df = words_df.sort_values(by='num', ascending=False)
    words_df = words_df.reset_index(drop=True)
    return words_df


def save_1000word_vector():
    # count and write to file
    words = count_words()
    words.to_csv('words_with_count.csv', sep=';', index=False)

    bad_words_list = {'font', 'family', '&nbsp', 'rgb', '14px', 'p', 'style', 'span', 'size'}
    # pick the first 1000
    stop_words = pd.read_csv('PersianStopWordList.txt', delimiter=';', header=None)
    words = words.dropna(subset={'term'})
    # remove the stop words
    for stop_word in stop_words.as_matrix():
        words = words[words.term != stop_word[0]]
    for bad_word in bad_words_list:
        words = words[words.term != bad_word]
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


def refine_questions():
    # items = pd.read_csv('result_filtered.csv', delimiter=';')
    items = pd.read_csv('./Porsak_data/qa_questions.csv', delimiter=';')
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

        term = items.loc[i, 'content']
        if str(term) == 'nan':
            continue
        term = term.replace('ي', 'ی')
        term = term.replace('ى', 'ی')
        term = term.replace('ك', 'ک')
        term = term.replace(' ', ' ')
        term = term.replace('‌', ' ')
        term = term.replace('‏', ' ')
        items.loc[i, 'content'] = term
    items = items[['topicid', 'topic', 'title', 'content']]
    # items.to_csv('result_filtered.csv', sep=';', index=False)
    items.to_csv('./Porsak_data/qa_questions-refined.csv', sep=';', index=False)


save_1000word_vector()
# refine_questions()
