# coding=utf-8
import pandas as pd


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    qstn = qstn.replace('ي', 'ی')
    qstn = qstn.replace('ى', 'ی')
    qstn = qstn.replace('ك', 'ک')
    qstn = qstn.replace(' ', ' ')
    qstn = qstn.replace('‌', ' ')
    qstn = qstn.replace('‏', ' ')
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
    # remove the stop words
    for stop_word in stop_words.as_matrix():
        words = words[words.term != stop_word[0]]
    # sort by frequency number
    words = words.sort_values(by='num', ascending=False)
    words = words.reset_index(drop=True)
    # save first 1000 to file
    words = words[:1000]
    words = words[['term']]
    words.to_csv('words_vector.csv', sep=';', index=False)


save_vector()

# words = pd.read_csv('result.csv', delimiter=';')
# for i, row in words.iterrows():
#     term = words.loc[i, 'sentence']
#     term = term.replace('ي', 'ی')
#     term = term.replace('ى', 'ی')
#     term = term.replace('ك', 'ک')
#     term = term.replace(' ', ' ')
#     term = term.replace('‌', ' ')
#     term = term.replace('‏', ' ')
#     words.loc[i, 'sentence'] = term
# words.to_csv('result.csv', sep=';', index=False)
