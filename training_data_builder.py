# coding=utf-8
import numpy as np
import pandas as pd
import sys
import pickle

from utilitarianism import Progresser, QuickDataFrame


def tokenise(qstn):
    for ch in [':', ';', ',', '?', '!', '\'', '\"', '\\', '/', '(', ')', '[', ']', '…', '...', '–', '-', '<', '>', '؟',
               '،', '.', '­', '«', '»', '_', '+', '=', ]:
        qstn = qstn.replace(ch, ' ')
    # TODO: use a better method
    return qstn.split()


def concat_word2vec_subjs():
    w2v = pd.read_csv('questions-word2vec.txt', header=None)
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    subjs = pd.read_csv('./combining_tags_data/subjs-result.csv', delimiter=';')
    trn = pd.DataFrame()
    for i, subj in subjs.iterrows():
        trn['sub' + str(subj['id'])] = 0
    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, subj in subjs.iterrows():
            sub_id = str(subj['id'])
            trn.loc[i, 'sub' + sub_id] = 0
            if qrow['subject1'] == subj['tag'] or qrow['subject2'] == subj['tag'] or qrow['subject3'] == subj['tag']:
                trn.loc[i, 'sub' + sub_id] = 1
                # print i, '----',trn.loc[i]

    result = pd.concat([trn, w2v], axis=1)
    result.to_csv('subj-word2vec.csv', index=False, header=None)


def concat_word2vec_types():
    questions = pd.read_csv('./Primary_data/result_filtered.csv', delimiter=';')
    w2v = pd.read_csv('./Primary_data/questions-word2vec.txt', header=None)
    types = pd.read_csv('./combining_tags_data/types-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, typ in types.iterrows():
        train['typ' + str(typ['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, typ in types.iterrows():
            typ_id = str(typ['id'])
            train.loc[i, 'typ' + typ_id] = 0
            if qrow['type1'] == typ['tag'] or qrow['type2'] == typ['tag'] or qrow['type3'] == typ['tag']:
                train.loc[i, 'typ' + typ_id] = 1

    result = pd.concat([train, w2v], axis=1)
    result.to_csv('type-word2vec.csv', index=False)


def create_1000word_vector():
    questions = pd.read_csv('./Primary_data/result_filtered.csv', delimiter=';')
    # questions_2 = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')
    words_vector = pd.read_csv('./Primary_data/words_vector.csv')
    # first_rows = 2799

    # create dataframe
    train = pd.DataFrame(dtype=object)
    for wrd in words_vector['term'].as_matrix():
        train[wrd] = 0
    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        train.loc[i, words_vector['term'][0]] = 0
        # set occurrence values
        for word in tokenise(qrow['sentence']):
            if word in train:
                train.loc[i, word] = 1

    # for i, qrow in questions_2.iterrows():
    #     if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i + first_rows))
    #     train.loc[i + first_rows, words_vector['term'][0]] = 0
    #     # set occurrence values
    #     for word in tokenise(str(qrow['content']) + ' ' + qrow['title']):
    #         if word in train:
    #             train.loc[i + first_rows, word] = 1

    print(train.shape)
    train = train.fillna(0)

    # rename columns
    number = 0
    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')
        train = train.rename(columns={col: 'wrd' + str(number)})
        number += 1
    print(train.shape)
    train.to_csv('1000word_vector_Q.csv', index=False)


def create_clean_questions():
    questions = pd.read_csv('./Primary_data/result_filtered.csv', delimiter=';')
    # stop_words = set(pd.read_csv('./Primary_data/PersianStopWordList.txt', header=None)[0])
    with open('./Primary_data/questions.txt', 'w', encoding='utf-8') as outf:
        for i, qrow in questions.iterrows():
            outf.write(' '.join(tokenise(qrow['sentence'])) + '\n')


def create_type_vector():
    questions = pd.read_csv('./Primary_data/result_filtered.csv', delimiter=';')
    types = pd.read_csv('./combining_tags_data/types-result.csv', delimiter=';')

    # creating dataframe
    train = pd.DataFrame(dtype=object)
    for i, typ in types.iterrows():
        train['typ' + str(typ['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        # set occurrence of the subjects
        for j, typ in types.iterrows():
            typ_id = str(typ['id'])
            train.loc[i, 'typ' + typ_id] = 0
            if qrow['type1'] == typ['tag'] or qrow['type2'] == typ['tag'] or qrow['type3'] == typ['tag']:
                train.loc[i, 'typ' + typ_id] = 1

    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')

    train.to_csv('./Primary_data/type_vector_Q.csv', index=False)


def create_topic_vector():
    questions = pd.read_csv('result_filtered.csv', delimiter=';')
    # questions_2 = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')
    topics = pd.read_csv('./Porsak_data/topic_list.csv')
    # first_rows = 2799

    # creating dataframe
    topic_id = dict()
    train = pd.DataFrame(dtype=object)
    for i, tpc in topics.iterrows():
        topic_id[tpc['topic']] = 'tpc' + str(tpc['id'])
        train['tpc' + str(tpc['id'])] = 0

    # build the train data
    for i, qrow in questions.iterrows():
        if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))

        # set occurrence of the topics
        for j, tpc in topics.iterrows():
            train.loc[i, 'tpc' + str(tpc['id'])] = 0

        try:
            train.loc[i, topic_id[qrow['subject1']]] = 1
            train.loc[i, topic_id[qrow['subject2']]] = 1
            train.loc[i, topic_id[qrow['subject3']]] = 1
        except Exception as e:
            if str(e) != 'nan':
                print(e)

    # build the train data from second list of questions
    # for i, qrow in questions_2.iterrows():
    #     if i % 100 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
    #     # set occurrence of the topics
    #     for j, tpc in topics.iterrows():
    #         col_name = 'tpc' + str(tpc['id'])
    #         train.loc[i + first_rows, col_name] = 0
    #         if qrow['topic'] == tpc['topic']:
    #             train.loc[i + first_rows, col_name] = 1

    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')

    train.to_csv('topic_vector_Q.csv', index=False)


def concat_1000vec_type_cat():
    wrd = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
    subj = pd.read_csv('./Primary_data/topic_vector_Q.csv')
    # type = pd.read_csv('type_vector_Q.csv')
    # result = pd.concat([type, subj, wrd], axis=1)
    # result = pd.concat([type, wrd], axis=1)
    result = pd.concat([subj, wrd], axis=1)
    result.to_csv('./Primary_data/arff/13_tpc;wrd.arff', index=False)


def concat_all():
    bow = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
    w2v = pd.read_csv('./Primary_data/w2v-100_vector_Q.csv')
    typ = pd.read_csv('./Primary_data/type_vector_Q.csv')
    tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')

    result = pd.concat([typ, bow], axis=1)
    result.to_csv('./Primary_data/arff/1_typ;bow.arff', index=False)

    result = pd.concat([typ, w2v], axis=1)
    result.to_csv('./Primary_data/arff/2_typ;w2v.arff', index=False)

    result = pd.concat([typ, tpc, bow], axis=1)
    result.to_csv('./Primary_data/arff/3_typ;tpc,bow.arff', index=False)

    result = pd.concat([tpc, bow], axis=1)
    result.to_csv('./Primary_data/arff/4_tpc;bow.arff', index=False)

    result = pd.concat([tpc, w2v], axis=1)
    result.to_csv('./Primary_data/arff/5_tpc;w2v.arff', index=False)


def create_arff_header():
    header = '@relation \'CQA\'\n\n'
    # for i in range(0, 12):
    #     header += '@attribute typ' + str(i) + ' {0,1}\n'
    for i in range(0, 26):
        header += '@attribute tpc' + str(i) + ' {0,1}\n'
    for i in range(0, 100):
        header += '@attribute wrd' + str(i) + ' numeric\n'
    # for i in range(0, 1000):
    #     header += '@attribute wrd' + str(i) + ' {0,1}\n'
    header += '\n@data\n'
    print(header)


def save_topic_list():
    topics = dict()
    questions_2 = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')
    for i, qrow in questions_2.iterrows():
        topics[qrow['topicid']] = qrow['topic']
    df = pd.DataFrame(topics.items(), columns={'id', 'topic'})
    df.to_csv('./Porsak_data/topic_list.csv', columns={'topic', 'id'}, index=False)


def read_w2v_data():
    w2v = dict()
    # with open('./word2vec/IRBlog/blog.fa.text.300.vec', 'r', encoding='utf-8') as infile:
    with open('./word2vec/Mixed/twitt_wiki_ham_blog.fa.text.100.vec', 'r', encoding='utf-8') as infile:
        first_line = True
        for line in infile:
            if first_line:
                first_line = False
                continue
            tokens = line.split()
            w2v[tokens[0]] = [float(el) for el in tokens[1:]]
            if len(w2v[tokens[0]]) != 300:  # 100:
                print('Bad line!')

    # with open('./word2vec/IRBlog/w2v_per_300.pkl', 'wb') as outfile:
    with open('./word2vec/Mixed/w2v_per.pkl', 'wb') as outfile:
        pickle.dump(w2v, outfile)


def create_w2v_vectors():
    # with open('./word2vec/IRBlog/w2v_per_300.pkl', 'rb') as infile:
    with open('./word2vec/Mixed/w2v_per.pkl', 'rb') as infile:
        w2v = pickle.load(infile)
    w2v_length = 100  # 300
    stop_words = set(pd.read_csv('./Primary_data/PersianStopWordList.txt', header=None)[0])
    questions = pd.read_csv('./Primary_data/result_filtered.csv', delimiter=';')

    train = QuickDataFrame(['w' + str(i) for i in range(0, w2v_length)])

    prog = Progresser(questions.shape[0])
    # build the train data
    for i, qrow in questions.iterrows():
        prog.count()
        sum_array = np.zeros(w2v_length)
        number_of_words = 0

        for word in tokenise(qrow['sentence']):
            if word not in stop_words and word in w2v:
                number_of_words += 1
                sum_array += w2v[word]
        if i != len(train):
            print('wat?!!')
        train.append(list(sum_array / number_of_words))

    train.to_csv('./Primary_data/w2v-100_vector_Q.csv')
    # train.to_csv('./Primary_data/w2v-300_vector_Q.csv')


def create_sentence_files():
    stop_words = set(pd.read_csv('./Primary_data/PersianStopWordList.txt', header=None)[0])
    questions = QuickDataFrame.read_csv('./Primary_data/result_filtered.csv', sep=';')
    topics = QuickDataFrame.read_csv('./Primary_data/topic_vector_Q.csv')

    files = dict()
    for tpc in topics.cols:
        files[tpc + '-p'] = open('./Primary_data/sent_topic/' + tpc + '.p', 'w', encoding='utf-8')
        files[tpc + '-n'] = open('./Primary_data/sent_topic/' + tpc + '.n', 'w', encoding='utf-8')

    prog = Progresser(len(questions['sentence']))
    # build the train data
    for i, qrow in enumerate(questions['sentence']):
        prog.count()
        snt = []
        for word in tokenise(qrow):
            if word not in stop_words:
                snt.append(word)
        snt = ' '.join(snt)
        for tpc in topics.cols:
            if topics[tpc][i] == '0':
                files[tpc + '-n'].write(snt + '\n')
            elif topics[tpc][i] == '1':
                files[tpc + '-p'].write(snt + '\n')
            else:
                print("wattt")

    for fl in files.values():
        fl.close()


def create_word_to_index():
    questions = pd.read_csv('./Primary_data/questions.txt', delimiter=';', header=None)[0].values
    w2i = {}
    for q in questions:
        for word in q.split():
            try:
                w2i[word] += 1
            except:
                w2i[word] = 1
    w2i_sorted = sorted(w2i.items(), key=lambda tup: tup[1],reverse=True)
    with open('./Primary_data/word_to_index.csv', 'w', encoding='utf-8') as outf:
        for i, w in enumerate(w2i_sorted):
            outf.write(w[0] + ',' + str(i+1) + '\n')


# create_arff_header()
# concat_word2vec_subjs()
# concat_word2vec_types()

# create_1000word_vector()
# create_type_vector()
# create_topic_vector()
# concat_1000vec_type_cat()
# concat_all()

# save_topic_list()

# read_w2v_data()
# create_w2v_vectors()

# create_sentence_files()
# create_clean_questions()
create_word_to_index()
