import os
import pickle

import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from utilitarianism import Progresser, QuickDataFrame


def process_html_files():
    id_mappings = pd.read_csv('./EurLex_data/eurlex_ID_mappings.csv', sep='\t')

    no_file_num = 0
    no_en_num = 0
    no_texte_num = 0
    prog = Progresser(id_mappings.shape[0])

    for i, row in id_mappings.iterrows():
        prog.count()
        try:
            # if file already processed then continue
            if os.path.isfile('./EurLex_data/eurlex_txt/' + str(row['DocID']) + '.txt'):
                continue

            # read the html
            with open('./EurLex_data/eurlex_html_EN_NOT/' + row['Filename'].replace(':', '_'), 'r',
                      encoding="utf8") as infile:
                html = infile.read()

            # extract raw text
            soup = BeautifulSoup(html, 'html.parser')
            elem = soup.findAll('div', {'class': 'texte'})
            if len(elem) == 0:
                no_texte_num += 1
                continue
            raw_text = elem[0].text.strip()
            if raw_text.startswith('/* There is no English') or raw_text == '':
                no_en_num += 1
                continue

            # write the text into a new file
            with open('./EurLex_data/eurlex_txt/' + str(row['DocID']) + '.txt', 'w', encoding="utf8") as outfile:
                outfile.write(raw_text)

        except IOError:
            no_file_num += 1
        except Exception as e:
            print(e)

    print('NO texte:', no_texte_num)
    print('NO EN:', no_en_num)
    print('NO file:', no_file_num)


def get_wordnet_pos(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatise_all():
    id_mappings = pd.read_csv('./EurLex_data/eurlex_ID_mappings.csv', sep='\t')

    lemmatiser = WordNetLemmatizer()
    stop_words = set()
    for w in stopwords.words('english'):
        stop_words.add(w)
    cleaner = re.compile('^\s*-*|-\s*$')

    prog = Progresser(id_mappings.shape[0])

    for i, row in id_mappings.iterrows():
        prog.count()
        try:
            # if file already processed then continue
            if os.path.isfile('./EurLex_data/lem_txt/' + str(row['DocID']) + '-lem.txt'):
                continue

            try:
                with open('./EurLex_data/eurlex_txt/' + str(row['DocID']) + '.txt', 'r', encoding="utf8") as infile:
                    raw_text = infile.read()
            except:
                continue

            lemmatised_doc = ''

            # lemmatise each sentence
            for sent in sent_tokenize(raw_text):
                lemmatised_sent = ''
                tokens_pos = pos_tag(word_tokenize(sent))

                # lemmatise each word in sentence
                for word_pos in tokens_pos:
                    if len(word_pos[0]) < 2: continue

                    word = word_pos[0].lower()
                    word = re.sub(cleaner, '', word)
                    if word in stop_words: continue

                    if len(word) > 2:
                        word = lemmatiser.lemmatize(word=word, pos=get_wordnet_pos(word_pos[1]))
                        if word in stop_words: continue

                    lemmatised_sent += word + ' '
                lemmatised_doc += lemmatised_sent + '\n'
            # write doc to file
            with open('./EurLex_data/lem_txt/' + str(row['DocID']) + '-lem.txt', 'w', encoding="utf8") as outfile:
                outfile.write(lemmatised_doc)
        except Exception as e:
            print(e)


def find_frequent_words():
    id_mappings = QuickDataFrame.read_csv('./EurLex_data/eurlex_ID_mappings.csv', sep='\t')
    words = dict()

    prog = Progresser(len(id_mappings))
    for i in range(len(id_mappings)):
        prog.count()
        try:
            # read the file
            try:
                with open('./EurLex_data/lem_txt/' + str(id_mappings['DocID'][i]) + '-lem.txt', 'r',
                          encoding="utf8") as infile:
                    doc_text = infile.read()
            except IOError as e:
                # print(e)
                continue
            # count the words
            for word in word_tokenize(doc_text):
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        except Exception as e:
            print(e)

    # remove bad words
    cleaner = re.compile('^(19\d\d)$|^(2\d\d\d)$|^((?!\d)\w)*$')
    filtered_words = dict()
    for word, freq in words.items():
        if cleaner.match(word):
            filtered_words[word] = freq

    sorted_words = sorted(filtered_words, key=lambda x: filtered_words[x], reverse=True)

    with open('./EurLex_data/words_frequency.csv', 'w', encoding="utf8") as outfile:
        for word in sorted_words:
            try:
                outfile.write(str(word) + ',' + str(words[word]) + '\n')
            except Exception as e:
                print(word, e)
                pass

    with open('./EurLex_data/1000words.csv', 'w', encoding="utf8") as outfile:
        for word in sorted_words[:1000]:
            outfile.write(str(word) + '\n')


def find_all_tags():
    subject_data = QuickDataFrame.read_csv('./EurLex_data/eurlex_id2class/id2class_eurlex_subject_matter.qrels',
                                           header=False, columns=['sub', 'doc_id', 'col2'], sep=' ')
    subjects = dict()
    for i in range(len(subject_data)):
        sub = subject_data['sub'][i]
        if sub not in subjects:
            subjects[sub] = 1
        else:
            subjects[sub] += 1

    sorted_tags = sorted(subjects, key=lambda x: subjects[x], reverse=True)
    with open('./EurLex_data/tags.csv', 'w') as outfile:
        outfile.write('term,freq\n')
        for tag in sorted_tags:
            outfile.write(tag + ',' + str(subjects[tag]) + '\n')


def build_w2v_vectors():
    with open('./word2vec/word2vec-En.pkl', 'rb') as infile:
        w2v = pickle.load(infile)

    w2v_length = 300
    stop_words = set()
    for w in stopwords.words('english'):
        stop_words.add(w)

    id_mappings = QuickDataFrame.read_csv('./EurLex_data/eurlex_ID_mappings.csv', sep='\t')

    # create DataFrame
    cols_list = ['doc_id'] + ['w' + str(i) for i in range(0, w2v_length)]
    train = QuickDataFrame(columns=cols_list)

    prog = Progresser(len(id_mappings))
    for i in range(len(id_mappings)):
        prog.count()
        # read the file
        try:
            with open('./EurLex_data/lem_txt/' + str(id_mappings['DocID'][i]) + '-lem.txt', 'r',
                      encoding="utf8") as infile:
                doc_text = infile.read()
        except IOError:
            continue
        try:
            sum_array = np.zeros(w2v_length)
            number_of_words = 0

            for word in word_tokenize(doc_text):
                if word not in stop_words and word in w2v:
                    number_of_words += 1
                    sum_array += w2v[word]
            if number_of_words > 0:
                sum_array = sum_array / number_of_words

            train.append([id_mappings['DocID'][i]] + list(sum_array))

        except Exception as e:
            print(e)

    train.to_csv('./EurLex_data/w2v_vector_Q.csv')


def build_all_vectors():
    id_mappings = QuickDataFrame.read_csv('./EurLex_data/eurlex_ID_mappings.csv', sep='\t')
    subject_data = QuickDataFrame.read_csv('./EurLex_data/eurlex_id2class/id2class_eurlex_subject_matter.qrels',
                                           header=False, columns=['sub', 'doc_id', 'col2'], sep=' ')
    words_vector = QuickDataFrame.read_csv('./EurLex_data/1000words.csv', header=False, columns=['term'])
    topics = QuickDataFrame.read_csv('./EurLex_data/tags.csv')

    # train = QuickDataFrame.read_csv('./EurLex_data/w2v_vector_Q.csv')
    # train.set_index(train['doc_id'], unique=True)

    # create DataFrame
    cols_list = ['doc_id'] + list(words_vector['term'])
    train = QuickDataFrame(columns=cols_list)

    # filling word columns
    prog = Progresser(len(id_mappings))
    for i in range(len(id_mappings)):
        prog.count()
        try:
            # read the file
            try:
                with open('./EurLex_data/lem_txt/' + str(id_mappings['DocID'][i]) + '-lem.txt', 'r',
                          encoding="utf8") as infile:
                    doc_text = infile.read()
            except IOError:
                continue

            # add a new row
            train.append(value=0)

            # complete the data in that row
            train['doc_id'][len(train) - 1] = id_mappings['DocID'][i]
            for word in word_tokenize(doc_text):
                if word in train.data:
                    train[word][len(train) - 1] = 1
        except Exception as e:
            print(e)

    # index by doc id
    train.set_index(train['doc_id'], unique=True)

    # rename word columns
    rename_dict = dict()
    index = 0
    for wrd in list(words_vector['term']):
        rename_dict[wrd] = 'wrd' + str(index)
        index += 1
    train.rename(columns=rename_dict)

    # add topic columns
    for col in list(topics['term']):
        train.add_column(name=col, value=0)

    # filling topic columns
    for i in range(len(subject_data)):
        try:
            sub = subject_data['sub'][i]
            doc_id = subject_data['doc_id'][i]
            train[sub, doc_id] = 1
        except Exception as e:
            print(e)

    # rename topic columns
    rename_dict = dict()
    index = 0
    for tpc in list(topics['term']):
        rename_dict[tpc] = 'tpc' + str(index)
        index += 1
    train.rename(columns=rename_dict)

    # write to file
    print('\nWriting to file...')
    # train.to_csv('./EurLex_data/eurlex_combined_vectors.csv')
    train.to_csv('./EurLex_data/eurlex_combined_vectors-w2v.csv')


# process_html_files()
# lemmatise_all()
# find_frequent_words()
# find_all_tags()
# build_w2v_vectors()
build_all_vectors()
