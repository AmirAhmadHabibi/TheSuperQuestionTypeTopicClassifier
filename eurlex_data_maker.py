import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from utilitarianism import Progresser


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

            with open('./EurLex_data/eurlex_txt/' + str(row['DocID']) + '.txt', 'r', encoding="utf8") as infile:
                raw_text = infile.read()

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


process_html_files()
lemmatise_all()
