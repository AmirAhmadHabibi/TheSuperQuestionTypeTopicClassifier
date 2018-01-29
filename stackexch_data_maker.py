import pandas as pd
import re
import time
from sys import stdout as so
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def reformat(text):
    text = text.replace(' lt ', '<')
    text = text.replace(' gt ', '>')
    text = text.replace(' quot ', '')
    text = text.replace(' xA ', '')
    text = text.replace(' amp ', ' and ')
    text = text.replace('--', '–')
    return text


def get_tags(txt):
    if txt[0] != '<' or txt[-1] != '>':
        raise Exception('Bad tag! ' + txt)
    tags = txt[1:-1].split('><')
    tag_set = set()
    cleaner = re.compile('^\s*-*|-\s*$|\s*')
    cleaner2 = re.compile('-+')
    for tag in tags:
        tag = re.sub(cleaner, '', tag)
        tag = re.sub(cleaner2, '-', tag)
        tag_set.add(tag)
    return tag_set


def combine_all():
    data = pd.DataFrame(columns={'body', 'title', 'tag'})
    for i in range(1, 6):
        fold = pd.read_excel('./StackExchange_data/Fold_' + str(i) + '.xlsx')
        data = data.append(fold)

    html_cleaner = re.compile('<.*?>')
    for i, row in data.iterrows():
        try:
            row['body'] = reformat(row['body'])
            row['tag'] = reformat(row['tag'])

            row['body'] = re.sub(html_cleaner, '', row['body'])
            row['tag'] = get_tags(row['tag'])
        except Exception as e:
            print(e)
            print(row)
            data.drop(data.index[i])

    data.to_csv('./StackExchange_data/all_data.csv', index=False)


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


def find_frequent_words():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    words = dict()
    lemmatiser = WordNetLemmatizer()
    stop_words = set()
    for w in stopwords.words('english'):
        stop_words.add(w)

    start_time = time.time()
    total_num = data.shape[0]
    cleaner = re.compile('^\s*-*|-\s*$')

    for i, row in data.iterrows():

        elapsed_time = time.time() - start_time
        retime = (total_num - i - 1) * elapsed_time / (i + 1)
        so.write('\rprocessed %' + str(round(100 * (i + 1) / total_num, 2))
                 + '\ttime: ' + str(int(elapsed_time)) + ' | ' + str(int(retime)))

        tokens_pos = pos_tag(word_tokenize(row['body']))
        for word_pos in tokens_pos:
            if len(word_pos[0]) < 2:
                continue

            word = word_pos[0].lower()
            word = re.sub(cleaner, '', word)

            if word in stop_words:
                continue

            if len(word) > 2:
                word = lemmatiser.lemmatize(word=word, pos=get_wordnet_pos(word_pos[1]))

            if word not in stop_words:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        # print(' ')
        if i % 5000 == 0:
            with open('./StackExchange_data/words' + str(i) + '.csv', 'w') as outfile:
                for word, freq in words.items():
                    outfile.write(word + ',' + str(freq) + '\n')

    sorted_words = sorted(words, key=lambda x: words[x], reverse=True)

    with open('./StackExchange_data/words_frequency.csv', 'w') as outfile:
        for word in sorted_words:
            outfile.write(word + ',' + words[word] + '\n')

    with open('./StackExchange_data/1000words.csv', 'w') as outfile:
        for word in sorted_words[:1000]:
            outfile.write(word + '\n')


# combine_all()
find_frequent_words()
