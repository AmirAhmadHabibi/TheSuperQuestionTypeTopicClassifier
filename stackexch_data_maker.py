import pandas as pd
import re
import time
from sys import stdout as so
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


def reformat(text):
    text = text.replace(' lt ', '<')
    text = text.replace(' gt ', '>')
    text = text.replace(' quot ', '')
    text = text.replace('\n', ' ')
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
    data = pd.DataFrame(columns=['id', 'body', 'title', 'tag'])
    for i in range(1, 6):
        fold = pd.read_excel('./StackExchange_data/Fold_' + str(i) + '.xlsx')
        data = data.append(fold)

    id_num = 0
    for i, row in data.iterrows():
        row['id'] = id_num
        id_num += 1

    data.set_index('id')
    bad_rows = set()
    html_cleaner = re.compile('<.*?>')
    for i, row in data.iterrows():
        try:
            row['body'] = reformat(row['body'])
            row['tag'] = reformat(row['tag'])

            row['body'] = re.sub(html_cleaner, '', row['body'])
            row['tag'] = get_tags(row['tag'])
        except Exception as e:
            print(e)
            print('--', i, row)
            bad_rows.add(row['id'])

    data = data.drop(data.index[list(bad_rows)])

    # reset the id numbers
    id_num = 0
    for i, row in data.iterrows():
        row['id'] = id_num
        id_num += 1

    data.to_csv('./StackExchange_data/all_data.csv', index=False, columns=['id', 'body', 'title', 'tag'])


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

    # with open('./StackExchange_data/words50000.csv', 'r') as infile:
    #     for line in infile:
    #         w, _, f = line.partition(',')
    #         words[w] = int(f)

    start_time = time.time()
    total_num = data.shape[0]
    cleaner = re.compile('^\s*-*|-\s*$')

    for i, row in data.iterrows():
        # if i <= 50000:
        #     continue
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

        if i % 5000 == 0:
            with open('./StackExchange_data/words' + str(i) + '.csv', 'w') as outfile:
                for word, freq in words.items():
                    outfile.write(word + ',' + str(freq) + '\n')

    sorted_words = sorted(words, key=lambda x: words[x], reverse=True)

    with open('./StackExchange_data/words_frequency.csv', 'w') as outfile:
        for word in sorted_words:
            try:
                outfile.write(str(word) + ',' + str(words[word]) + '\n')
            except:
                pass

    with open('./StackExchange_data/1000words.csv', 'w') as outfile:
        for word in sorted_words[:1000]:
            outfile.write(str(word) + '\n')


def build_word_vectors():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    lemmatiser = WordNetLemmatizer()
    words_vector = pd.read_csv('./StackExchange_data/1000words.csv', header=None, names={'term'})

    # create dataframe
    train = pd.DataFrame(dtype=object)
    for wrd in words_vector['term'].as_matrix():
        train[wrd] = 0

    cleaner = re.compile('^\s*-*|-\s*$')
    total_num = data.shape[0]
    start_time = time.time()
    # build the train data
    for i, qrow in data.iterrows():
        elapsed_time = time.time() - start_time
        retime = (total_num - i - 1) * elapsed_time / (i + 1)
        so.write('\rprocessed %' + str(round(100 * (i + 1) / total_num, 2))
                 + '\ttime: ' + str(int(elapsed_time)) + ' | ' + str(int(retime)))

        train.loc[i, words_vector['term'][0]] = 0

        # set occurrence values
        tokens_pos = pos_tag(word_tokenize(qrow['body']))
        for word_pos in tokens_pos:
            word = word_pos[0].lower()
            word = re.sub(cleaner, '', word)
            word = lemmatiser.lemmatize(word=word, pos=get_wordnet_pos(word_pos[1]))
            if word in train:
                train.loc[i, word] = 1

    print(train.shape)
    train = train.fillna(0)

    # rename columns
    number = 0
    for col in train:
        train[col] = train.apply(lambda row: int(row[col]), axis='columns')
        train = train.rename(columns={col: 'wrd' + str(number)})
        number += 1
    print(train.shape)
    train.to_csv('./StackExchange_data/data_1000word.csv', index=False)


def find_all_tags():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    tags = set()
    for i, row in data.iterrows():
        try:
            tags = tags | eval(row['tag'])
        except Exception as e:
            print(e)
            print('--', row['id'], row['tag'])

    with open('./StackExchange_data/tags.csv', 'w') as outfile:
        for tag in tags:
            outfile.write(tag + '\n')


# combine_all()
# find_frequent_words()
# find_all_tags()
build_word_vectors()
