import pandas as pd
import re
import time
from sys import stdout as so
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


class Progresser:
    def __init__(self, total_num):
        self.total = total_num
        self.start_time = time.time()

    def show_progress(self, current_num):
        if current_num % 10 == 0:
            eltime = time.time() - self.start_time
            retime = (self.total - current_num - 1) * eltime / (current_num + 1)

            el_str = str(int(eltime / 3600)) + ':' + str(int((eltime % 3600) / 60)) + ':' + str(int(eltime % 60))
            re_str = str(int(retime / 3600)) + ':' + str(int((retime % 3600) / 60)) + ':' + str(int(retime % 60))

            so.write('\rprocessed %' + str(round(100 * (current_num + 1) / self.total, 2))
                     + '\ttime: ' + el_str + ' | ' + re_str)


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

    data = data.set_index(data['id'])
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

    p = Progresser(data.shape[0])
    cleaner = re.compile('^\s*-*|-\s*$')

    for i, row in data.iterrows():
        # if i <= 50000:
        #     continue
        p.show_progress(i)

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


def find_all_tags():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    tags = dict()
    for i, row in data.iterrows():
        for tag in eval(row['tag']):
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1

    bad_keys = set()
    for tag, f in tags.items():
        if f == 1 or tag == '':
            bad_keys.add(tag)

    for bad in bad_keys:
        del tags[bad]

    sorted_tags = sorted(tags, key=lambda x: tags[x], reverse=True)
    with open('./StackExchange_data/tags.csv', 'w') as outfile:
        outfile.write('term,freq')
        for tag in sorted_tags:
            outfile.write(tag + ',' + str(tags[tag]) + '\n')


def build_word_vectors():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    words_vector = pd.read_csv('./StackExchange_data/1000words.csv', header=None, names={'term'})
    lemmatiser = WordNetLemmatizer()

    data = data.set_index('id')
    # create DataFrame
    cols_list = list(words_vector['term']) + ['question_id']
    train = pd.DataFrame(dtype=object, columns=cols_list)

    cleaner = re.compile('^\s*-*|-\s*$')
    p = Progresser(data.shape[0])
    # build the train data
    for i, qrow in data.iterrows():
        p.show_progress(i)

        train.loc[i, 'question_id'] = i

        # set occurrence values
        tokens_pos = pos_tag(word_tokenize(qrow['body']))
        for word_pos in tokens_pos:
            word = word_pos[0].lower()
            word = re.sub(cleaner, '', word)
            word = lemmatiser.lemmatize(word=word, pos=get_wordnet_pos(word_pos[1]))
            if word in train:
                train.loc[i, word] = 1

        # making backups
        if i % 5000 == 0:
            train.to_csv('./StackExchange_data/data_1000word' + str(i) + '.csv', index=False)

    train = train.fillna(0)

    # rename columns
    number = 0
    for col in train:
        train = train.rename(columns={col: 'wrd' + str(number)})
        number += 1

    train.to_csv('./StackExchange_data/data_1000word.csv', index=False, float_format='%.f')


def build_tag_vectors():
    data = pd.read_csv('./StackExchange_data/all_data.csv')
    topics = pd.read_csv('./StackExchange_data/tags.csv')

    cols_list = list(topics['term']) + ['id']
    data = data.set_index('id')
    train = pd.DataFrame(dtype=object, columns=cols_list)

    p = Progresser(data.shape[0])
    # build the train data
    for i, qrow in data.iterrows():
        p.show_progress(i)
        # if i <= 50000:
        #     continue
        train.loc[i, 'id'] = i
        # set occurrence of the topics
        row_tags = eval(qrow['tag'])
        for tp in row_tags:
            try:
                train.loc[i, tp] = 1
            except Exception as e:
                print(e)
        if i % 2000 == 0:
            train.to_csv('./StackExchange_data/data_tags' + str(int(i / 2000)) + '.csv', index=False)
            train = pd.DataFrame(dtype=object, columns=cols_list)
    print('\n--')

    # concat all data
    train_all = pd.DataFrame(dtype=object, columns=list(topics['term']))
    for i in range(0, 26):
        try:
            train_temp = pd.read_csv('./StackExchange_data/data_tags' + str(i) + '.csv')
            train_temp = train_temp[cols_list]
            train_temp = train_temp.fillna(0)
            train_all = pd.concat([train_all, train_temp], axis=0)
            print(i, 'done!')
        except Exception as e:
            print(e)

    train = train.fillna(0)
    train = train[cols_list]
    train_all = pd.concat([train_all, train], axis=0)
    print('The last one done!')

    train_all = train_all[cols_list]

    train_all.to_csv('./StackExchange_data/data_tags.csv', index=False, float_format='%.f', columns=cols_list)


# combine_all()
# find_frequent_words()
# find_all_tags()
build_word_vectors()
# build_tag_vectors()
