import pickle

import pandas as pd
import warnings

from utilitarianism import QuickDataFrame

warnings.filterwarnings('ignore')

import csv
import json
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

print('packages loaded')


class TextCNN:
    max_features = 9302
    stpwd_thd = 20
    max_size = 200
    word_vector_dim = 100
    do_static = False
    nb_filters = 150
    filter_size_0 = 2
    filter_size_a = 3
    filter_size_b = 4
    drop_rate = 0.3
    batch_size = 64
    nb_epoch = 12
    my_optimizer = 'adam'
    my_patience = 2

    def __init__(self):
        # make sure you replace the root path with your own!
        path_root = 'D:\\Documents\\uni\\The Final Project\\TheSuperQuestionTyper\\Primary_data\\'
        path_to_IMDB = path_root + 'new_data\\'
        path_to_pretrained_wv = 'G:\\'
        path_to_plot = path_root
        self.path_to_save = path_root + 'cnn\\'

        self.name_save = 'my_cnn_two_branches.hdf5'
        print('best model will be saved with name:', self.name_save)

        self.load_data()
        self.load_w2v()
        self.define_cnn()

    def load_data(self):
        questions = pd.read_csv('./Primary_data/questions.txt', delimiter=';', header=None)[0].values
        with open('./Primary_data/word_to_index.csv', 'r', encoding='utf-8') as my_file:
            self.word_to_index = {}
            for line in my_file:
                k, v = line.split(',')
                self.word_to_index[k] = int(v)

        # make index list of the questions
        self.x_all = []
        for q in questions:
            tmp = []
            for wrd in q.split():
                tmp.append(self.word_to_index[wrd])
                self.x_all.append(tmp)

        tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')
        tpc.drop('tpc41', axis=1, inplace=True)
        tpc.drop('tpc42', axis=1, inplace=True)

        self.topics = tpc.values

        print('data loaded')

        ##################################
        self.index_to_word = dict((v, k) for k, v in self.word_to_index.items())
        print(' '.join([self.index_to_word[elt] for elt in self.x_all[4]]))

        ###################################Stopwords and out-of-vocab words removal
        stpwds = [self.index_to_word[idx] for idx in range(1, self.stpwd_thd)]
        print('stopwords are:', stpwds)

        self.x_all = [[elt for elt in rev if elt <= self.max_features and elt >= self.stpwd_thd] for rev in self.x_all]

        print('pruning done')
        ################################## Truncation and zero-padding

        self.x_all = [rev[:self.max_size] for rev in self.x_all]

        print('padding', len([elt for elt in self.x_all if len(elt) < self.max_size])
              , 'reviews from the training set')

        self.x_all = [rev + [0] * (self.max_size - len(rev)) if len(rev) < self.max_size else rev for rev in self.x_all]

        self.x_all = np.array(self.x_all)

        # sanity check: all reviews should now be of size 'max_size'
        assert (self.max_size == list(set([len(rev) for rev in self.x_all]))[0]), "1st sanity check failed!"

        print('truncation and padding done')

    def load_w2v(self):
        # Loading pre-trained word vectors
        with open('./word2vec/Mixed/w2v_per.pkl', 'rb') as infile:
            w2v = pickle.load(infile)
        # create numpy array of embeddings
        self.embeddings = np.zeros((self.max_features + 1, self.word_vector_dim))
        for word in w2v.keys():
            try:
                idx = self.word_to_index[word]
                # word_to_index is 1-based! the 0-th row, used for padding, stays at zero
                self.embeddings[idx,] = np.array(w2v[word])
            except KeyError:
                pass
        print('embeddings created')

    def define_cnn(self):
        # Defining CNN
        my_input = Input(shape=(
            self.max_size,))  # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument

        embedding = Embedding(input_dim=self.embeddings.shape[0],
                              # vocab size, including the 0-th word used for padding
                              output_dim=self.word_vector_dim,
                              weights=[self.embeddings],  # we pass our pre-trained embeddings
                              input_length=self.max_size,
                              trainable=not self.do_static,
                              )(my_input)

        embedding_dropped = Dropout(self.drop_rate)(embedding)

        # feature map size should be equal to max_size-filter_size+1
        # tensor shape after conv layer should be (feature map size, nb_filters)
        print('branch 0:', self.nb_filters, 'feature maps of size', self.max_size - self.filter_size_0 + 1)
        print('branch A:', self.nb_filters, 'feature maps of size', self.max_size - self.filter_size_a + 1)
        print('branch B:', self.nb_filters, 'feature maps of size', self.max_size - self.filter_size_b + 1)

        # 0 branch
        conv_0 = Conv1D(filters=self.nb_filters,
                        kernel_size=self.filter_size_0,
                        activation='relu',
                        )(embedding_dropped)

        pooled_conv_0 = GlobalMaxPooling1D()(conv_0)

        pooled_conv_dropped_0 = Dropout(self.drop_rate)(pooled_conv_0)

        # A branch
        conv_a = Conv1D(filters=self.nb_filters,
                        kernel_size=self.filter_size_a,
                        activation='relu',
                        )(embedding_dropped)

        pooled_conv_a = GlobalMaxPooling1D()(conv_a)

        pooled_conv_dropped_a = Dropout(self.drop_rate)(pooled_conv_a)

        # B branch
        conv_b = Conv1D(filters=self.nb_filters,
                        kernel_size=self.filter_size_b,
                        activation='relu',
                        )(embedding_dropped)

        pooled_conv_b = GlobalMaxPooling1D()(conv_b)

        pooled_conv_dropped_b = Dropout(self.drop_rate)(pooled_conv_b)

        concat = Concatenate()([pooled_conv_dropped_0, pooled_conv_dropped_a, pooled_conv_dropped_b])

        concat_dropped = Dropout(self.drop_rate)(concat)

        # we finally project onto a single unit output layer with sigmoid activation
        prob = Dense(units=1,  # dimensionality of the output space
                     activation='sigmoid',
                     )(concat_dropped)

        self.model = Model(my_input, prob)

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.my_optimizer,
                           metrics=['accuracy'])

        print('model compiled')
        ###############################################
        self.model.summary()
        print(self.model.layers[4].output_shape)  # dimensionality of document encodings (nb_filters*2)

    def train_cnn(self, train_idx, topic):
        # Training CNN
        early_stopping = EarlyStopping(monitor='val_acc',
                                       # go through epochs as long as accuracy on validation set increases
                                       patience=self.my_patience,
                                       mode='max')

        # make sure that the model corresponding to the best epoch is saved
        checkpointer = ModelCheckpoint(filepath=self.path_to_save + self.name_save,
                                       monitor='val_acc',
                                       save_best_only=True,
                                       verbose=0)
        total_len = len(self.x_all[train_idx])
        self.model.fit(self.x_all[train_idx][total_len // 10:],
                       self.topics[train_idx][total_len // 10:, topic],
                       batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       validation_data=(
                           self.x_all[train_idx][:total_len // 10], self.topics[train_idx][:total_len // 10, topic]),
                       callbacks=[early_stopping, checkpointer])

        #####################################
        self.model = load_model(self.path_to_save + self.name_save)

    def predict_text(self, test_idx):
        # Predictive text regions
        # get_region_embedding_a = K.function([self.model.layers[0].input, K.learning_phase()],
        #                                     [self.model.layers[3].output])
        #
        # get_region_embedding_b = K.function([self.model.layers[0].input, K.learning_phase()],
        #                                     [self.model.layers[4].output])
        #
        # get_softmax = K.function([self.model.layers[0].input, K.learning_phase()],
        #                          [self.model.layers[11].output])
        #
        # x_test = self.x_all[test_idx]
        # y_test = self.topics[topic][test_idx]
        #
        # n_doc_per_label = 2
        # idx_pos = [idx for idx, elt in enumerate(y_test) if elt == 1]
        # idx_neg = [idx for idx, elt in enumerate(y_test) if elt == 0]
        # my_idxs = idx_pos[:n_doc_per_label] + idx_neg[:n_doc_per_label]
        #
        # x_test_my_idxs = np.array([x_test[elt] for elt in my_idxs])
        # # y_test_my_idxs = [y_test[elt] for elt in my_idxs]
        # #
        # # reg_emb_a = get_region_embedding_a([x_test_my_idxs, 0])[0]
        # # reg_emb_b = get_region_embedding_b([x_test_my_idxs, 0])[0]
        #
        # # predictions are probabilities of belonging to class 1
        # predictions = get_softmax([x_test_my_idxs, 0])[0]
        # note: you can also use directly: predictions = model.predict(x_test[:100]).tolist()
        predictions = self.model.predict(self.x_all[test_idx]).tolist()
        binary_pred = []
        for el in predictions:
            if el[0] >= 0.5:
                binary_pred.append(1)
            else:
                binary_pred.append(0)
        return binary_pred
        # n_show = 3  # number of most predictive regions we want to display
        #
        # for idx, doc in enumerate(x_test_my_idxs):
        #     tokens = [self.index_to_word[elt] for elt in doc if elt != 0]  # the 0 index is for padding
        #
        #     # extract regions (sliding window over text)
        #     regions_a = extract_regions(tokens, self.filter_size_a)
        #     regions_b = extract_regions(tokens, self.filter_size_b)
        #
        #     print('\n *********')
        #     print('===== text: =====')
        #     print(' '.join(tokens))
        #     print('===== label:', y_test_my_idxs[idx], '=====')
        #     print('===== prediction:', predictions[idx], '=====')
        #     norms_a = np.linalg.norm(reg_emb_a[idx, :, :], axis=1)
        #     norms_b = np.linalg.norm(reg_emb_b[idx, :, :], axis=1)
        #     print('===== most predictive regions of size', self.filter_size_a, ': =====')
        #     print([elt for idxx, elt in enumerate(regions_a) if
        #            idxx in np.argsort(norms_a)[-n_show:]])  # 'np.argsort' sorts by increasing order
        #     print('===== most predictive regions of size', self.filter_size_b, ': =====')
        #     print([elt for idxx, elt in enumerate(regions_b) if idxx in np.argsort(norms_b)[-n_show:]])


def extract_regions(tokens, filter_size):
    regions = []
    regions.append(' '.join(tokens[:filter_size]))
    for i in range(filter_size, len(tokens)):
        regions.append(' '.join(tokens[(i - filter_size + 1):(i + 1)]))
    return regions
