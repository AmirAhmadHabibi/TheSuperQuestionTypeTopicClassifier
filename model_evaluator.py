# coding=utf-8
import sys
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, label_ranking_loss, jaccard_similarity_score, \
    jaccard_score
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import pickle

from cnn_imdb.cnn_imdb import TextCNN
from utilitarianism import QuickDataFrame, Progresser
from question_classifier import QuestionClassifier
from random import randint


def evaluate_model_cnn(x, y, learn_path, k=10):
    print(len(y), len(y[0]))
    # create a k fold with no unique classes
    count = 0
    while True:
        count += 1
        # print(count, 'Finding a proper KF...')
        kf = list(KFold(n_splits=k, shuffle=True, random_state=randint(0, 100000)).split(x))
        good_folds = True
        for train_index, test_index in kf:
            for i in range(len(y[0])):
                if len(np.unique(y[train_index, i])) < 2:  # or len(np.unique(y[test_index, i])) < 2:
                    # print(y[train_index, i],np.unique(y[train_index, i]))
                    print(i)
                    good_folds = False
                    break
            if not good_folds:
                break
        if good_folds:
            break
    print('Found a good KF in', count, 'try!')

    with open(learn_path + 'topic_classifier-folds.pkl', 'wb') as out_file:
        pickle.dump(kf, out_file)
    fold_num = 0

    stats = QuickDataFrame(['Jaccard (normalised)', 'Accuracy (normalised)', 'Accuracy', 'F1_score (micro averaged)',
                            'F1_score (macro averaged by labels)', 'F1_score (averaged by samples)', 'Hamming loss',
                            'Label Ranking loss:'])

    txt_cnns = [TextCNN() for _ in range(y.shape[1])]
    prog = Progresser(k)
    for train_index, test_index in kf:
        # print(train_index, test_index)
        print('___________________________________________________')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # cls = SVC(kernel='linear')
        # cls = SVC(kernel='poly', probability=True, tol=1e-5)
        # cls = GaussianNB()
        # cls = RandomForestClassifier(max_features='auto', random_state=1)

        # topic_classifier = BinaryRelevance(classifier=cls, require_dense=[True, True])

        # try:
        # topic_classifier.fit(x_train, y_train)
        for i in range(y.shape[1]):
            txt_cnns[i].train_cnn(train_index, i)
        # except Exception as e:
        #     print('\nfit error!:', e)
        #     continue

        # with open(learn_path + 'topic_classifier-SVC' + str(fold_num) + '.pkl', 'wb') as out_file:
        #     pickle.dump(topic_classifier, out_file)

        # try:
        predictions = np.zeros((len(x_test), y.shape[1]))
        for i in range(y.shape[1]):
            predictions[:, i] = np.array(txt_cnns[i].predict_text(test_index))

        s = [jaccard_score(y_test, predictions, average='micro'),
             accuracy_score(y_test, predictions, normalize=True),
             accuracy_score(y_test, predictions, normalize=False),
             f1_score(y_test, predictions, average='micro'),
             f1_score(y_test, predictions, average='macro'),
             f1_score(y_test, predictions, average='samples'),
             hamming_loss(y_test, predictions),
             label_ranking_loss(y_test, predictions)]

        stats.append(s)
        print(stats[stats.length - 1])
        # except Exception as e:
        #     print('Eval error!:', e)

        fold_num += 1
        prog.count()

    for col in stats.cols:
        print(col, np.mean(stats[col]))


def evaluate_model(x, y, learn_path, k=10):
    print(len(y), len(y[0]))
    # create a k fold with no unique classes
    count = 0
    while True:
        count += 1
        # print(count, 'Finding a proper KF...')
        kf = list(KFold(n_splits=k, shuffle=True, random_state=randint(0, 100000)).split(x))
        good_folds = True
        for train_index, test_index in kf:
            for i in range(len(y[0])):
                if len(np.unique(y[train_index, i])) < 2:  # or len(np.unique(y[test_index, i])) < 2:
                    # print(y[train_index, i],np.unique(y[train_index, i]))
                    print(i)
                    good_folds = False
                    break
            if not good_folds:
                break
        if good_folds:
            break
    print('Found a good KF in', count, 'try!')

    with open(learn_path + 'topic_classifier-folds.pkl', 'wb') as out_file:
        pickle.dump(kf, out_file)
    fold_num = 0

    stats = QuickDataFrame(['Jaccard (normalised)', 'Accuracy (normalised)', 'Accuracy', 'F1_score (micro averaged)',
                            'F1_score (macro averaged by labels)', 'F1_score (averaged by samples)', 'Hamming loss',
                            'Label Ranking loss:'])

    prog = Progresser(k)
    for train_index, test_index in kf:
        # print(train_index, test_index)
        print('___________________________________________________')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # cls = SVC(kernel='linear')
        cls = SVC(kernel='poly', probability=True, tol=1e-5)
        # cls = GaussianNB()
        # cls = RandomForestClassifier(max_features='auto', random_state=1)

        topic_classifier = BinaryRelevance(classifier=cls, require_dense=[True, True])

        try:
            topic_classifier.fit(x_train, y_train)
        except Exception as e:
            print('\nfit error!:', e)
            continue

        with open(learn_path + 'topic_classifier-SVC' + str(fold_num) + '.pkl', 'wb') as out_file:
            pickle.dump(topic_classifier, out_file)

        try:
            predictions = topic_classifier.predict(x_test)
            s = [jaccard_similarity_score(y_test, predictions, normalize=True),
                 accuracy_score(y_test, predictions, normalize=True),
                 accuracy_score(y_test, predictions, normalize=False),
                 f1_score(y_test, predictions, average='micro'),
                 f1_score(y_test, predictions, average='macro'),
                 f1_score(y_test, predictions, average='samples'),
                 hamming_loss(y_test, predictions),
                 label_ranking_loss(y_test, predictions.toarray())]

            stats.append(s)
            print(stats[stats.length - 1])
        except Exception as e:
            print('Eval error!:', e)

        fold_num += 1
        prog.count()

    for col in stats.cols:
        print(col, np.mean(stats[col]))


def evaluate_model_svm(x, y, learn_path, k=10, thresh=0.5):
    print(len(y), len(y[0]))
    # create a k fold with no unique classes
    count = 0
    while True:
        count += 1
        # print(count, 'Finding a proper KF...')
        kf = list(KFold(n_splits=k, shuffle=True, random_state=randint(0, 100000)).split(x))
        good_folds = True
        for train_index, test_index in kf:
            for i in range(len(y[0])):
                if len(np.unique(y[train_index, i])) < 2:  # or len(np.unique(y[test_index, i])) < 2:
                    # print(y[train_index, i],np.unique(y[train_index, i]))
                    print(i)
                    good_folds = False
                    break
            if not good_folds:
                break
        if good_folds:
            break
    print('Found a good KF in', count, 'try!')

    with open(learn_path + 'topic_classifier-folds.pkl', 'wb') as out_file:
        pickle.dump(kf, out_file)
    fold_num = 0

    stats = QuickDataFrame(['Jaccard (normalised)', 'Accuracy (normalised)', 'Accuracy', 'F1_score (micro averaged)',
                            'F1_score (macro averaged by labels)', 'F1_score (averaged by samples)', 'Hamming loss',
                            'Label Ranking loss:'])

    prog = Progresser(k)
    for train_index, test_index in kf:
        # print(train_index, test_index)
        print('___________________________________________________')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # cls = SVC(kernel='linear')
        # cls = SVC(kernel='poly', probability=True, tol=1e-5)
        cls = SVC(kernel='linear', probability=True, tol=1e-5)
        # cls = GaussianNB()
        # cls = RandomForestClassifier(max_features='auto', random_state=1)

        topic_classifier = BinaryRelevance(classifier=cls, require_dense=[True, True])

        try:
            topic_classifier.fit(x_train, y_train)
        except Exception as e:
            print('\nfit error!:', e)
            continue

        # with open(learn_path + 'topic_classifier-SVC' + str(fold_num) + '.pkl', 'wb') as out_file:
        #     pickle.dump(topic_classifier, out_file)

        try:
            # predictions = topic_classifier.predict(x_test)
            predictions = np.zeros((len(x_test), y.shape[1]))
            preds = topic_classifier.predict_proba(x_test)
            for i in range(len(x_test)):
                for j in range(y.shape[1]):
                    predictions[i, j] = 1.0 if preds[i, j] > thresh else 0.0
            s = [jaccard_similarity_score(y_test, predictions, normalize=True),
                 accuracy_score(y_test, predictions, normalize=True),
                 accuracy_score(y_test, predictions, normalize=False),
                 f1_score(y_test, predictions, average='micro'),
                 f1_score(y_test, predictions, average='macro'),
                 f1_score(y_test, predictions, average='samples'),
                 hamming_loss(y_test, predictions),
                 label_ranking_loss(y_test, predictions)]

            stats.append(s)
            print(stats[stats.length - 1])
        except Exception as e:
            print('Eval error!:', e)

        fold_num += 1
        prog.count()

    for col in stats.cols:
        print(col, np.mean(stats[col]))


def eval_porsak_questions():
    questions = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')

    q_classifier = QuestionClassifier()
    total = 0
    TP = 0
    TP_5 = 0
    for i, qrow in questions.iterrows():
        if i % 10 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        try:
            predictions, _ = q_classifier.bow_classify(qrow['content'])

            total += 1
            if qrow['topic'] == predictions['topic'][0]:
                TP += 1
            for p in predictions['topic'][:5]:
                if qrow['topic'] == p:
                    TP_5 += 1
                    break
        except Exception as e:
            print('\n', e)
            print('--', qrow['content'], qrow)

    print('res: ', TP, total, TP / total)
    print('res5: ', TP_5, total, TP_5 / total)


# wrd = pd.read_csv('./StackExchange_data/data_1000word.csv')
# words_vector = pd.read_csv('./StackExchange_data/1000words.csv', header=None, names={'term'})
# wrd_cols_list = list(words_vector['term'])
# wrd = wrd[wrd_cols_list].values
#
# tpc = pd.read_csv('./StackExchange_data/data_tags.csv')
# topics = pd.read_csv('./StackExchange_data/tags.csv')
# tpc_cols_list = list(topics['term'])
# tpc = tpc[tpc_cols_list].values

###################################################################################
# data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors.csv')
# q_vector_length=1000
# data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors-w2v.csv')
# q_vector_length = 300
# data.delete_column('doc_id')
# x_list = []
# for col in data.cols[:q_vector_length]:
#     x_list.append(data[col])
# x_array = np.array(x_list, dtype=float).transpose()
#
# # for col in data.cols[1000:]:
# #     print(col, np.unique(data[col]))
#
# y_list = []
# i = -1
# for col in data.cols[q_vector_length:]:
#     i += 1
#     if i < 160:
#         y_list.append(data[col])
#     else:
#         count = [0, 0]
#         for el in data[col]:
#             count[int(el)] += 1
#         print(i, count)
# y_array = np.array(y_list, dtype=int).transpose()
#
# evaluate_model(x_array, y_array, './EurLex_data/models/', k=5)

## BoW ###############################
# cls = GaussianNB()
# Jaccard (normalised) 0.090209213705
# Accuracy (normalised) 0.000103039670273
# Accuracy 0.2
# F1_score (micro averaged) 0.111080218615
# F1_score (macro averaged by labels) 0.101469370537
# F1_score (averaged by samples) 0.154538916184
# Hamming loss 0.141384671323
# Label Ranking loss: 0.29793993981

# cls = SVC(kernel='linear' )
# Jaccard (normalised) 0.610522034486
# Accuracy (normalised) 0.367433944685
# Accuracy 1426.6
# F1_score (micro averaged) 0.697793617218
# F1_score (macro averaged by labels) 0.533867906627
# F1_score (averaged by samples) 0.684405104741
# Hamming loss 0.00834492297433
# Label Ranking loss: 0.275148839555

## W2V ##############################
# cls = SVC(kernel='linear')
# Jaccard (normalised) 0.405679062661
# Accuracy (normalised) 0.235152250201
# Accuracy 913.0
# F1_score (micro averaged) 0.548598195022
# F1_score (macro averaged by labels) 0.140844211483
# F1_score (averaged by samples) 0.45700493993
# Hamming loss 0.00886647721883
# Label Ranking loss: 0.582609931228

# cls = GaussianNB()
# Jaccard (normalised) 0.178940562149
# Accuracy (normalised) 0.0136506334233
# Accuracy 53.0
# F1_score (micro averaged) 0.209756754355
# F1_score (macro averaged by labels) 0.177835398016
# F1_score (averaged by samples) 0.274044646778
# Hamming loss 0.0823452730684
# Label Ranking loss: 0.250578788312

###################################################################################
###################################################################################
wrd = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
# wrd = pd.read_csv('./Primary_data/w2v-100_vector_Q.csv')
typ = pd.read_csv('./Primary_data/type_vector_Q.csv')
tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')

tpc.drop('tpc41', axis=1, inplace=True)
tpc.drop('tpc42', axis=1, inplace=True)
typ.drop('typ9', axis=1, inplace=True)
typ.drop('typ10', axis=1, inplace=True)

evaluate_model_cnn(wrd.values, typ.values, './Primary_data/models/', k=10)
# evaluate_model(pd.concat([wrd, tpc], axis=1).values, typ.values, './Primary_data/models/', k=10)
# evaluate_model_svm(wrd.values, typ.values, './Primary_data/models/', k=10, thresh=0.5)
# evaluate_model_svm(wrd.values, typ.values, './Primary_data/models/', k=10, thresh=0.8)

# # # w2v100+tpc > typ #######################################
# cls = SVC(kernel='poly', probability=True, tol=1e-5)
# Jaccard (normalised) 0.502408580816
# Accuracy (normalised) 0.463733998976
# Accuracy 129.8
# F1_score (micro averaged) 0.584980279926
# F1_score (macro averaged by labels) 0.329533855336
# F1_score (averaged by samples) 0.437980798771
# Hamming loss 0.0672378392217
# Label Ranking loss: 0.438358717815

# # # w2v100 > typ ###########################################
# cls = SVC(kernel='poly', probability=True, tol=1e-5)
# Jaccard (normalised) 0.495660086038
# Accuracy (normalised) 0.45301431127
# Accuracy 253.6
# F1_score (micro averaged) 0.577756793334
# F1_score (macro averaged by labels) 0.327427258353
# F1_score (averaged by samples) 0.443642175654
# Hamming loss 0.0707045744953
# Label Ranking loss: 0.43237197678

# # # w2v100 > tpc ##########################################
# cls = SVC(kernel='poly', probability=True, tol=1e-5)
# Jaccard (normalised) 0.570614652671
# Accuracy (normalised) 0.448014592934
# Accuracy 125.4
# F1_score (micro averaged) 0.683831596709
# F1_score (macro averaged by labels) 0.559220635898
# F1_score (averaged by samples) 0.611545058884
# Hamming loss 0.0312766150367
# Label Ranking loss: 0.375597457338

# cls = SVC(kernel='poly')
# Jaccard (normalised) 0.565494111623
# Accuracy (normalised) 0.442652329749
# Accuracy 123.9
# F1_score (micro averaged) 0.682355932347
# F1_score (macro averaged by labels) 0.555164508284
# F1_score (averaged by samples) 0.606294290835
# Hamming loss 0.0312167712067
# Label Ranking loss: 0.383444962055

# cls = SVC(kernel='linear')
# Jaccard (normalised) 0.554875277351
# Accuracy (normalised) 0.409434203789
# Accuracy 114.6
# F1_score (micro averaged) 0.653495126747
# F1_score (macro averaged by labels) 0.524043698375
# F1_score (averaged by samples) 0.604658419282
# Hamming loss 0.0371566713603
# Label Ranking loss: 0.363341210126

# # # w2v300 ##############################################
# cls = SVC(kernel='linear')
# Jaccard (normalised) 0.546905582408
# Accuracy (normalised) 0.363340494092
# Accuracy 145.285714286
# F1_score (micro averaged) 0.625354823243
# F1_score (macro averaged by labels) 0.555707019553
# F1_score (averaged by samples) 0.612444212146
# Hamming loss 0.0446738870987
# Label Ranking loss: 0.325811153074

# # # 1000 words: ##########################################
# cls = SVC(kernel='linear')
# Jaccard (normalised) 0.521736601809
# Accuracy (normalised) 0.390135688684
# Accuracy 109.2
# F1_score (micro averaged) 0.639605743586
# F1_score (macro averaged by labels) 0.529568110956
# F1_score (averaged by samples) 0.566632317802
# Hamming loss 0.0363832351937
# Label Ranking loss: 0.416022735404

# cls = RandomForestClassifier(max_features='auto', random_state=1)
# Jaccard (normalised) 0.462395481311
# Accuracy (normalised) 0.350126728111
# Accuracy 98.0
# F1_score (micro averaged) 0.592559836075
# F1_score (macro averaged by labels) 0.451692876028
# F1_score (averaged by samples) 0.500898361495
# Hamming loss 0.0382869837003
# Label Ranking loss: 0.490010359556

# cls = GaussianNB()
# Jaccard (normalised) 0.398375226268
# Accuracy (normalised) 0.237950588838
# Accuracy 66.6
# F1_score (micro averaged) 0.408633605008
# F1_score (macro averaged by labels) 0.306246968965
# F1_score (averaged by samples) 0.460985270043
# Hamming loss 0.0979040258577
# Label Ranking loss: 0.444275451023

# cls = SVC thr=0.5
# Jaccard (normalised) 0.13079901951820153
# Accuracy (normalised) 0.0
# Accuracy 0.0
# F1_score (micro averaged) 0.22961488444606423
# F1_score (macro averaged by labels) 0.14745198708203616
# F1_score (averaged by samples) 0.22367873933542132
# Hamming loss 0.6497510240655402
# Label Ranking loss: 0.6359530349624916
###################################################################
# cnn >0
# Jaccard (micro) 0.0930624213300329
# Accuracy (normalised) 0.03321428571428571
# Accuracy 9.3
# F1_score (micro averaged) 0.17012156952745797
# F1_score (macro averaged by labels) 0.08761245250289004
# F1_score (averaged by samples) 0.17496908906931946
# Hamming loss 0.3894194828469022
# Label Ranking loss: 0.6534011662460684

# cnn > 0.5 - epoch 6 - stopwrd 1
# Jaccard (micro) 0.03653130943112978
# Accuracy (normalised) 0.10040066564260113
# Accuracy 28.1
# F1_score (micro averaged) 0.07011494654950781
# F1_score (macro averaged by labels) 0.0436767597892361
# F1_score (averaged by samples) 0.046768561187916016
# Hamming loss 0.1486947004608295
# Label Ranking loss: 0.8318718188440251

# cnn > 0.5 - epoch 12 - stopwrd 1
# Jaccard (micro) 0.05702004719300282
# Accuracy (normalised) 0.12289938556067588
# Accuracy 34.4
# F1_score (micro averaged) 0.10750092784864887
# F1_score (macro averaged by labels) 0.05832761228486455
# F1_score (averaged by samples) 0.07141854412015701
# Hamming loss 0.14269815668202765
# Label Ranking loss: 0.8061138846219491

# cnn > 0.5 - epoch 20 - stopwrd 20
# Jaccard (micro) 0.05291172108117589
# Accuracy (normalised) 0.12645929339477724
# Accuracy 35.4
# F1_score (micro averaged) 0.10034009616668475
# F1_score (macro averaged by labels) 0.05944546774737468
# F1_score (averaged by samples) 0.07055854241338114
# Hamming loss 0.15013556067588324
# Label Ranking loss: 0.8083093673550662

# cnn > 0.5 - epoch 12 - stopwrd 1 - filter 2 3 4
# Jaccard (normalised) 0.05818289551661602
# Accuracy (normalised) 0.1171825396825397
# Accuracy 32.8
# F1_score (micro averaged) 0.10980055980814525
# F1_score (macro averaged by labels) 0.0585355076276816
# F1_score (averaged by samples) 0.07395908004778971
# Hamming loss 0.1488489503328213
# Label Ranking loss: 0.8023237309410025


# check if this CNN is for multi label
#
# Send all data again