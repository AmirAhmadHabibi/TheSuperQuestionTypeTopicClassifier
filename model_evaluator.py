# coding=utf-8
import sys
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, label_ranking_loss, jaccard_similarity_score
from sklearn.model_selection import KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import pickle
from utilitarianism import QuickDataFrame, Progresser
from question_classifier import QuestionClassifier
from random import randint


def evaluate_model(x, y, learn_path, k=10):
    print(len(y), len(y[0]))
    # create a k fold with no unique classes
    count = 0
    while True:
        count += 1
        print(count, 'Finding a proper KF...')
        kf = list(KFold(n_splits=k, shuffle=True, random_state=randint(0, 100000)).split(x))
        good_folds = True
        for train_index, test_index in kf:
            for i in range(len(y[0])):
                if len(np.unique(y[train_index, i])) < 2:
                    # print(y[train_index, i],np.unique(y[train_index, i]))
                    print(i)
                    good_folds = False
                    break
            if not good_folds:
                break
        if good_folds:
            break
    print('Found a good KF!')

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

        cls = SVC(kernel='linear')
        # cls = GaussianNB()
        # cls = RandomForestClassifier(max_features='auto', random_state=1)

        topic_classifier = BinaryRelevance(classifier=cls, require_dense=[True, True])

        try:
            topic_classifier.fit(x_train, y_train)
        except Exception as e:
            print('fit error!:', e)
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


def eval_porsak_questions():
    questions = pd.read_csv('./Porsak_data/qa_questions-refined.csv', delimiter=';')

    q_classifier = QuestionClassifier()
    total = 0
    TP = 0
    TP_5 = 0
    for i, qrow in questions.iterrows():
        if i % 10 == 0: sys.stdout.write('\r' + 'processed question ' + str(i))
        try:
            predictions, _ = q_classifier.classify_it(qrow['content'])

            total += 1
            if qrow['topic'] == predictions['topic'][0]:
                TP += 1
            for p in predictions['topic']:
                if qrow['topic'] == p:
                    TP_5 += 1
                    break
        except Exception as e:
            print('\n', e)
            print('--', qrow['content'], qrow)

    print('res: ', TP, total, TP / total)
    print('res3: ', TP_5, total, TP_5 / total)


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
data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors.csv')
data.delete_column('doc_id')
x_list = []
for col in data.cols[:1000]:
    x_list.append(data[col])
x_array = np.array(x_list, dtype=int).transpose()

# for col in data.cols[1000:]:
#     print(col, np.unique(data[col]))

bad_cols = {156, 161, 162, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
            183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200}
y_list = []
i = -1
for col in data.cols[1000:]:
    i += 1
    if i < 160:
        y_list.append(data[col])
    else:
        count = [0, 0]
        for el in data[col]:
            count[int(el)] += 1
        print(i, count)
y_array = np.array(y_list, dtype=int).transpose()

evaluate_model(x_array, y_array, './EurLex_data/models/', k=5)

# cls = GaussianNB()
# Jaccard (normalised) 0.090209213705
# Accuracy (normalised) 0.000103039670273
# Accuracy 0.2
# F1_score (micro averaged) 0.111080218615
# F1_score (macro averaged by labels) 0.101469370537
# F1_score (averaged by samples) 0.154538916184
# Hamming loss 0.141384671323
# Label Ranking loss: 0.29793993981

# cls = SVC(kernel='linear')
# Jaccard (normalised) 0.610522034486
# Accuracy (normalised) 0.367433944685
# Accuracy 1426.6
# F1_score (micro averaged) 0.697793617218
# F1_score (macro averaged by labels) 0.533867906627
# F1_score (averaged by samples) 0.684405104741
# Hamming loss 0.00834492297433
# Label Ranking loss: 0.275148839555

###################################################################################
# wrd = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
# tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')
#
# tpc.drop('tpc41', axis=1, inplace=True)
# tpc.drop('tpc42', axis=1, inplace=True)
#
# evaluate_model(wrd.values, tpc.values, './Primary_data/models/')

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
