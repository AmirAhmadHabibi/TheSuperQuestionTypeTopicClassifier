# coding=utf-8
import sys
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


def evaluate_model(x, y, learn_path, k=10):
    kf = list(KFold(n_splits=k, shuffle=True).split(x))
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
            print(e)
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
            print(e)

        fold_num += 1
        prog.count()

    for col in stats.cols:
        print(col, np.mean(stats[col]))


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

y_list = []
for col in data.cols[1000:]:
    y_list.append(data[col])
y_array = np.array(y_list, dtype=int).transpose()

evaluate_model(x_array, y_array, './EurLex_data/models/', k=3)

# cls = GaussianNB()
# Jaccard (normalised) 0.090209213705
# Accuracy (normalised) 0.000103039670273
# Accuracy 0.2
# F1_score (micro averaged) 0.111080218615
# F1_score (macro averaged by labels) 0.101469370537
# F1_score (averaged by samples) 0.154538916184
# Hamming loss 0.141384671323
# Label Ranking loss: 0.29793993981

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
