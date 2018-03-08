# coding=utf-8
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, label_ranking_loss, jaccard_similarity_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import LinearSVC, SVC
from SVM import SVM
import pickle
from utilitarianism import QuickDataFrame


def evaluate_model(x, y, learn_path):
    kf = list(KFold(n_splits=10).split(x))
    with open(learn_path + 'topic_classifier-folds.pkl', 'wb') as out_file:
        pickle.dump(kf, out_file)
    fold_num = 0

    stats = QuickDataFrame(['Jaccard (normalised)', 'Accuracy (normalised)', 'Accuracy', 'F1_score (micro averaged)',
                            'F1_score (macro averaged by labels)', 'F1_score (averaged by samples)', 'Hamming loss',
                            'Label Ranking loss:'])

    for train_index, test_index in kf:
        # print(train_index, test_index)
        print('___________________________________________________')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        topic_classifier = BinaryRelevance(
                            classifier=SVC(kernel='linear'),
                            require_dense=[True, True])
        # topic_classifier = BinaryRelevance(classifier=GaussianNB(), require_dense=[True, True])
        # topic_classifier = BinaryRelevance(classifier=RandomForestClassifier(max_depth=5, random_state=0),
        #                                    require_dense=[True, True])

        try:
            topic_classifier.fit(x_train, y_train)
        except Exception as e:
            print(e)
            continue

        with open(learn_path + 'topic_classifier-NB' + str(fold_num) + '.pkl', 'wb') as out_file:
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

    for col in stats.cols:
        print(col, np.mean(stats[col]))


# data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors.csv')
# data.delete_column('doc_id')
# x_list = []
# for col in data.cols[:1000]:
#     x_list.append(data[col])
# x_array = np.array(x_list, dtype=int).transpose()
#
# y_list = []
# for col in data.cols[1000:]:
#     y_list.append(data[col])
# y_array = np.array(y_list, dtype=int).transpose()
#
# evaluate_model(x_array, y_array, './EurLex_data/models/')

###################################################################################
wrd = pd.read_csv('./Primary_data/1000word_vector_Q.csv')
tpc = pd.read_csv('./Primary_data/topic_vector_Q.csv')

tpc.drop('tpc41', axis=1, inplace=True)
tpc.drop('tpc42', axis=1, inplace=True)

evaluate_model(wrd.values, tpc.values, './Primary_data/models/')

# Jaccard (normalised) 0.35541005291
# Accuracy (normalised) 0.229365079365
# Accuracy 64.2222222222
# F1_score (micro averaged) 0.478719511505
# F1_score (macro averaged by labels) 0.259154851321
# F1_score (averaged by samples) 0.399880952381
# Hamming loss 0.050496031746
# Label Ranking loss: 0.584855966248
