# coding=utf-8
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, label_ranking_loss
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import pickle
from utilitarianism import QuickDataFrame


def evaluate_model(x, y, learn_path):
    kf = list(KFold(n_splits=4).split(x))
    with open(learn_path + 'topic_classifier-folds.pkl', 'wb') as out_file:
        pickle.dump(kf, out_file)
    fold_num = 0

    for train_index, test_index in kf:
        print(train_index, test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        topic_classifier = BinaryRelevance(classifier=SVC(probability=True, verbose=True, shrinking=False),
                                           require_dense=[True, True])
        # topic_classifier = BinaryRelevance(classifier=GaussianNB(), require_dense=[True, True])
        # topic_classifier = BinaryRelevance(classifier=RandomForestClassifier(max_depth=5, random_state=0),
        #                                    require_dense=[True, True])

        topic_classifier.fit(x_train, y_train)

        with open(learn_path + 'topic_classifier-NB' + str(fold_num) + '.pkl', 'wb') as out_file:
            pickle.dump(topic_classifier, out_file)

        try:
            predictions = topic_classifier.predict(x_test)

            print('Accuracy (normalised):', accuracy_score(y_test, predictions, normalize=True))
            print('Accuracy:', accuracy_score(y_test, predictions, normalize=False))
            print('F1_score (micro averaged):', f1_score(y_test, predictions, average='micro'))
            print('F1_score (macro averaged by labels):', f1_score(y_test, predictions, average='macro'))
            print('F1_score (averaged by samples):', f1_score(y_test, predictions, average='samples'))
            print('Hamming loss:', hamming_loss(y_test, predictions))
            print('Label Ranking loss:', label_ranking_loss(y_test, predictions.toarray()))
        except Exception as e:
            print(e)

        fold_num += 1


data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors.csv')
data.delete_column('doc_id')
x_list = []
for col in data.cols[:1000]:
    x_list.append(data[col])
x_array = np.array(x_list).transpose()
print(x_array)
y_list = []
for col in data.cols[1000:]:
    y_list.append(data[col])
y_array = np.array(y_list).transpose()
print('read')

evaluate_model(x_array, y_array, './EurLex_data/models/')
