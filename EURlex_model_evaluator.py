# coding=utf-8
import sys
import numpy as np
from scipy.stats import binom_test
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
# from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from utilitarianism import QuickDataFrame, Progresser
from random import randint
import matplotlib.pyplot as plt

c = ['#E58606', '#99C945', '#52BCA3', '#5D69B1', '#CC61B0', '#24796C', '#DAA51B', '#2F8AC4', '#764E9F', '#ED645A']


def inter_cross_validation(x, y, algs, k=10):
    print('ytrain.shape:', y.shape)
    # create a k fold with no unique classes
    count = 0
    while True:
        count += 1
        kf = list(KFold(n_splits=k, shuffle=True, random_state=randint(0, 100000)).split(x))
        good_folds = True
        for train_index, test_index in kf:
            for i in range(len(y[0])):
                if len(np.unique(y[train_index, i])) < 2:
                    print(i)
                    good_folds = False
                    break
            if not good_folds:
                break
        if good_folds:
            break

    fold_num = 0
    f1scr = {alg: [] for alg in algs.keys()}

    prog = Progresser(k)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for alg_name, alg_cls in algs.items():
            topic_classifier = BinaryRelevance(classifier=alg_cls, require_dense=[True, True])
            try:
                topic_classifier.fit(x_train, y_train)
            except Exception as e:
                print('\nfit error!:', e, alg_name)
                continue
            try:
                predictions = topic_classifier.predict(x_test)
                f1scr[alg_name].append(f1_score(y_test, predictions, average='macro'))
                print('--', alg_name, f1scr[alg_name])
            except Exception as e:
                print('Eval error!:', e)
        fold_num += 1
        prog.count()

    best_alg = ''
    best_score = 0
    for alg_name, score in f1scr.items():
        mean_score = np.mean(score)
        if mean_score > best_score:
            best_alg = alg_name
            best_score = mean_score

    print(best_alg, best_score, '+-', np.std(f1scr[best_alg]))
    return best_alg


def load_data(n_x=1000, n_y=160):
    data = QuickDataFrame.read_csv('./EurLex_data/eurlex_combined_vectors.csv')
    q_vector_length = 1000
    data.delete_column('doc_id')
    x_list = [data[col] for col in data.cols[:n_x]]
    x_array = np.array(x_list, dtype=int).transpose()

    y_list = []
    i = -1
    for col in data.cols[q_vector_length:]:
        i += 1
        if i < n_y:
            y_list.append(data[col])
    y_array = np.array(y_list, dtype=int).transpose()
    print('loaded data.')
    return x_array, y_array


def load_and_split_data():
    x_arr, y_arr = load_data(n_x=500, n_y=50)

    # # split the data into train and test and save the indices to file
    # num_samples = x_arr.shape[0]
    # rand_ind = np.arange(num_samples)
    # np.random.shuffle(rand_ind)
    # train_ind = rand_ind[num_samples // 5:]
    # test_ind = rand_ind[:num_samples // 5]
    # with open('./eurlex_indices.pkl', 'wb') as outfile:
    #     pickle.dump((train_ind, test_ind), outfile)

    # load the indices
    with open('./eurlex_indices.pkl', 'rb') as infile:
        train_ind, test_ind = pickle.load(infile)

    xtrain, xtest = x_arr[train_ind], x_arr[test_ind]
    ytrain, ytest = y_arr[train_ind], y_arr[test_ind]
    return xtrain, xtest, ytrain, ytest


def run_all_models():
    nb_cls = GaussianNB()
    svm_cls = {
        'svm linear': SVC(kernel='linear'),
        'svm poly-2': SVC(kernel='poly', probability=True, degree=2),
        'svm poly-3': SVC(kernel='poly', probability=True, degree=3),
        'svm poly-4': SVC(kernel='poly', probability=True, degree=4),
        'svm poly-5': SVC(kernel='poly', probability=True, degree=5),
        'svm rbf': SVC(kernel='rbf', probability=True)}
    rand_forest_cls = {
        'rand_forest nt5 sqrt': RandomForestClassifier(n_estimators=5, max_features='sqrt'),
        'rand_forest nt5 log2': RandomForestClassifier(n_estimators=5, max_features='log2'),
        'rand_forest nt10 sqrt': RandomForestClassifier(n_estimators=10, max_features='sqrt'),
        'rand_forest nt10 log2': RandomForestClassifier(n_estimators=10, max_features='log2'),
        'rand_forest nt15 sqrt': RandomForestClassifier(n_estimators=15, max_features='sqrt'),
        'rand_forest nt15 log2': RandomForestClassifier(n_estimators=15, max_features='log2'),
        'rand_forest nt20 sqrt': RandomForestClassifier(n_estimators=20, max_features='sqrt'),
        'rand_forest nt20 log2': RandomForestClassifier(n_estimators=20, max_features='log2')}

    xtrain, xtest, ytrain, ytest = load_and_split_data()
    print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

    # best classifiers
    best_cls = {}
    # for naive bayes
    best_cls['naive bayes'] = nb_cls
    # for SVM
    # best_svm = inter_cross_validation(xtrain, ytrain, k=5, algs=svm_cls)
    best_svm = 'svm linear'
    best_cls[best_svm] = svm_cls[best_svm]
    # for Random forest
    # best_rf_name = inter_cross_validation(xtrain, ytrain, k=5, algs=rand_forest_cls)
    best_rf_name = 'rand_forest nt5 sqrt'
    best_cls[best_rf_name] = rand_forest_cls[best_rf_name]

    # test each algorithm
    for alg_name, alg_cls in best_cls.items():
        print('-', alg_name)
        try:
            topic_classifier = BinaryRelevance(classifier=alg_cls, require_dense=[True, True])
            topic_classifier.fit(xtrain, ytrain)
        except Exception as e:
            print('fit error!:', e, alg_name)
            continue
        try:
            predictions = topic_classifier.predict(xtest)
            # Save the predictions to file
            with open('./predictions' + alg_name + '.pkl', 'wb') as outfile:
                pickle.dump(predictions, outfile)
        except Exception as e:
            print('Eval error!:', e)


def eval_models():
    _, _, _, ytest = load_and_split_data()
    models = ['naive bayes', 'svm linear', 'rand_forest nt5 sqrt']
    preds = {}
    for m in models:
        with open('./predictions' + m + '.pkl', 'rb') as infile:
            preds[m] = pickle.load(infile).toarray()

    fig, ax = plt.subplots(figsize=(12, 6))
    m_fscores = []
    m_precisions = []
    m_recall = []
    for mcount, m1 in enumerate(models):
        print('\n', m1)
        fs = f1_score(ytest, preds[m1], average='macro')
        print('fscore', fs)
        m_fscores.append(fs)
        rec = recall_score(ytest, preds[m1], average='macro')
        print('recall', rec)
        m_recall.append(rec)
        pre = precision_score(ytest, preds[m1], average='macro')
        print('precision', pre)
        m_precisions.append(pre)

        xs = [i + 1 for i in range(ytest.shape[1])]
        fscores = []
        recs = []
        pres = []
        for i in range(ytest.shape[1]):
            yt = ytest[:, i]
            # print(yt.shape, preds[m1][:, i].shape)
            fscores.append(f1_score(yt, preds[m1][:, i]))
            recs.append(recall_score(yt, preds[m1][:, i]))
            pres.append(precision_score(yt, preds[m1][:, i]))
        ax.plot(xs, fscores, linewidth=1.5, color=c[mcount], label=m1 + ' fscore', alpha=1)
        # ax.plot(xs, recs, linewidth=1.5, color=c[mcount], label=m1 + ' recall', alpha=1, dashes=[5, 2])
        # ax.plot(xs, pres, linewidth=1.5, color=c[mcount], label=m1 + ' precision', alpha=1, dashes=[1, 1])

        for m2 in models:
            if m2 == m1:
                continue
            # do the binomial test on m1 and m2 for statistical significance
            num_m1_success = 0
            for i in range(ytest.shape[1]):
                yt = ytest[:, i]
                # print(yt.shape, preds[m1][:, i].shape)
                fs_m1 = f1_score(yt, preds[m1][:, i])
                fs_m2 = f1_score(yt, preds[m2][:, i])
                num_m1_success += 1 if fs_m1 > fs_m2 else 0

            p_value = binom_test(x=num_m1_success, n=ytest.shape[1])
            print(m1, 'vs', m2, 'p-value:', p_value)
            if p_value < 0.05:
                print('the null hypothesis is rejected.')
            else:
                print('the null hypothesis is not rejected.')

    ax.legend(loc='lower left', ncol=1, fontsize=11)
    ax.set_xlabel('labels sorted by frequency', fontsize=12)
    ax.set_ylabel('fscore', fontsize=12)
    plt.tight_layout()
    fig.savefig('./EurLex_data/fscore by label.png', bbox_inches='tight')
    fig.savefig('./EurLex_data/fscore by label.pdf', format='pdf', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()

    fig, ax = plt.subplots(figsize=(6, 3))
    opacity = 0.9
    bar_width = 0.30
    ax.set_xlabel('Algorithms')
    # plt.ylabel('')

    ax.set_xticks(range(len(m_precisions)))
    ax.set_xticklabels(['Naive Bayes', 'SVM', 'Random Forest'])
    bar1 = ax.bar(np.arange(len(m_precisions)) + bar_width, m_precisions, bar_width, align='center',
                  alpha=opacity, color=c[0], label='Precision')
    bar2 = ax.bar(range(len(m_recall)), m_recall, bar_width, align='center', alpha=opacity,
                  color=c[1], label='Recall')

    # Add counts above the two bar graphs
    for rect in bar1 + bar2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 3), ha='center', va='bottom')
    plt.ylim([0.0, 1.0])
    ax.legend()
    plt.tight_layout()
    fig.savefig('./EurLex_data/precision recall.png', bbox_inches='tight')
    fig.savefig('./EurLex_data/precision recall.pdf', format='pdf', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()


def bar_plot_it(xs, labels, outfile):
    fig, ax = plt.subplots(figsize=(7, 3))
    opacity = 0.9
    bar_width = 0.30
    ax.set_xlabel('parameters')
    ax.set_ylabel('F1-score')
    # plt.ylabel('')

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(labels)
    bar1 = ax.bar(np.arange(len(xs)), xs, bar_width, align='center',
                  alpha=opacity, color=c[0])

    # Add counts above the two bar graphs
    for rect in bar1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, round(height, 3), ha='center', va='bottom')

    plt.ylim([0.0, 1.0])
    # ax.legend()
    plt.tight_layout()
    fig.savefig('./EurLex_data/' + outfile + '.png', bbox_inches='tight')
    fig.savefig('./EurLex_data/' + outfile + '.pdf', format='pdf', transparent=True, bbox_inches='tight')
    fig.clear()
    plt.clf()


if __name__ == '__main__':
    # run_all_models()
    eval_models()
    # xs = [0.6289155491353986, 0.20832442509601784, 0.11896835786022568,
    #       0.07259824683606117, 0.043068552973203646, 0.42255994991128437]
    # lab = ['linear', 'poly-2', 'poly-3', 'poly-4', 'poly-5', 'rbf']
    # bar_plot_it(xs, lab, 'svms')

    # xs = [0.599642702889794, 0.47525964838921664,0.5130083893632041,0.45951199920331676,0.5353395777332043,0.485409258305178,0.515107951694183,0.4634936641268135]
    # lab = ['nt5 sqrt', 'nt5 log2', 'nt10 sqrt', 'nt10 log2', 'nt15 sqrt', 'nt15 log2','nt20 sqrt','nt20 log2']
    #
    # bar_plot_it(xs, lab, 'rf')

    # naive bayes
    # fscore 0.21346154560917022
    # recall 0.8106127000851562
    # precision 0.14842043693810558
    # naive bayes vs svm linear p-value: 1.7763568394002505e-15
    # the null hypothesis is true.
    # naive bayes vs rand_forest nt5 sqrt p-value: 1.7763568394002505e-15
    # the null hypothesis is true.

    # svm linear
    # fscore 0.6339555541556898
    # recall 0.6376267048512678
    # precision 0.636749882334031
    # svm linear vs naive bayes p-value: 1.7763568394002505e-15
    # the null hypothesis is true.
    # svm linear vs rand_forest nt5 sqrt p-value: 9.021490107130607e-05
    # the null hypothesis is true.

    # rand_forest nt5 sqrt
    # fscore 0.5635191781014112
    # recall 0.4495161313349241
    # precision 0.8424432028120107
    # rand_forest nt5 sqrt vs naive bayes p-value: 1.7763568394002505e-15
    # the null hypothesis is true.
    # rand_forest nt5 sqrt vs svm linear p-value: 9.021490107130607e-05
    # the null hypothesis is true.
