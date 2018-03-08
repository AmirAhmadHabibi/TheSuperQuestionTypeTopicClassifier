import pandas as pd
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from skmultilearn.ext import Meka
import os

home = os.environ['HOME']
print(home)

# wrd = pd.read_csv('/mnt/d/Documents/uni/The Final Project/TheSuperQuestionTyper/Primary_data/1000word_vector_Q.csv')
# tpc = pd.read_csv('/mnt/d/Documents/uni/The Final Project/TheSuperQuestionTyper/Primary_data/topic_vector_Q.csv')
#
# tpc.drop('tpc41', axis=1, inplace=True)
# tpc.drop('tpc42', axis=1, inplace=True)
# X = wrd.values
# y = tpc.values
X, y = make_multilabel_classification(sparse=True, return_indicator='sparse')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

meka = Meka(
    meka_classifier="meka.classifiers.multilabel.LC",
    weka_classifier="weka.classifiers.bayes.NaiveBayes",
    meka_classpath="/opt/meka-1.9/lib/",
    java_command='/usr/bin/java')
meka.fit(X_train, y_train)


print(X_test[15])
try:
    predictions = meka.predict(X_test)
    hamming_loss(y_test, predictions)
except:
    print(meka.error)


