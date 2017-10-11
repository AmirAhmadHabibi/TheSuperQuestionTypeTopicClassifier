import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

wrd = pd.read_csv('1000word_vector_Q.csv')
sub = pd.read_csv('subject_vector_Q.csv')
typ = pd.read_csv('type_vector_Q.csv')

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

classifier = BinaryRelevance(classifier=SVC(probability=True), require_dense=[False, True])

# train
classifier.fit(wrd[10:], sub[10:])

print classifier.partition

# predict
print classifier.predict_proba(wrd[:1])
