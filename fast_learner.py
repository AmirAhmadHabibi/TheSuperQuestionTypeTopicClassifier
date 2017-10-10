import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

wrd = pd.read_csv('1000word_vector_Q.csv')
subj = pd.read_csv('subject_vector_Q.csv')
type = pd.read_csv('type_vector_Q.csv')

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

classifier = BinaryRelevance(classifier=SVC(), require_dense=[False, True])

# train
classifier.fit(wrd, subj)

# predict
predictions = classifier.predict(wrd)
print predictions
