# The Super Question Type-Topic Classifier
There has not been much work done on this topic in the Persian language so the resulting tool will be quite useful in Persian Q&A websites. Machine Learning techniques are used here to determine the type and the category of questions so they can be more easily tagged and classified. Also identifying the question type can be very helpful in further NLP tasks such as summarization.
## Data
The dataset used for our experiments is a set of 2800 Persian questions randomly selected by crawling 140 different social question-and-answer forums or FAQ pages. To define the annotation scheme for the question topic classification, we used the most frequent tags of questions in the main international CQA sites. For the annotation scheme of question types, we integrated the available models mentioned in Table 3 to achieve a more general scheme for this goal. In total, 23 different topics and 12 types were defined for our task.
For both question topics and types, the data were annotated by three annotators who are graduate students and native speakers of Persian. 
For each question, the annotators can select up to 3 category labels, while the order of labels should also be taken into the account; i.e., the first label has a higher priority compared to the second one. If none of the available labels are appropriate, they can suggest a new label for the question. The interface also provides a check box for the uncertainty of the annotators. They should fill it if they are not sure about their selected label(s).
<br><b>combinator.py</b> contains the code for combining the tags of these annotators and evaluating some of the statistics of their tags. A further analysis of the statistics is done in <b>analyser_pro.py</b>.
#### This dataset will be available soon.
## Training data
We use bag of words as the input for our learning methods. In <b>word_vector_builder.py</b> we find the most frequent words in the questions of our dataset excluding the stop words. Then in <b>training_data_builder.py</b> we create the feature vector and the vector of types and topics for each question. 
## Learning a model and prediction
In <b>fast_learner.py</b> we first use SVM to learn a model from the training data and then we use that model for the prediction of type and topic for the neew questions. 

## Web API
The web_interface directory contains the web app based on Flask and the API would include the file <b>question_classifier.py</b>. It's use would be like what follows:
```python
from question_classifier import QuestionClassifier

# initialising the class would load the pre-trained files
classifier = QuestionClassifier()

# then for each question you can use the BoW or W2V model
topics_df, types_df = classifier.bow_classify(input_question)
topics_df, types_df = classifier.w2v_classify(input_question)

# these two are pandas DataFrames
# they are lists of all tags along with the likelihood of their assignment to the input question
for item in topics_df.values:
    print('tag:', item[0])
    print('likelihood:', item[1])
```

A more detailed description of the project will be added soon...
