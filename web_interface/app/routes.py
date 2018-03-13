from flask import render_template, request
from app import app

from question_classifier import QuestionClassifier

classifier = QuestionClassifier()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', question='', topics=[], types=[])


@app.route('/submit', methods=['POST'])
def submit_textarea():
    # store the given text in a variable
    text = request.form.get("question_text")

    topics, types = classifier.classify_it(text)
    best_topics = []
    for item in topics.values[:3]:
        best_topics.append(str(item[0]))
    best_types = []
    for item in types.values[:3]:
        best_types.append(str(item[0]))

    return render_template('index.html', question=text, topics=best_topics, types=best_types)
