from flask import render_template, request
from app import app

from question_classifier import QuestionClassifier

classifier = QuestionClassifier()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', question='', topics=[], types=[], topics2=[], types2=[])


@app.route('/submit', methods=['POST'])
def submit_textarea():
    # store the given text in a variable
    text = request.form.get("question_text")
    topics = []
    types = []
    topics2 = []
    types2 = []

    if text != '':
        topics_df, types_df = classifier.bow_classify(text)
        for item in topics_df.values[:]:
            topics.append((str(item[0]), round(item[1], 2), round((item[1] + 0.2) / 1.2, 2)))
        for item in types_df.values[:]:
            types.append((str(item[0]), round(item[1], 2), round((item[1] + 0.2) / 1.2, 2)))

        topics2_df, types2_df = classifier.w2v_classify(text)
        for item in topics2_df.values[:]:
            topics2.append((str(item[0]), round(item[1], 2), round((item[1] + 0.2) / 1.2, 2)))
        for item in types2_df.values[:]:
            types2.append((str(item[0]), round(item[1], 2), round((item[1] + 0.2) / 1.2, 2)))

    return render_template('index.html', question=text, topics=topics, types=types, topics2=topics2, types2=types2)
