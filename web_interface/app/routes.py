from flask import render_template, request
from app import app

from question_classifier import QuestionClassifier
from data_handler import DataHandler

classifier = QuestionClassifier()
data_handler = DataHandler()


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


@app.route('/submit_tags', methods=['POST'])
def submit_tags():
    question = ''
    types = []
    topics = []
    for key, value in request.form.items():
        if key == "question_text":
            question = value
        elif key.startswith('typ-'):
            if int(value) > 0:
                types.append(key[4:])
        elif key.startswith('tpc-'):
            if int(value) > 0:
                topics.append(key[4:])

    data_handler.add_question(question, topics, types)

    return render_template('index.html', question='', topics=[], types=[], topics2=[], types2=[])
