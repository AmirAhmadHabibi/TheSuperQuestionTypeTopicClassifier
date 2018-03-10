from flask import render_template, request
from app import app


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', question='', words=[])


@app.route('/submit', methods=['POST'])
def submit_textarea():
    # store the given text in a variable
    text = request.form.get("question_text")

    text2 = text.split(' ')

    return render_template('index.html', question=text, words=text2)
