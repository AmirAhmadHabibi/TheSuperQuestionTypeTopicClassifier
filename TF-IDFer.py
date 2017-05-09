import pandas as pd


def go():
    questions = pd.read_csv('result.csv', delimiter=';')
    questions = questions['sentence']
    # num_questions_containing - frequency_in_question
    words = pd.DataFrame(columns=('word', 'IDF', 'TF', 'num_q', 'f_in_q'))

    for q_id,qstn in questions.iterrows():
        for word in tokenize(qstn):



go()
