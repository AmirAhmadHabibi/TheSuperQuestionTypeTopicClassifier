# encoding: utf-8
import pandas as pd
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def read_files():
    global data16
    global data17
    global data18

    # id;title;sentence;subject1;subject2;subject3;subject_notsure;suggested_subject;type1;type2;type3;type_notsure;suggested_type
    data16 = pd.read_csv('16-javaheri_e.csv', delimiter=';', index_col=['id'])
    data17 = pd.read_csv('17-sheikholeslami_e.csv', delimiter=';', index_col=['id'])
    data18 = pd.read_csv('18-sayahi_e.csv', delimiter=';', index_col=['id'])

    data17 = data17.append(data16[2567:])
    data17.loc[2568:2799, ('subject1', 'subject2', 'subject3', 'suggested_subject',
                           'type1', 'type2', 'type3', 'suggested_type')] = np.nan
    data17.loc[2568:2799, ('subject_notsure', 'type_notsure')] = 0


def eval_subjects():
    print 'not sure subjects for 16 : ' + str(data16['subject_notsure'].value_counts()[1])
    print 'not sure subjects for 17 : ' + str(data17['subject_notsure'].value_counts()[1])
    print 'not sure subjects for 18 : ' + str(data18['subject_notsure'].value_counts()[1])
    print 'suggested subjects for 16 : ' + str(data16[~data16['suggested_subject'].isnull()].shape[0])
    print 'suggested subjects for 17 : ' + str(data17[~data17['suggested_subject'].isnull()].shape[0])
    print 'suggested subjects for 18 : ' + str(data18[~data18['suggested_subject'].isnull()].shape[0])


def eval_types():
    print 'not sure types for 16 : ' + str(data16['type_notsure'].value_counts()[1])
    print 'not sure types for 17 : ' + str(0)
    print 'not sure types for 18 : ' + str(data18['type_notsure'].value_counts()[1])
    print 'suggested types for 16 : ' + str(data16[~ data16['suggested_type'].isnull()].shape[0])
    print 'suggested types for 17 : ' + str(data17[~ data17['suggested_type'].isnull()].shape[0])
    print 'suggested types for 18 : ' + str(data18[~ data18['suggested_type'].isnull()].shape[0])


def eval_questions():
    # creating a result DataFrame and initiate its columns with nan and 0 values
    result = data16.copy()
    result.loc[:, ('subject1', 'subject2', 'subject3', 'suggested_subject',
                   'type1', 'type2', 'type3', 'suggested_type')] = np.nan
    result.loc[:, ('subject_notsure', 'type_notsure')] = 0

    first_score = 5
    second_score = 4
    third_score = 3

    for row_id, result_row in result.iterrows():  # for each question
        # if row_id<2770: continue
        # if row_id % 50 == 0: print row_id
        d16 = data16.loc[row_id]
        d17 = data17.loc[row_id]
        d18 = data18.loc[row_id]
        scores = dict()

        scores = add_value(d16['subject1'], first_score, scores)
        scores = add_value(d16['subject2'], second_score, scores)
        scores = add_value(d16['subject3'], third_score, scores)

        scores = add_value(d17['subject1'], first_score, scores)
        scores = add_value(d17['subject2'], second_score, scores)
        scores = add_value(d17['subject3'], third_score, scores)

        scores = add_value(d18['subject1'], first_score, scores)
        scores = add_value(d18['subject2'], second_score, scores)
        scores = add_value(d18['subject3'], third_score, scores)

        sorted_scores = sorted(scores, key=scores.get, reverse=True)
        try:
            result_row['subject1'] = [sorted_scores[0], scores[sorted_scores[0]]]
            result_row['subject2'] = [sorted_scores[1], scores[sorted_scores[1]]]
            result_row['subject3'] = [sorted_scores[2], scores[sorted_scores[2]]]
        except Exception, e:
            if str(e) != "list index out of range":
                print e
            pass
        try:
            if result_row['subject1'][1] == result_row['subject2'][1]:
                result_row['subject_notsure'] = 1
        except:
            pass

        result.loc[row_id] = result_row

    result.to_csv("result.csv", sep=';')
    print result[:]


def add_value(name, score_value, scores):
    name = str(name).encode('utf-8')
    if name is not np.nan and str(name) != 'nan':
        if name in scores:
            scores[name] += score_value
        else:
            scores[name] = score_value
    return scores


read_files()
# eval_subjects()
# eval_types()
eval_questions()
