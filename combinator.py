# encoding: utf-8
import pandas as pd
import numpy as np
import sys

FIRST_SCORE = 10
SECOND_SCORE = 9
THIRD_SCORE = 8


def read_files():
    global data16
    global data17
    global data18

    # id;title;sentence;subject1;subject2;subject3;subject_notsure;suggested_subject;type1;type2;type3;type_notsure;suggested_type
    data16 = pd.read_csv('./1_combine_tags/16-javaheri_e.csv', delimiter=';', index_col=['id'])
    data17 = pd.read_csv('./1_combine_tags/17-sheikholeslami_e.csv', delimiter=';', index_col=['id'])
    data18 = pd.read_csv('./1_combine_tags/18-sayahi_e.csv', delimiter=';', index_col=['id'])

    data16['sentence'] = data16.apply(lambda row: row['sentence'].replace('؛', '.'), axis='columns')
    data17['sentence'] = data17.apply(lambda row: row['sentence'].replace('؛', '.'), axis='columns')
    data18['sentence'] = data18.apply(lambda row: row['sentence'].replace('؛', '.'), axis='columns')
    data16['sentence'] = data16.apply(lambda row: row['sentence'].replace(';', '.'), axis='columns')
    data17['sentence'] = data17.apply(lambda row: row['sentence'].replace(';', '.'), axis='columns')
    data18['sentence'] = data18.apply(lambda row: row['sentence'].replace(';', '.'), axis='columns')
    data16['sentence'] = data16.apply(lambda row: row['sentence'].replace('"', ''), axis='columns')
    data17['sentence'] = data17.apply(lambda row: row['sentence'].replace('"', ''), axis='columns')
    data18['sentence'] = data18.apply(lambda row: row['sentence'].replace('"', ''), axis='columns')
    data16['sentence'] = data16.apply(lambda row: row['sentence'].replace(',', '،'), axis='columns')
    data17['sentence'] = data17.apply(lambda row: row['sentence'].replace(',', '،'), axis='columns')
    data18['sentence'] = data18.apply(lambda row: row['sentence'].replace(',', '،'), axis='columns')

    data16['title'] = data16.apply(lambda row: str(row['title']).replace('؛', '.'), axis='columns')
    data17['title'] = data17.apply(lambda row: str(row['title']).replace('؛', '.'), axis='columns')
    data18['title'] = data18.apply(lambda row: str(row['title']).replace('؛', '.'), axis='columns')
    data16['title'] = data16.apply(lambda row: str(row['title']).replace(';', '.'), axis='columns')
    data17['title'] = data17.apply(lambda row: str(row['title']).replace(';', '.'), axis='columns')
    data18['title'] = data18.apply(lambda row: str(row['title']).replace(';', '.'), axis='columns')
    data16['title'] = data16.apply(lambda row: str(row['title']).replace('"', ''), axis='columns')
    data17['title'] = data17.apply(lambda row: str(row['title']).replace('"', ''), axis='columns')
    data18['title'] = data18.apply(lambda row: str(row['title']).replace('"', ''), axis='columns')
    data16['title'] = data16.apply(lambda row: str(row['title']).replace(',', '،'), axis='columns')
    data17['title'] = data17.apply(lambda row: str(row['title']).replace(',', '،'), axis='columns')
    data18['title'] = data18.apply(lambda row: str(row['title']).replace(',', '،'), axis='columns')
    data16['title'] = data16.apply(lambda row: str(row['title']).replace('nan', ''), axis='columns')
    data17['title'] = data17.apply(lambda row: str(row['title']).replace('nan', ''), axis='columns')
    data18['title'] = data18.apply(lambda row: str(row['title']).replace('nan', ''), axis='columns')

    data17 = data17.append(data16[2567:])
    data17.loc[2568:2799, ('subject1', 'subject2', 'subject3', 'suggested_subject',
                           'type1', 'type2', 'type3', 'suggested_type')] = np.nan
    data17.loc[2568:2799, ('subject_notsure', 'type_notsure')] = 0


def eval_types():
    print('not sure types for 16 : ' + str(data16['type_notsure'].value_counts()[1]))
    print('not sure types for 17 : ' + str(0))
    print('not sure types for 18 : ' + str(data18['type_notsure'].value_counts()[1]))
    print('suggested types for 16 : ' + str(data16[~ data16['suggested_type'].isnull()].shape[0]))
    print('suggested types for 17 : ' + str(data17[~ data17['suggested_type'].isnull()].shape[0]))
    print('suggested types for 18 : ' + str(data18[~ data18['suggested_type'].isnull()].shape[0]))


def add_to_subjs(sub, col):
    global subjs
    if sub is not np.nan and str(sub) != 'nan':
        if subjs[subjs['tag'] == sub].empty:
            subjs = subjs.append({'tag': sub, 'total_num': 0, 'num16': 0, 'num17': 0, 'num18': 0}, ignore_index=True)

        subjs.loc[subjs['tag'] == sub, 'total_num'] = subjs.loc[subjs['tag'] == sub, 'total_num'] + 1
        subjs.loc[subjs['tag'] == sub, col] = subjs.loc[subjs['tag'] == sub, col] + 1


def add_to_types(sub, col):
    global types
    if sub is not np.nan and str(sub) != 'nan':
        if types[types['tag'] == sub].empty:
            types = types.append({'tag': sub, 'total_num': 0, 'num16': 0, 'num17': 0, 'num18': 0}, ignore_index=True)

        types.loc[types['tag'] == sub, 'total_num'] = types.loc[types['tag'] == sub, 'total_num'] + 1
        types.loc[types['tag'] == sub, col] = types.loc[types['tag'] == sub, col] + 1


def eval_questions(result):
    for row_id, result_row in result.iterrows():  # for each question
        if row_id % 100 == 0: print(row_id)
        d16 = data16.loc[row_id]
        d17 = data17.loc[row_id]
        d18 = data18.loc[row_id]

        # evaluate subjects
        scores = dict()
        sbj_num = 0

        add_to_subjs(d16['subject1'], 'num16')
        add_to_subjs(d16['subject2'], 'num16')
        add_to_subjs(d16['subject3'], 'num16')
        add_to_subjs(d17['subject1'], 'num17')
        add_to_subjs(d17['subject2'], 'num17')
        add_to_subjs(d17['subject3'], 'num17')
        add_to_subjs(d18['subject1'], 'num18')
        add_to_subjs(d18['subject2'], 'num18')
        add_to_subjs(d18['subject3'], 'num18')

        scores, sbj_num = add_value(d16['subject1'], FIRST_SCORE - d16['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d16['subject2'], SECOND_SCORE - d16['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d16['subject3'], THIRD_SCORE - d16['subject_notsure'], scores, sbj_num)

        scores, sbj_num = add_value(d17['subject1'], FIRST_SCORE - d17['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d17['subject2'], SECOND_SCORE - d17['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d17['subject3'], THIRD_SCORE - d17['subject_notsure'], scores, sbj_num)

        scores, sbj_num = add_value(d18['subject1'], FIRST_SCORE - d17['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d18['subject2'], SECOND_SCORE - d17['subject_notsure'], scores, sbj_num)
        scores, sbj_num = add_value(d18['subject3'], THIRD_SCORE - d17['subject_notsure'], scores, sbj_num)

        Avg_sbj_num = sbj_num / 3.0
        sorted_scores = sorted(scores, key=scores.get, reverse=True)
        try:
            if scores[sorted_scores[0]] > 10:
                result_row['subject1'] = sorted_scores[0]
                result_row['s1 score'] = scores[sorted_scores[0]]
                result_row['subject2'] = sorted_scores[1]
                result_row['s2 score'] = scores[sorted_scores[1]]
                result_row['subject3'] = sorted_scores[2]
                result_row['s3 score'] = scores[sorted_scores[2]]
        except Exception as e:
            if str(e) != "list index out of range":
                print(e)
            pass
        try:
            if scores[sorted_scores[0]] > 10 and result_row['s1 score'] == result_row['s2 score']:
                result_row['subject_notsure'] = 1
        except:
            pass

        #########################################################
        # evaluate types
        scores = dict()
        typ_num = 0

        add_to_types(d16['type1'], 'num16')
        add_to_types(d16['type2'], 'num16')
        add_to_types(d16['type3'], 'num16')
        add_to_types(d17['type1'], 'num17')
        add_to_types(d17['type2'], 'num17')
        add_to_types(d17['type3'], 'num17')
        add_to_types(d18['type1'], 'num18')
        add_to_types(d18['type2'], 'num18')
        add_to_types(d18['type3'], 'num18')

        scores, typ_num = add_value(d16['type1'], FIRST_SCORE - d16['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d16['type2'], SECOND_SCORE - d16['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d16['type3'], THIRD_SCORE - d16['type_notsure'], scores, typ_num)

        scores, typ_num = add_value(d17['type1'], FIRST_SCORE - d17['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d17['type2'], SECOND_SCORE - d17['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d17['type3'], THIRD_SCORE - d17['type_notsure'], scores, typ_num)

        scores, typ_num = add_value(d18['type1'], FIRST_SCORE - d17['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d18['type2'], SECOND_SCORE - d17['type_notsure'], scores, typ_num)
        scores, typ_num = add_value(d18['type3'], THIRD_SCORE - d17['type_notsure'], scores, typ_num)

        Avg_typ_num = typ_num / 3.0
        sorted_scores = sorted(scores, key=scores.get, reverse=True)

        try:
            if scores[sorted_scores[0]] > 10:
                result_row['type1'] = sorted_scores[0]
                result_row['t1 score'] = scores[sorted_scores[0]]
                result_row['type2'] = sorted_scores[1]
                result_row['t2 score'] = scores[sorted_scores[1]]
                result_row['type3'] = sorted_scores[2]
                result_row['t3 score'] = scores[sorted_scores[2]]
        except Exception as e:
            if str(e) != "list index out of range":
                print(e)
            pass
        try:
            if scores[sorted_scores[0]] > 10 and result_row['t1 score'] == result_row['t2 score']:
                result_row['type_notsure'] = 1
        except:
            pass

        result_row['Avg_typ_num'] = Avg_typ_num
        result_row['Avg_sbj_num'] = Avg_sbj_num
        result.loc[row_id] = result_row


def add_value(name, score_value, scores, num):
    if name is not np.nan and str(name) != 'nan':
        num += 1
        if name in scores:
            scores[name] += score_value
        else:
            scores[name] = score_value
    return scores, num


def extract_suggestions():
    sub_sug = pd.DataFrame(columns=('subject', 'num', 'occurrence_list'))
    typ_sug = pd.DataFrame(columns=('type', 'num', 'occurrence_list'))
    for i, row in data16.iterrows():
        sub_sug = eval_row_for_suggestions(i, row, sub_sug, 'n16', 'subject')
        typ_sug = eval_row_for_suggestions(i, row, typ_sug, 'n16', 'type')
    for i, row in data17.iterrows():
        sub_sug = eval_row_for_suggestions(i, row, sub_sug, 'n17', 'subject')
        typ_sug = eval_row_for_suggestions(i, row, typ_sug, 'n17', 'type')
    for i, row in data18.iterrows():
        sub_sug = eval_row_for_suggestions(i, row, sub_sug, 'n18', 'subject')
        typ_sug = eval_row_for_suggestions(i, row, typ_sug, 'n18', 'type')

    sub_sug.to_csv("suggested_subjects.csv", sep=';', doublequote=True)
    typ_sug.to_csv("suggested_types.csv", sep=';', doublequote=True)


def eval_row_for_suggestions(i, row, sug, name, col):
    if str(row['suggested_' + col]) != 'nan':
        sub = row['suggested_' + col]
        if sug[sug[col] == sub].empty:
            sug = sug.append({col: sub,
                              'num': 1,
                              'occurrence_list': [[name, i]]}, ignore_index=True)
        else:
            sug.loc[sug[col] == sub, 'num'] = sug.loc[sug[col] == sub, 'num'] + 1
            sug.loc[sug[col] == sub, 'occurrence_list'].iloc[0].append([name, i])
    return sug


def combine():
    global types
    global subjs

    read_files()
    # eval_subjects()
    # eval_types()

    # creating a result DataFrame and initiate its columns with nan and 0 values
    result = data16.copy()
    result.loc[:, ('subject1', 'subject2', 'subject3', 'suggested_subject',
                   'type1', 'type2', 'type3', 'suggested_type')] = np.nan
    result.loc[:, ('subject_notsure', 'type_notsure')] = 0
    result['s1 score'] = [0 for _ in range(len(result))]
    result['s2 score'] = [0 for _ in range(len(result))]
    result['s3 score'] = [0 for _ in range(len(result))]

    result['t1 score'] = [0 for _ in range(len(result))]
    result['t2 score'] = [0 for _ in range(len(result))]
    result['t3 score'] = [0 for _ in range(len(result))]

    result['Avg_sbj_num'] = [0 for _ in range(len(result))]
    result['Avg_typ_num'] = [0 for _ in range(len(result))]

    types = pd.DataFrame(columns=('tag', 'total_num', 'num16', 'num17', 'num18'))
    subjs = pd.DataFrame(columns=('tag', 'total_num', 'num16', 'num17', 'num18'))

    eval_questions(result)

    print(types)
    print(subjs)

    result.to_csv("result.csv", sep=';', doublequote=True)
    types.to_csv("types.csv", sep=';', doublequote=True)
    subjs.to_csv("subjs.csv", sep=';', doublequote=True)


def shared_opinion():
    # read_files()
    # extract_suggestions()
    sug_sub = pd.read_csv('suggested_subjects.csv', delimiter=';')
    typ_sub = pd.read_csv('suggested_types.csv', delimiter=';')
    print(typ_sub)

    # # for i, row in sug_sub.iterrows():
    # #     occ_list = eval(row['occurrence_list'])
    # #     if len(occ_list) > 1:
    # #         for occ1 in occ_list:
    # #             for occ2 in occ_list:
    # #                 if occ1[0] != occ2[0] and occ1[1] == occ2[1]:
    # #                     print row['subject'], occ1, occ2  # Shared opinion
    # print "---------"
    for i, row in typ_sub.iterrows():
        occ_list = eval(row['occurrence_list'])
        if len(occ_list) > 1:
            for occ1 in occ_list:
                for occ2 in occ_list:
                    if occ1[0] != occ2[0] and occ1[1] == occ2[1]:
                        print(row['type'], occ1, occ2)  # Shared opinion


def remove_low_scores_and_save_subtyps():
    global types
    global subjs
    result = pd.read_csv('result_filtered.csv', delimiter=';')
    sub = [0, 0, 0, 0]
    typ = [0, 0, 0, 0]
    types = pd.DataFrame(columns=('tag', 'num'))
    subjs = pd.DataFrame(columns=('tag', 'num'))
    for row_id, row in result.iterrows():  # remove the ones with a score less than 11
        if row_id % 100 == 0: print(row_id)

        # if row['s1 score'] <= 10:
        #     sub[0] = sub[0] + 1
        #     result.loc[row_id, ('subject1', 'subject2', 'subject3')] = np.nan
        #     result.loc[row_id, ('s1 score', 's2 score', 's3 score')] = 0
        # elif row['s2 score'] <= 10:
        #     sub[1] = sub[1] + 1
        #     result.loc[row_id, ('subject2', 'subject3')] = np.nan
        #     result.loc[row_id, ('s2 score', 's3 score')] = 0
        # elif row['s3 score'] <= 10:
        #     sub[2] = sub[2] + 1
        #     result.loc[row_id, 'subject3'] = np.nan
        #     result.loc[row_id, 's3 score'] = 0
        # else:
        #     sub[3] = sub[3] + 1
        #
        # if row['t1 score'] <= 10:
        #     typ[0] = typ[0] + 1
        #     result.loc[row_id, ('type1', 'type2', 'type3')] = np.nan
        #     result.loc[row_id, ('t1 score', 't2 score', 't3 score')] = 0
        # elif row['t2 score'] <= 10:
        #     typ[1] = typ[1] + 1
        #     result.loc[row_id, ('type2', 'type3')] = np.nan
        #     result.loc[row_id, ('t2 score', 't3 score')] = 0
        # elif row['t3 score'] <= 10:
        #     typ[2] = typ[2] + 1
        #     result.loc[row_id, 'type3'] = np.nan
        #     result.loc[row_id, 't3 score'] = 0
        # else:
        #     typ[3] = typ[3] + 1

        subject = row['subject1']
        if subject is not np.nan and str(subject) != 'nan':
            if subjs[subjs['tag'] == subject].empty:
                subjs = subjs.append({'tag': subject, 'num': 0}, ignore_index=True)
            subjs.loc[subjs['tag'] == subject, 'num'] = subjs.loc[subjs['tag'] == subject, 'num'] + 1

        subject = row['subject2']
        if subject is not np.nan and str(subject) != 'nan':
            if subjs[subjs['tag'] == subject].empty:
                subjs = subjs.append({'tag': subject, 'num': 0}, ignore_index=True)
            subjs.loc[subjs['tag'] == subject, 'num'] = subjs.loc[subjs['tag'] == subject, 'num'] + 1

        subject = row['subject3']
        if subject is not np.nan and str(subject) != 'nan':
            if subjs[subjs['tag'] == subject].empty:
                subjs = subjs.append({'tag': subject, 'num': 0}, ignore_index=True)
            subjs.loc[subjs['tag'] == subject, 'num'] = subjs.loc[subjs['tag'] == subject, 'num'] + 1

        ####################################################
        type = row['type1']
        if type is not np.nan and str(type) != 'nan':
            if types[types['tag'] == type].empty:
                types = types.append({'tag': type, 'num': 0}, ignore_index=True)
            types.loc[types['tag'] == type, 'num'] = types.loc[types['tag'] == type, 'num'] + 1

        type = row['type2']
        if type is not np.nan and str(type) != 'nan':
            if types[types['tag'] == type].empty:
                types = types.append({'tag': type, 'num': 0}, ignore_index=True)
            types.loc[types['tag'] == type, 'num'] = types.loc[types['tag'] == type, 'num'] + 1

        type = row['type3']
        if type is not np.nan and str(type) != 'nan':
            if types[types['tag'] == type].empty:
                types = types.append({'tag': type, 'num': 0}, ignore_index=True)
            types.loc[types['tag'] == type, 'num'] = types.loc[types['tag'] == type, 'num'] + 1

    print('sub', sub)
    print('typ', typ)
    # result.to_csv('result_filtered.csv', sep=';')

    types.to_csv("types-result.csv", sep=';', doublequote=True)
    subjs.to_csv("subjs-result.csv", sep=';', doublequote=True)

combine()
# shared_opinion()
# remove_low_scores_and_save_subtyps()
# read_files()
