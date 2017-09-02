# encoding: utf-8
import pandas as pd
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def cat_typ_correlation():
    subjs = pd.read_csv('subject_vector_Q.csv')
    types = pd.read_csv('type_vector_Q.csv')
    corrlist = pd.DataFrame(columns={'sub', 'typ', 'corr'})
    corr = pd.DataFrame()
    for subj in subjs:
        subject_col = subjs[subj]
        for typ in types:
            type_col = types[typ]
            c = np.corrcoef(subject_col, type_col)[0][1]
            corr.loc[subj, typ] = c
            corrlist = corrlist.append({'sub': subj, 'typ': typ, 'corr': float(c)}, ignore_index=True)

    corrlist['abs corr'] = corrlist['corr']
    corrlist['abs corr'] = corrlist.apply(lambda row: abs(row['abs corr']), axis=1)
    corrlist = corrlist.sort('abs corr', ascending=False)
    corrlist.__delitem__('abs corr')
    corrlist = corrlist.reset_index(drop=True)
    print corrlist.as_matrix()
    print '-----------------'


# cat_typ_correlation()

##############################################################################################################

def get_items(name, q_row):
    items = set()
    for i in range(1, 4):
        if str(q_row[name + str(i)]) != 'nan':
            items.add(q_row[name + str(i)])
    return items


def get_agreement(data1, data2):
    # each question Agreement  = 2 ∗ common labels in both anns / (labels in ann1 + labels in ann2)
    agreements = pd.DataFrame(
        columns={'q id', 'subjects1', 'subjects2', 'sub agreement', 'types1', 'types2', 'typ agreement'})
    avg_on_sub_agreement = 0.0
    sub_num = 0.0
    avg_on_typ_agreement = 0.0
    typ_num = 0.0
    for i in range(1, 2800):
        row1 = data1.loc[i]
        row2 = data2.loc[i]
        sub1 = get_items('subject', row1)
        sub2 = get_items('subject', row2)
        typ1 = get_items('type', row1)
        typ2 = get_items('type', row2)
        sub_agreement = 0.0
        if len(sub1) > 0 and len(sub2) > 0:
            sub_agreement = 2.0 * (len(sub1.intersection(sub2))) / (len(sub1) + len(sub2))
            sub_num += 1
            avg_on_sub_agreement += sub_agreement

        typ_agreement = 0.0
        if len(typ1) > 0 and len(typ2) > 0:
            typ_agreement = 2.0 * (len(typ1.intersection(typ2))) / (len(typ1) + len(typ2))
            typ_num += 1
            avg_on_typ_agreement += typ_agreement

            # agreements = agreements.append(
            #     {'q id': i, 'subjects1': list(sub1), 'subjects2': list(sub2), 'sub agreement': sub_agreement,
            #      'types1': list(typ1), 'types2': list(typ2), 'typ agreement': typ_agreement}, ignore_index=True)
            # print agreements

    avg_on_sub_agreement = avg_on_sub_agreement / sub_num
    avg_on_typ_agreement = avg_on_typ_agreement / typ_num

    return avg_on_sub_agreement, avg_on_typ_agreement, agreements


def get_agreement_of_all():
    data16 = pd.read_csv('./1_combine_tags/16-javaheri_e.csv', delimiter=';', index_col=['id'])
    data17 = pd.read_csv('./1_combine_tags/17-sheikholeslami_e.csv', delimiter=';', index_col=['id'])
    data18 = pd.read_csv('./1_combine_tags/18-sayahi_e.csv', delimiter=';', index_col=['id'])

    print '           agreement on subjects  agreement on types'
    sub_agg, typ_agg, agreements16_17 = get_agreement(data16, data17)
    print '16 - 17   ', sub_agg, '       ', typ_agg
    # agreements16_17.to_csv('./1_combine_tags/agreement_16_17.csv', index=False)

    sub_agg, typ_agg, agreements16_18 = get_agreement(data16, data18)
    print '16 - 18   ', sub_agg, '       ', typ_agg
    # agreements16_18.to_csv('./1_combine_tags/agreement_16_18.csv', index=False)

    sub_agg, typ_agg, agreements17_18 = get_agreement(data17, data18)
    print '17 - 18   ', sub_agg, '       ', typ_agg
    # agreements17_18.to_csv('./1_combine_tags/agreement_17_18.csv', index=False)


# get_agreement_of_all()
##################################################################################################################

def eval_category(cat):
    data16 = pd.read_csv('./1_combine_tags/16-javaheri_e.csv', delimiter=';', index_col=['id'])
    data17 = pd.read_csv('./1_combine_tags/17-sheikholeslami_e.csv', delimiter=';', index_col=['id'])
    data18 = pd.read_csv('./1_combine_tags/18-sayahi_e.csv', delimiter=';', index_col=['id'])

    print 'not sure ' + cat + 's for 16 : ' + str(data16[cat + '_notsure'].value_counts()[1])
    print 'suggested ' + cat + 's for 16 : ' + str(data16[~data16['suggested_' + cat].isnull()].shape[0])
    print 'no label and Suggestion for 16 : ' + str(
        data16[(data16['suggested_' + cat].isnull()) & (data16[cat + '1'].isnull())].shape[0])
    print '1 label 16 : ' + str(data16[(data16[cat + '2'].isnull()) & (~data16[cat + '1'].isnull())].shape[0])
    print '2 label 16 : ' + str(data16[data16[cat + '3'].isnull() & ~data16[cat + '2'].isnull()].shape[0])
    print '3 label 16 : ' + str(data16[~data16[cat + '3'].isnull()].shape[0])

    print 'not sure ' + cat + 's for 17 : ' + str(0)
    print 'suggested ' + cat + 's for 17 : ' + str(data17[~data17['suggested_' + cat].isnull()].shape[0])
    print 'no label and Suggestion for 17 : ' + str(
        data17[data17['suggested_' + cat].isnull() & data17[cat + '1'].isnull()].shape[0])
    print '1 label 17 : ' + str(data17[data17[cat + '2'].isnull() & ~data17[cat + '1'].isnull()].shape[0])
    print '2 label 17 : ' + str(data17[data17[cat + '3'].isnull() & ~data17[cat + '2'].isnull()].shape[0])
    print '3 label 17 : ' + str(data17[~data17[cat + '3'].isnull()].shape[0])

    print 'not sure ' + cat + 's for 18 : ' + str(data18[cat + '_notsure'].value_counts()[1])
    print 'suggested ' + cat + 's for 18 : ' + str(data18[~data18['suggested_' + cat].isnull()].shape[0])
    print 'no label and Suggestion for 18 : ' + str(
        data18[data18['suggested_' + cat].isnull() & data18[cat + '1'].isnull()].shape[0])
    print '1 label 18 : ' + str(data18[data18[cat + '2'].isnull() & ~data18[cat + '1'].isnull()].shape[0])
    print '2 label 18 : ' + str(data18[data18[cat + '3'].isnull() & ~data18[cat + '2'].isnull()].shape[0])
    print '3 label 18 : ' + str(data18[~data18[cat + '3'].isnull()].shape[0])


eval_category(cat='type')
