import pandas as pd
import numpy as np


def read_files():
    global data16
    global data17
    global data18

    # id;title;sentence;subject1;subject2;subject3;subject_notsure;suggested_subject;type1;type2;type3;type_notsure;suggested_type
    data16 = pd.read_csv('16-javaheri.csv', delimiter=';', index_col=['id'])
    data17 = pd.read_csv('17-sheikholeslami.csv', delimiter=';', index_col=['id'])
    data18 = pd.read_csv('18-sayahi.csv', delimiter=';', index_col=['id'])

    data16 = data16[0:2799]
    data18 = data18[0:2799]
    data17 = data17.append(data16[2567:2799])

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


read_files()
# eval_subjects()
# eval_types()


