import pandas as pd


def read_files():
    global questions16
    global questions16_2
    global questions17
    global questions18
    global questions18_2

    # id;title;sentence;subject1;subject2;subject3;subject_notsure;suggested_subject;type1;type2;type3;type_notsure;suggested_type
    questions16 = pd.read_csv('16-javaheri.csv', delimiter=';')
    questions17 = pd.read_csv('17-sheikholeslami.csv', delimiter=';')
    questions18 = pd.read_csv('18-sayahi.csv', delimiter=';')

    questions16 = questions16[0:2567]
    questions16_2 = questions16[2567:2799]
    questions18 = questions18[0:2567]
    questions18_2 = questions18[2567:2799]
    # print questions16['sentence']
    # print questions16_2['sentence']
    # print questions17['sentence']
    # print questions18['sentence']
    # print questions18_2['sentence']


def eval_subjects():
    print 'not sure subjects for 16 : ' + str(questions16['subject_notsure'].value_counts()[1])
    print 'not sure subjects for 17 : ' + str(questions17['subject_notsure'].value_counts()[1])
    print 'not sure subjects for 18 : ' + str(questions18['subject_notsure'].value_counts()[1])

    print 'suggested subjects for 16 : ' + str(questions16.shape[0]
                                               - (questions16[questions16['suggested_subject'].isnull()].shape[0]))
    print 'suggested subjects for 17 : ' + str(questions17.shape[0]
                                               - (questions17[questions17['suggested_subject'].isnull()].shape[0]))
    print 'suggested subjects for 18 : ' + str(questions18.shape[0]
                                               - (questions18[questions18['suggested_subject'].isnull()].shape[0]))

read_files()
eval_subjects()