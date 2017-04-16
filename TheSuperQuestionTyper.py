import pandas as pd

# id;"title";"sentence";"subject1";"subject2";"subject3";"subject_notsure";"suggested_subject";"type1";"type2";"type3";"type_notsure";"suggested_type"

questions16 = pd.read_csv('16-javaheri.csv',  delimiter=';')
questions17 = pd.read_csv('17-sheikholeslami.csv', delimiter=';')
questions18 = pd.read_csv('18-sayahi.csv', delimiter=';')

print 'The number of not sure subjects for 16 is ' + str(questions16['subject_notsure'].value_counts()[1])
print 'The number of not sure subjects for 17 is ' + str(questions17['subject_notsure'].value_counts()[1])
print 'The number of not sure subjects for 18 is ' + str(questions18['subject_notsure'].value_counts()[1])
