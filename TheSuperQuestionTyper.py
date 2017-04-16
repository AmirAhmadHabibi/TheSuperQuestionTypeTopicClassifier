import pandas as pd

# id;"title";"sentence";"subject1";"subject2";"subject3";"subject_notsure";"suggested_subject";"type1";"type2";"type3";"type_notsure";"suggested_type"

questions16 = pd.read_csv('16-javaheri.csv',  delimiter=';')
questions17 = pd.read_csv('17-sheikholeslami.csv', delimiter=';')
questions18 = pd.read_csv('18-sayahi.csv', delimiter=';')
