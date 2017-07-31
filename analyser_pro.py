import pandas as pd
import numpy as np


def cat_typ_correlation():
    subjs = pd.read_csv('subject_vector_Q.csv')
    types = pd.read_csv('type_vector_Q.csv')
    # subjs=subjs.as_matrix()
    # types=types.as_matrix()
    corr = pd.DataFrame()
    for subj in subjs:
        subject_col = subjs[subj]
        for typ in types:
            type_col = types[typ]
            corr.loc[subj, typ] = np.corrcoef(subject_col, type_col)[0][1]

    print corr
    print '-----------------'


cat_typ_correlation()
