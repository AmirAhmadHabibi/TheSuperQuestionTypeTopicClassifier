import os
from utilitarianism import QuickDataFrame


class DataHandler:
    def __init__(self):
        if os.path.isfile('./raw_data.csv'):
            self.data = QuickDataFrame.read_csv('./raw_data.csv')
        else:
            self.data = QuickDataFrame(columns=['question', 'topics', 'types'])

    def add_question(self, qst, tpcs, typs):
        if self.data.length > 0 and self.data['question'][-1] == qst:
            self.data.delete_row(-1)
        self.data.append([qst, tpcs, typs])
        self.data.to_csv('./raw_data.csv')
