from copy import copy
from sys import stdout as so
import time
import csv


class Progresser:
    def __init__(self, total_num, msg=''):
        self.total = total_num
        self.num = 0
        self.start_time = time.time()
        self.msg = msg

    def count(self):
        self.show_progress(self.num)
        self.num += 1

    def show_progress(self, current_num):
        if current_num % 10 != 0:
            return
        eltime = time.time() - self.start_time
        retime = (self.total - current_num - 1) * eltime / (current_num + 1)

        el_str = str(int(eltime / 3600)) + ':' + str(int((eltime % 3600) / 60)) + ':' + str(int(eltime % 60))
        re_str = str(int(retime / 3600)) + ':' + str(int((retime % 3600) / 60)) + ':' + str(int(retime % 60))

        so.write('\r' + self.msg + '\ttime: ' + el_str + ' + ' + re_str
                 + '\t\tprogress: %' + str(round(100 * (current_num + 1) / self.total, 2)) + '\t')


class QuickDataFrame:
    """A dictionary of lists
        each column is a dictionary key
        each row is an index in all of the lists"""

    def __init__(self, columns=None):
        if columns is None:
            columns = []
        self.cols = []
        self.data = dict()

        # set column names
        unnamed_index = 0
        for col in columns:
            # if no name
            if col == '' or col is None:
                col = 'Unnamed: ' + str(unnamed_index)
                unnamed_index += 1
            else:
                col = str(col)
            # if duplicate name
            if col in self.data:
                col += 'I'
            # set the name
            self.cols.append(col)
            self.data[col] = []

        self.length = 0
        self.index = None

    def append(self, row=None, value=None):
        """ three options for input
            row = list
            row = dict
            value = not None   : puts the value for in element in the row
        """
        if value is not None:
            for col in self.cols:
                self.data[col].append(copy(value))

            self.length += 1
            self.index = None
            return
        elif row is not None:
            if len(row) != len(self.cols):
                raise Exception('Number of items in input row must be equal to the number of columns.')

            if type(row) == dict:
                for key, val in row.items():
                    self.data[key].append(copy(val))
            elif type(row) == list:
                for i in range(len(self.cols)):
                    self.data[self.cols[i]].append(copy(row[i]))
            self.length += 1
            self.index = None

    def add_column(self, name, value=None):
        """Adds a column and fills it with None values"""
        name = str(name)
        if name in self.data:
            name += 'I'
        self.cols.append(name)
        self.data[name] = [copy(value) for _ in range(self.length)]

    def delete_column(self, name):
        if name in self.data:
            del self.data[name]
            self.cols.remove(name)

    def rename(self, columns):
        """renames each key in the input dictionary to its value"""
        for old, new in columns.items():
            if old in self.cols:
                self.data[new] = self.data.pop(old)
                self.cols[self.cols.index(old)] = new
        return self

    def rows_equal_to(self, column, value):
        """get a QDF containing the rows in which the given column have the given value"""
        qdf = QuickDataFrame(self.cols)
        for i in range(self.length):
            if self.data[column][i] == value:
                qdf.append(self[i])
        return qdf

    def shape(self):
        """returns the number of rows, the number of columns"""
        return [self.length, len(self.cols)]

    def row_as_dict(self, i):
        """don't use it in large numbers. It slows you down"""
        if i < 0 or i >= self.length:
            raise IndexError('index out of range')
        row = dict()
        for col in self.cols:
            row[col] = self.data[col][i]
        return row

    def row_as_list(self, i):
        if i < 0 or i >= self.length:
            raise IndexError('index out of range')
        row = []
        for col in self.cols:
            row.append(self.data[col][i])
        return row

    def delete_row(self, i):
        """ deletes the ith row
            resets the index
            all rows after i would shift by one"""

        if i < 0 or i >= self.length:
            raise IndexError('index out of range')
        for col in self.cols:
            del self.data[col][i]
        self.index = None
        self.length -= 1

    def apply(self, func, axis='columns'):
        result = []
        if axis == 'columns':
            for i in range(self.length):
                result.append(func(self.row_as_dict(i)))
        if axis == 'rows':
            for i in range(len(self.cols)):
                result.append(func(self.data[self.cols[i]]))
        return result

    def set_index(self, index_list, unique=True):
        """ if unique is True:
                assigns a row number to each element in index_list

            if unique is false:
                assigns a list of row numbers to each element in index_list
                that can be used like this afterwards:

                i_list= qdf.index['foo']

                for i in i_list:
                    val = qdf['col'][i]
                    ...

            * index keys would be cast to str
        """
        if len(index_list) != self.length:
            raise Exception('index must have equal length with the QDF')
        self.index = dict()
        if unique:
            for i in range(self.length):
                key = str(index_list[i])
                if key in self.index:
                    raise Exception('index values must be unique if unique=True.')
                # if key in self.data:
                #     raise Exception('index values must not be in column names.')
                self.index[key] = i
        else:
            for i in range(self.length):
                key = str(index_list[i])
                # if key in self.data:
                #     raise Exception('index values must not be in column names.')
                if key not in self.index:
                    self.index[key] = []
                self.index[key].append(i)

    def index_is_unique(self):
        if self.index is None:
            return None
        if type(next(iter(self.index.values()))) == list:
            return False
        else:
            return True

    def __str__(self):
        out_str = ''
        # add column names
        for i in range(len(self.cols)):
            out_str += str(self.cols[i])
            if i + 1 < len(self.cols):
                out_str += ','
        out_str += '\n'

        for r in range(self.length):
            row_str = ''
            for i in range(len(self.cols)):
                item = str(self.data[self.cols[i]][r])
                if ',' in item:
                    item = '"' + item + '"'
                row_str += item
                if i + 1 < len(self.cols):
                    row_str += ','

            if r + 1 < self.length:
                row_str += '\n'

            out_str += row_str
        return out_str

    def __getitem__(self, arg):
        """
            qdf[5]
                if arg is int returns the arg'th row as a dict

            qdf['foo_col'] or qdf['foo_index']
                if arg is a str
                    returns the column arg if arg is in column
                    else returns the row(s) with the index if arg in index
                        if unique one row else a new QDF with those rows

            qdf['foo_col', 'foo_index']
                if it's a tuple then uses arg[0] as column name and arg[1] as index name
                    then if index is unique returns the one element
                    if not, returns a list of all the elements
        """
        if type(arg) == int:
            return self.row_as_dict(arg)

        elif type(arg) == str:
            # if a column return the list in that column
            if arg in self.data:
                return self.data[arg]
            # if an index then return a row or a new QDF of those rows
            elif self.index is not None and arg in self.index:
                row_num = self.index[arg]
                if type(row_num) == int:
                    return self.row_as_dict(row_num)
                else:
                    qdf = QuickDataFrame(self.cols)
                    for i in row_num:
                        qdf.append(self.row_as_list(i))
                    return qdf
            else:
                raise Exception('key not in column list nor index list')

        elif type(arg) == tuple:
            if self.index is None:
                raise Exception('the index has not been set. this method of get Item needs index')
            col, ind = arg
            ind = str(ind)
            if col not in self.data:
                raise Exception('column not in column list')
            if ind not in self.index:
                raise Exception('Index key not in index')

            row_num = self.index[ind]
            if type(row_num) == int:
                return self.data[col][row_num]
            else:
                elements = []
                for i in row_num:
                    elements.append(self.data[col][i])
                return elements

    def __setitem__(self, key, value):
        """
            qdf['foo_col', 'foo_index'] =value

            if index is not unique then value would be set to all those rows having the index
        """
        col, ind = key
        ind = str(ind)
        if col not in self.data:
            raise Exception('column not in column list')
        if self.index is None:
            raise Exception('the index has not been set. this method of get Item needs index')
        if ind not in self.index:
            raise Exception('Index key not in index')

        row_num = self.index[ind]
        if type(row_num) == int:
            self.data[col][row_num] = value
        else:
            for i in row_num:
                self.data[col][i] = value

    def __len__(self):
        return self.length

    def to_csv(self, path):
        with open(path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(self.cols)
            for r in range(self.length):
                writer.writerow(self.row_as_list(r))

    @staticmethod
    def read_csv(path, columns=[], header=True, sep=','):
        qdf = QuickDataFrame()
        with open(path, 'r', encoding='utf-8') as infile:
            first_line = True
            for line_tokens in csv.reader(infile, delimiter=sep):
                if first_line:
                    if header:
                        if columns:
                            qdf = QuickDataFrame(columns)
                        else:
                            qdf = QuickDataFrame(line_tokens)
                    else:
                        if columns:
                            qdf = QuickDataFrame(columns)
                        else:
                            qdf = QuickDataFrame(['col' + str(i) for i in range(len(line_tokens))])
                        qdf.append(line_tokens)
                    first_line = False
                else:
                    qdf.append(line_tokens)
        return qdf
