import numpy as np
import pandas as pd

class Biofile(object):
    """this class is the data get from the input txt file.
       data: transfered txt data from the given address """

    def __init__(self, table):
        self.table = pd.read_table(table, sep='\s+')

    def get_matrix(self):
        """get the value matrix of table"""
        return self.table.values

    def get_header(self):
        return self.table.columns

    def get_index(self):
        return self.table.index

    def get_column(self, column):
        """get the column of table, where column should be name"""
        return np.array(self.table[column])

    def get_value(self, column, row):
        """get the certain value of table, where column and row should be name"""
        return self.table[column][row]
