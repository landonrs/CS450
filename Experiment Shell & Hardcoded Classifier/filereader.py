import pandas
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
import re
import csv


class FileReader(object):

    def __init__(self):
        pass

    def read_data_from_file(self, file_path, delimiter=','):
        """reads in data from a file and separates targets from data. The method assumes the target is the last column.
        If any of the data or targets of the file contain non-numeric values, the values are converted to an id number
        using skLearns labelEncoder class.
        file_path - path to the text or csv file
        delimiter - the character separating data points, default set to ','"""

        # used to convert non numeric data into numeric
        encoder = LabelEncoder()
        # read the file using pandas function
        data_set = pandas.read_csv(file_path, delimiter=delimiter, header=None)
        # print(data_set)

        # check each column for non numeric data and use the encoder to convert it
        for column in data_set:
            if is_string_dtype(data_set[column]):
                encoder.fit(data_set[column])
                data_set[column] = encoder.transform(data_set[column])

        # get the targets by grabbing the last col of the data set
        data_targets = data_set.iloc[:, -1]
        # print(data_targets)

        # return the data and targets as matrices to work with skLearn
        return data_set.iloc[:, :-1].as_matrix(), data_targets.as_matrix()










