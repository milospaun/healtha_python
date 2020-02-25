# Parent class
import re
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import util
import src.SignalProcessing as sp


class DataAdapter:
    # Class attribute
    _raw_dir = ""
    _processed_dir = ""
    _features_dir = ""

    # Initializer / Instance attributes
    def __init__(self):
        pass

    # instance method
    def preprocess_file(self, filename):
        pass

    def preprocess_dir(self):
        pass

    def build_file(self, filename):
        pass

    def build_dir(self):
        pass

    # read labels.txt file
    @staticmethod
    def read_labels(file):
        """
        Read labels data and create dictionary wher experiment is the key to the list of tuples
        (label, start, end). Start and end values can be in seconds or in number of examples.
        :param file:
        :return:
        """
        file = open(file, "r")

        nl = 0
        data = dict()
        tuplist = []
        last_exp_num = -1
        for line in file:
            nl = nl + 1
            tmp = line.split()
            tup = (tmp[2], tmp[3], tmp[4])
            if last_exp_num == tmp[0]:
                tuplist.append(tup)
            else:
                data[last_exp_num] = tuplist
                last_exp_num = tmp[0]
                tuplist = [tup]
        data[last_exp_num] = tuplist

        data.pop(-1)
        file.close()
        return data

    @staticmethod
    def read_ndarray_from_cvs(file):
        """

        :param file: string
        :return: numpy array num_lines x num_columns
        """
        file = open(file, "r")

        nl = 0
        data = []
        for line in file:
            nl = nl + 1
            tmp = line.split()
            for x in tmp:
                data.append(np.float32(x))

        # RESHAPE n is num of features, m is num of examples
        num_columns = len(tmp)
        data = np.reshape(data, (nl, num_columns))
        file.close()
        return data

    @staticmethod
    def read_dataframe_from_cvs(file):
        """

        :param file: string
        :return: dataframe num_lines x num_columns
        """
        data = pd.read_csv(file, sep=' ', header=None)
        return data
