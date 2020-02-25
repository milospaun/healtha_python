from os.path import isfile, join

import numpy as np
import os
import re
import shutil

import pandas as pd

import signal_processing as sp
import util
from data.DataAdapter import DataAdapter
from SignalProcessing import SignalProcessing


class HaptAdapter(DataAdapter):
    _freq = 50  # sample frequency
    _window = 128  # size pf the window - umber of samples from which we calculate features
    _overlap = 11  # overlap of the windows
    _tr_test_split_coef = 0.8  # split to train and test files

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        super().__init__()
        self._raw_dir = raw_dir
        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        self._processed_dir = "{0}\\datasets\\Processed\\Hapt".format(parentDir)
        self._features_dir = "{0}\\datasets\\Features\\Hapt".format(parentDir)

        if not os.path.isdir(self._processed_dir):
            os.makedirs(self._processed_dir)

        if not os.path.isdir(self._features_dir):
            os.makedirs(self._features_dir)

        self._sp = SignalProcessing(self._freq, self._window, self._overlap)

    # instance method
    def preprocess_file(self, filename):
        print("\n Processing file: " + filename)
        m = re.search('acc_', filename)
        n = re.search('gyro_', filename)
        if m is not None or n is not None:
            data = np.loadtxt(self._raw_dir + "\\" + filename)
            # x = self._sp.pre_process_3D_data(data, self._freq, self._window, self._overlap)
            np.savetxt(self._features_dir + "\\" + filename, data, "%.9f")
        return x

    def preprocess_dir(self):
        if not os.path.isdir(self._features_dir):
            os.makedirs(self._features_dir)

        count = 0
        for filename in os.listdir(self._raw_dir):
            print("\n Processing file: " + filename)
            m = re.search('acc_', filename)
            n = re.search('gyro_', filename)
            if m is not None or n is not None:
                data = np.loadtxt(self._raw_dir + "\\" + filename)
                # x = sp.pre_process_data(data, self._freq, self._window, self._overlap)
                np.savetxt(self._processed_dir + "\\" + filename, data, "%.9f")
                count = count + 1

        if os.path.isfile(self._raw_dir + "\\" + "labels.txt"):
            shutil.copyfile(self._raw_dir + "\\" + "labels.txt", self._processed_dir + "\\" + "labels.txt")

        print("Total files processed: " + str(count))
        pass

    def build_file(self, filename):
        pass

    def build_dir(self, labels="time", mod="ALL", num_exp=61):
        """
        :param labels: feature is labeled by "time" interval, or "number" interval
        :param mod: A for activity, PT for posture and ALL for all
        :param num_exp: number of experiments or files
        :return:
        """
        path = self._processed_dir
        dst_dir = self._features_dir

        # read label data
        y = []
        x = []
        features = []

        # set labels for examples
        labels_data = self.read_labels(path + "\\labels.txt")

        # get all files from directory
        file_names = ""
        for f in os.listdir(path):
            if isfile(join(path, f)):
                file_names = file_names + " " + f

        for exp_num in range(1, num_exp + 1):
            print("Exp: " + str(exp_num))
            try:
                # acc raw file
                m = re.search('acc_exp' + format(exp_num, '02d'), file_names)
                if m is None:
                    raise FileNotFoundError
                file_path = path + "\\" + file_names[m.start():m.start() + 20]
                acc_raw_data = self.read_ndarray_from_cvs(file_path)
                acc_raw_data = acc_raw_data[:, 0:3]  # in case we had 4th column time

                # gyro raw file
                m = re.search('gyro_exp' + format(exp_num, '02d'), file_names)
                file_path = path + "\\" + file_names[m.start():m.start() + 21]
                gyro_raw_data = self.read_ndarray_from_cvs(file_path)
                gyro_raw_data = gyro_raw_data[:, 0:3]

                # normalize - should i normilize raw input? No acc_raw_data = util.normalize(acc_raw_data)

                # Prepossessing step - data must be split into n * window values. Each window has size dt * _window.
                # for ex. dt = 0.02 (50HZ semp. rate), _window=128. Data splited in 2.56s windows. Fill missing data.

                # calculate features and append all examples
                file_features = self._sp.process_raw_data(acc_raw_data, gyro_raw_data)
                features.append(file_features)
                file_features = file_features.T

                # find labels
                slide = self._window - self._overlap
                for f in range(0, file_features.shape[0]):
                    p = 0
                    for t in labels_data.get(str(exp_num)):
                        if float(t[1]) <= (f + 1) * slide and ((f + 1) * slide <= float(t[2])):
                            p = 1
                            break

                    if p == 1:
                        y.append([int(t[0]), int((f + 1) * slide), int(exp_num)])
                        x.append(file_features[f])

            except FileNotFoundError:
                print("Oops!  File with number" + str(exp_num) + " was not processed. File path: " + file_path)
                continue

        x = np.vstack(x)
        x = util.normalize(x)
        np.savetxt(self._features_dir + "\\features.txt", x, "%.9f")

        y = np.array(y)
        y = np.vstack(y)
        np.savetxt(self._features_dir + "\\labels.txt", y, "%.2f")
        print(f" Shape of X after labeling: {x.shape} and Y {y.shape}")

        if mod == "A":
            # set all PT to value 7 as we are only interested in activities
            filt = (y >= 7)
            filt[:, 1:3] = False
            y[filt] = 7
            # x, y = filter_lower_then(np.array(x), np.array(y[:, 0]), 7)
        elif mod == 'PT':
            x, y = self._sp.filter_greater_then(np.array(x), np.array(y[:, 0]), 7)

        return x
