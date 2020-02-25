import numpy as np
import os
import re
import shutil
import logging

import util
from os.path import isfile, join
from src.SignalProcessing import SignalProcessing
from src.data_adapters.DataAdapter import DataAdapter

logger = logging.getLogger()


class VtsAdapterMobile(DataAdapter):
    _freq = 50  # sample frequency
    _window = 100  # size pf the window - umber of samples from which we calculate features
    _overlap = 50  # overlap of the windows
    _tr_test_split_coef = 0.8  # split to train and test files
    _num_exp = 15
    labels_to_text = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
        7: "STAND_TO_SIT",
        8: "SIT_TO_STAND",
        9: "SIT_TO_LIE",
        10: "LIE_TO_SIT",
        11: "STAND_TO_LIE",
        12: "LIE_TO_STAND"
    }

    _sp = None

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        super().__init__()
        self._raw_dir = raw_dir
        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        self._processed_dir = "{0}\\datasets\\Processed\\VtsMobile".format(parentDir)
        self._features_dir = "{0}\\datasets\\Features\\VtsMobile".format(parentDir)

        if not os.path.isdir(self._processed_dir):
            os.makedirs(self._processed_dir)

        if not os.path.isdir(self._features_dir):
            os.makedirs(self._features_dir)

        self._sp = SignalProcessing(self._freq, self._window, self._overlap)

        logger.info(f"Created VtsMobileAdapter with freq: {self._freq} window: {self._window} "
                    f"overlap: {self._overlap} raw dir: {self._raw_dir} features_dir: {self._features_dir}")

    def set_parameters(self, raw_dir, processed_dir, features_dir, freq, window, overlap, num_exp):
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self._features_dir = features_dir
        self._freq = freq
        self._window = window
        self._overlap = overlap
        self._num_exp = num_exp

        if not os.path.isdir(self._processed_dir):
            os.makedirs(self._processed_dir)

        if not os.path.isdir(self._features_dir):
            os.makedirs(self._features_dir)

        self._sp = SignalProcessing(self._freq, self._window, self._overlap)
        logger.info(f"Parameters changed to freq: {self._freq} window: {self._window} "
                    f"overlap: {self._overlap} raw dir: {self._raw_dir} features_dir: {self._features_dir}")

    # instance method
    def preprocess_file(self, filename):
        print("\n Processing file: " + filename)
        m = re.search('acc_', filename)
        n = re.search('gyro_', filename)
        x = []
        if m is not None or n is not None:
            data = np.loadtxt(self._raw_dir + "\\" + filename)
            x = self._sp.pre_process_4d_data(data)
        return x

    def preprocess_dir(self):
        """
        Method goes thorugh experiments, 2 files at the time and does some preprocessing.
        It verifys that the files are the same size.
        Calls method to corrects measured data by adding or removing samples so that each window will have exactly.
        :return: nothing
        """
        logger.info(f"Start Preprocessing from raw dir: {self._raw_dir} preprocess_dir: {self._processed_dir}")
        file_names = ""
        for f in os.listdir(self._raw_dir):
            if isfile(join(self._raw_dir, f)):
                file_names = file_names + " " + f

        for exp_num in range(1, self._num_exp + 1):
            print("Exp: " + str(exp_num))
            try:
                # acc raw file
                m = re.search('acc_exp' + format(exp_num, '02d'), file_names)
                if m is None:
                    raise FileNotFoundError
                filename_acc = file_names[m.start():m.start() + 20]
                acc_raw_data = self.read_ndarray_from_cvs(self._raw_dir + "\\" + filename_acc)

                # gyro raw file
                m = re.search('gyro_exp' + format(exp_num, '02d'), file_names)
                filename_gyro = file_names[m.start():m.start() + 21]
                gyro_raw_data = self.read_ndarray_from_cvs(self._raw_dir + "\\" + filename_gyro)

                # normalize - should i normalize raw input? No
                acc_raw_data = self._sp.pre_process_4d_data(acc_raw_data)
                gyro_raw_data = self._sp.pre_process_4d_data(gyro_raw_data)

                # check if they are the same size and correct
                if gyro_raw_data.shape[0] > acc_raw_data.shape[0]:
                    print(f"Gyro file has more samples, deleting {gyro_raw_data.shape[0] - acc_raw_data.shape[0]}")
                    gyro_raw_data = gyro_raw_data[:acc_raw_data.shape[0], :]

                elif acc_raw_data.shape[0] > gyro_raw_data.shape[0]:
                    print(f"Acc file has more samples, deleting {acc_raw_data.shape[0] - gyro_raw_data.shape[0]}")
                    acc_raw_data = acc_raw_data[:gyro_raw_data.shape[0], :]

                np.savetxt(self._processed_dir + "\\" + filename_acc, acc_raw_data, "%.9f")
                np.savetxt(self._processed_dir + "\\" + filename_gyro, gyro_raw_data, "%.9f")
            except FileNotFoundError:
                print("Oops!  Experiment with number" + str(exp_num) + " was not processed. ")
                continue

        if os.path.isfile(self._raw_dir + "\\" + "labels.txt"):
            shutil.copyfile(self._raw_dir + "\\" + "labels.txt", self._processed_dir + "\\" + "labels.txt")

        pass

    def build_data(self, acc, gyro):
        acc_raw_data = self._sp.pre_process_4d_data(acc)
        gyro_raw_data = self._sp.pre_process_4d_data(gyro)

        # check if they are the same size and correct
        if gyro_raw_data.shape[0] > acc_raw_data.shape[0]:
            print(f"Gyro file has more samples, deleting {gyro_raw_data.shape[0] - acc_raw_data.shape[0]}")
            gyro_raw_data = gyro_raw_data[:acc_raw_data.shape[0], :]

        elif acc_raw_data.shape[0] > gyro_raw_data.shape[0]:
            print(f"Acc file has more samples, deleting {acc_raw_data.shape[0] - gyro_raw_data.shape[0]}")
            acc_raw_data = acc_raw_data[:gyro_raw_data.shape[0], :]

        acc_raw_data = acc_raw_data[:, 0:3]  # in case we had 4th column time
        gyro_raw_data = gyro_raw_data[:, 0:3]  # in case we had 4th column time
        file_features = self._sp.process_raw_data(acc_raw_data, gyro_raw_data)
        file_features = util.normalize(file_features)

        return file_features

    def build_file(self, filename):
        pass

    def build_dir(self):
        """
        path: to raw files
        dst_dir: path to directory where we save processed files
        num_exp: number of experiments or files

        feature is labeled by "time" interval, or "number" interval
        A for activity, PT for posture and ALL for all
        :return:
        """
        logger.info(f"Start Building features from dir: {self.preprocess_dir()} to features_dir: {self._features_dir}")
        x = []
        y = []

        # read labels and set labels for examples
        labels_data = self.read_labels(self._processed_dir + "\\labels.txt")

        # get all files from directory
        file_names = ""
        for f in os.listdir(self._processed_dir):
            if isfile(join(self._processed_dir, f)):
                file_names = file_names + " " + f

        for exp_num in range(1, self._num_exp + 1):
            print("Exp: " + str(exp_num))
            try:
                # acc raw file
                m = re.search('acc_exp' + format(exp_num, '02d'), file_names)
                if m is None:
                    raise FileNotFoundError
                file_path = self._processed_dir + "\\" + file_names[m.start():m.start() + 20]
                acc_raw_data = self.read_ndarray_from_cvs(file_path)
                acc_raw_data = acc_raw_data[:, 0:3]  # in case we had 4th column time

                # gyro raw file
                m = re.search('gyro_exp' + format(exp_num, '02d'), file_names)
                file_path = self._processed_dir + "\\" + file_names[m.start():m.start() + 21]
                gyro_raw_data = self.read_ndarray_from_cvs(file_path)
                gyro_raw_data = gyro_raw_data[:, 0:3]

                # Preprocess step - data must be split into n * window values. Each window has size dt * _window.
                # for ex. dt = 0.02 (50HZ semp. rate), _window=128. Data splitted in 2.56s windows. Fill missing data.
                # calculate features and append all examples
                file_features = self._sp.process_raw_data(acc_raw_data, gyro_raw_data)

                # find labels
                x_file, y_file = self.find_labels(file_features, labels_data,
                                                  self._window, self._overlap, self._freq, exp_num)
                x.append(x_file)
                y.append(y_file)
                print(f"  Found labels for {len(x_file)} Examples")
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
        return x, y

    @staticmethod
    def find_labels(file_features, labels_data, window, overlap, freq, exp_num):
        x = []
        y = []
        dt = 1 / freq

        # if labels are in time else just the number of label:
        slide = dt * (window - overlap)
        # slide = window - overlap

        for f in range(0, file_features.shape[0]):
            p = 0
            for t in labels_data.get(str(exp_num)):
                if float(t[1]) <= (f + 1) * slide and ((f + 1) * slide <= float(t[2])):
                    p = 1
                    break

            if p == 1:
                y.append([int(t[0]), int((f + 1) * slide), int(exp_num)])
                x.append(file_features[f])
            # else:     #add unknown label
            #     y.append([13, int((f + 1) * 64), int(exp_num)])
            #     x.append(file_features[f])
        return x, y

    @staticmethod
    def filter_labels(x, y, mod):
        """
        :param x:
        :param y:
        :param mod: A for activities, PT for Position Transitions
        :return:
        """
        if mod == "A":
            # set all PT to value 7 as we are only interested in activities
            filt = (y >= 7)
            filt[:, 1:3] = False
            y[filt] = 7
            # x, y = filter_lower_then(np.array(x), np.array(y[:, 0]), 7)
        elif mod == 'PT':
            x, y = util.filter_greater_then(np.array(x), np.array(y[:, 0]), 7)

        return x
