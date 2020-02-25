import numpy as np
import util as util
import os
import re
from os import listdir
from os.path import isfile, join
import shutil
import signal_processing as sp

###
# The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in
# fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has
# gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and
# gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff
# frequency was used. From each window, a vector of 561 features was obtained by calculating variables from the time and
# frequency domain. See 'features_info.txt' for more details.
# From 2000 measures you get 31 labels

# accc time data were processed with median filter and a 3rd order low pass Butterworth filter with a
# corner frequency of 20 Hz to remove noise
# One gotcha is that Wn is a fraction of the Nyquist frequency (half the sampling frequency). So if the sampling rate is
# 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5.
# acc_exp01_user01.txt
# gyro_exp38_user19.txt
###

_freq = 50  # sample frequency
_window = 128  # size pf the window - umber of samples from which we calculate features
_overlap = 11  # overlap of the windows
_tr_test_split_coef = 0.8  # split to train and test files


def preprocess_directory(srcdir, dstdir, freq, window, overlap):
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)

    count = 0
    for filename in os.listdir(srcdir):
        print("\n Processing file: " + filename)
        m = re.search('acc_', filename)
        n = re.search('gyro_', filename)
        if m is not None or n is not None:
            data = read_raw_data(srcdir + "\\" + filename)
            x = sp.pre_process_4D_data(data, freq, window, overlap)
            np.savetxt(dstdir + "\\" + filename, x, "%.9f")
            count = count + 1

    if os.path.isfile(srcdir + "\\" + "labels.txt"):
        shutil.copyfile(srcdir + "\\" + "labels.txt", dstdir + "\\" + "labels.txt")

    print("Total files processed: " + str(count))


def process_directory(path, dst_dir, freq, window, overlap, labels="time", mod="ALL", num_exp=61):
    """
    :param path: to raw files
    :param dst_dir: path to directory where we save processed files
    :param freq:
    :param window:
    :param overlap:
    :param labels: feature is labeled by "time" interval, or "number" interval
    :param mod: A for activity, PT for posture and ALL for all
    :param num_exp: number of experiments or files
    :return:
    """

    # read label data
    y = []
    x = []
    features = []

    # set labels for examples
    file_labels = path + "\\labels.txt"
    labels_data = read_labels(file_labels)

    # get all files from directory
    file_names = ""
    for f in listdir(path):
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
            acc_raw_data = read_raw_data(file_path)
            acc_raw_data = acc_raw_data[:, 0:3]  # in case we had 4th column time

            # gyro raw file
            m = re.search('gyro_exp' + format(exp_num, '02d'), file_names)
            file_path = path + "\\" + file_names[m.start():m.start() + 21]
            gyro_raw_data = read_raw_data(file_path)
            gyro_raw_data = gyro_raw_data[:, 0:3]

            # normalize - should i normilize raw input? No
            # acc_raw_data = util.normalize(acc_raw_data)
            # gyro_raw_data = util.normalize(gyro_raw_data)

            # Preprocessing step - data must be split into n * window values. Each window to be of size dt * _window.
            # for example dt = 0.02 (50HZ semp. rate), _window=128. Data splited in 2.56s windows. Fill mising data.

            # calculate features
            file_features = sp.process_raw_data(acc_raw_data, gyro_raw_data)

            # append all examples
            features.append(file_features)
            file_features = file_features.T

            # find labels
            dt = 1 / _freq
            if labels == "time":
                slide = dt * (_window - _overlap)
            else:
                slide = _window - _overlap

            for f in range(0, file_features.shape[0]):
                p = 0
                for t in labels_data[exp_num]:
                    if float(t[1]) <= (f + 1) * slide and ((f + 1) * slide <= float(t[2])):
                        p = 1
                        break

                if p == 1:
                    y.append([int(t[0]), int((f + 1) * slide), int(exp_num)])
                    x.append(file_features[f])
                # else:     #add unknown label
                #     y.append([13, int((f + 1) * 64), int(exp_num)])
                #     x.append(file_features[f])

        except FileNotFoundError:
            print("Oops!  File with number" + str(exp_num) + " was not processed. File path: " + file_path)
            continue

    features = np.hstack(features)
    features = features.T
    x = np.vstack(x)

    print("Shape of samples array after : " + str(features.shape))
    print("Shape of samples X after labeling: " + str(x.shape))
    y = np.array(y)

    x = util.normalize(x)
    np.savetxt("..\\data\\mk_y_all.txt", y, "%d")
    np.savetxt("..\\data\\vts_x_mob.txt", x, "%.9f")
    np.savetxt("..\\data\\features.txt", features)

    if mod == "A":
        # set all PT to value 7 as we are only interested in activities
        filt = y >= 7
        filt[:, 1:3] = False
        y[filt] = 7
        # x, y = filter_lower_then(np.array(x), np.array(y[:, 0]), 7)
    elif mod == "PT":
        x, y = sp.filter_greater_then(np.array(x), np.array(y[:, 0]), 7)

    tmp = int(y.shape[0] * _tr_test_split_coef)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
        os.mkdir(os.path.join(dst_dir, "Train"))
        os.mkdir(os.path.join(dst_dir, "Test"))

    np.savetxt(dst_dir + "\\Train\Y_train.txt", y[0:tmp], "%d")
    np.savetxt(dst_dir + "\\Train\X_train.txt", x[0:tmp], "%.9f")
    np.savetxt(dst_dir + "\\Test\Y_test.txt", y[tmp + 1:], "%d")
    np.savetxt(dst_dir + "\\Test\X_test.txt", x[tmp + 1:], "%.9f")
    return x


# read labels.txt file
def read_labels(file):
    file = open(file, "r")

    nl = 0
    data = []
    tuplist = []
    exp_num = 1
    for line in file:
        nl = nl + 1
        tmp = line.split()
        tup = (tmp[2], tmp[3], tmp[4])
        if exp_num == tmp[0]:
            tuplist.append(tup)
        else:
            exp_num = tmp[0]
            data.append(tuplist)
            tuplist = [tup]
    data.append(tuplist)

    # RESHAPE n is num of features, m is num of examples
    data = np.reshape(data, (nl, 3))
    file.close()
    return data


def read_raw_data(file):
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


def read_raw_labels(file):
    file = open(file, 'r')

    nl = 0
    data = []
    for line in file:
        nl = nl + 1
        tmp = line.split()
        for x in tmp:
            data.append(np.int(x))

    data = np.reshape(data, (nl, 5))
    file.close()
    return data
