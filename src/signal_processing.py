import numpy as np
import src.util as util
import scipy.stats
from statsmodels import robust
import math
import os
import re
from os import listdir
from os.path import isfile, join
from statsmodels.tsa.ar_model import AR
from scipy.signal import butter, lfilter, freqz
import shutil

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


def set_global_vars(freq, window, overlap, split_coef):
    global _freq
    _freq = freq
    global _window
    _window = window
    global _overlap
    _overlap = overlap
    global _tr_test_split_coef
    _tr_test_split_coef = split_coef


def process_raw_data(acc_raw_data, gyro_raw_data):
    dt = 1 / _freq
    # return 272 features (error returns 468 features)
    acc_features, acc, grav = process_raw_acc_data(acc_raw_data)

    # return 165 gyro features (error returns 267)
    gyro_features, gyro = process_raw_gyro_data(gyro_raw_data)
    # print(acc_features.shape)
    # print(gyro_features.shape)
    features = np.vstack((acc_features, gyro_features))

    # calculate data for angle
    acc_mean = split_3d_with_mean(acc)
    grav_mean = split_3d_with_mean(grav)
    gyro_mean = split_3d_with_mean(gyro)

    j1 = np.insert(acc, acc.shape[0], np.zeros(3), 0)
    j2 = np.insert(acc, 0, np.zeros(3), 0)
    body_accJerk = np.delete((j2 - j1) / dt, -1, axis=0)
    accJerk_mean = split_3d_with_mean(body_accJerk)

    j2 = np.insert(gyro, gyro.shape[0], np.zeros(3), 0)
    j1 = np.insert(gyro, 0, np.zeros(3), 0)
    gyro_jerk = np.delete((j2 - j1) / dt, -1, axis=0)
    gyro_jerk_mean = split_3d_with_mean(gyro_jerk)

    # add 7 angle features
    ang_acc_gravity = util.angle_between_md(acc_mean, grav_mean)
    ang_acc_jerk_gravity = util.angle_between_md(accJerk_mean, grav_mean)
    ang_gyro_gravity = util.angle_between_md(gyro_mean, grav_mean)
    ang_gyro_jerk_gravity = util.angle_between_md(gyro_jerk_mean, grav_mean)

    acc_x = np.vstack((acc_mean[:, 0], np.zeros(acc_mean.shape[0]), np.zeros(acc_mean.shape[0]))).T
    acc_y = np.vstack((acc_mean[:, 1], np.zeros(acc_mean.shape[0]), np.zeros(acc_mean.shape[0]))).T
    acc_z = np.vstack((acc_mean[:, 2], np.zeros(acc_mean.shape[0]), np.zeros(acc_mean.shape[0]))).T
    ang_x_acc_gravity = util.angle_between_md(acc_x, grav_mean)
    ang_y_acc_gravity = util.angle_between_md(acc_y, grav_mean)
    ang__zacc_gravity = util.angle_between_md(acc_z, grav_mean)

    ang_features = [ang_acc_gravity, ang_acc_jerk_gravity, ang_gyro_gravity, ang_gyro_jerk_gravity, ang_x_acc_gravity,
                    ang_y_acc_gravity, ang__zacc_gravity]

    ang_features = np.vstack(ang_features)
    features = np.vstack((features, ang_features))

    # total 742
    return features


def process_1d_signal(x):
    """ Process one axes of 3d acc or gyro signal
    8 featrues in total
    :param x:
    :return:
    """
    rez = []
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    x_mad = robust.mad(x, axis=1)
    x_max = np.max(x, axis=1)
    x_min = np.min(x, axis=1)
    x_energy = np.sum(x ** 2, axis=1) / len(x)
    x_iqr = scipy.stats.iqr(x, axis=1)
    # x_entropy = scipy.stats.entropy(x.T)

    rez.append(x_mean)
    rez.append(x_std)
    rez.append(x_mad)
    rez.append(x_max)
    rez.append(x_min)
    rez.append(x_energy)
    rez.append(x_iqr)

    # produces nan when doing normalization, max and min are the same
    # rez.append(x_entropy)
    return rez


def process_fft_1d_signal(x):
    """ Calculate 12 features
    :param x: 1D measured raw data
    :return:
    """
    rez = []
    x = split_data(x, _window, _overlap)
    x = np.fft.fft(x)
    freq = np.fft.fftfreq(_window, d=0.02)
    x = np.abs(x) ** 2

    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    x_mad = robust.mad(x, axis=1)
    x_max = np.max(x, axis=1)
    x_min = np.min(x, axis=1)
    x_energy = np.sum(x ** 2, axis=1) / len(x)
    x_iqr = scipy.stats.iqr(x, axis=1)
    x_max_inds = np.nanargmax(x, axis=1)
    x_mean_freq = np.sum(x * freq / len(freq), axis=1)
    x_skew = scipy.stats.skew(x, axis=1)
    x_kurt = scipy.stats.kurtosis(x, axis=1)

    rez.append(x_mean)
    rez.append(x_std)
    rez.append(x_mad)
    rez.append(x_max)
    rez.append(x_min)
    rez.append(x_energy)
    rez.append(x_iqr)
    rez.append(x_max_inds)
    rez.append(x_mean_freq)
    rez.append(x_skew)
    rez.append(x_kurt)

    # it is -inf so it brekas everything
    # x_entropy = scipy.stats.entropy(x.T)
    # rez.append(x_entropy)

    return rez


def process_fft_3D_signal(data):
    """ Calculate 36 + 1 + 42 features = 73
    :param data:
    :return:
    """
    a1, a2, a3 = split_3d_data(data)
    features = np.array(np.ones(int(data.shape[0] / (_window - _overlap)) - 1))

    for a in [a1, a2, a3]:
        x = np.fft.fft(a)
        # x_freq = np.fft.fftfreq(_window, d=1 / _freq)
        x = np.abs(x) ** 2

        # 12 * 3 = 36 features
        features = np.vstack((features, process_fft_1d_signal(x)))
        # features = np.vstack((features, np.mean(x, axis=1)))
        # features = np.vstack((features, np.std(x, axis=1)))
        # features = np.vstack((features, robust.mad(x, axis=1)))
        # features = np.vstack((features, np.max(x, axis=1)))
        # features = np.vstack((features, np.min(x, axis=1)))
        # # to do entropy
        # features = np.vstack((features, scipy.stats.entropy(x.T)))
        # features = np.vstack((features, np.sum(x ** 2, axis=1) / (x.shape[1])))
        # features = np.vstack((features, scipy.stats.iqr(x, axis=1)))
        #
        # # maxInd, meanFreq, skew, kurtosis
        # features = np.vstack((features, np.nanargmax(x, axis=1)))
        # features = np.vstack((features, np.sum(x * x_freq / len(x_freq), axis=1)))
        # features = np.vstack((features, scipy.stats.skew(x, axis=1)))
        # features = np.vstack((features, scipy.stats.kurtosis(x, axis=1)))

    # Calculate SMA
    a1 = split_data(a1, _window, _overlap)
    a2 = split_data(a2, _window, _overlap)
    a3 = split_data(a3, _window, _overlap)
    x_sma = np.sum(a1 + a2 + a3, axis=1) / len(data) * _freq
    features = np.vstack((features, x_sma))

    # Energy bands w/3 features in total. For w = 128 we get 42 features    #Energy 42 bands
    band_num = int(_window / 3) - 1
    for x in (a1, a2, a3):
        f_eb = []
        for i in range(0, band_num):
            f = (x[:, 3 * i] + x[:, 3 * i + 1] + x[:, 3 * i + 2]) / 3
            f_eb.append(f)

        f_eb = np.array(f_eb).reshape(band_num, a1.shape[0])
        features = np.vstack((features, f_eb))

    return features[1:]


def process_3d_signal(data):
    """ Calculate 8*4+3+1+12 = 40 features
    :param data:
    :return:
    """
    a1, a2, a3 = split_3d_data(data)

    # a1 = util.normalize(a1)
    # a2 = util.normalize(a1)
    # a3 = util.normalize(a1)
    x1 = split_data(a1, _window, _overlap)
    x2 = split_data(a2, _window, _overlap)
    x3 = split_data(a3, _window, _overlap)

    # 8x3 = 24 in total
    rez1 = process_1d_signal(x1)
    rez2 = process_1d_signal(x2)
    rez3 = process_1d_signal(x3)

    # for all 3 axis - 3 features for corelation, 1 for sma and 12 for AR
    x_correlation12 = []
    x_correlation13 = []
    x_correlation23 = []
    x_ar = []
    for i in range(0, x1.shape[0]):
        x_correlation12.append(np.correlate(x1[i], x2[i]))
        x_correlation13.append(np.correlate(x1[i], x3[i]))
        x_correlation23.append(np.correlate(x2[i], x3[i]))

        ar_coef = []
        ar_mod = AR(x1[i])
        ar_coef.extend(ar_mod.fit(3).params)
        ar_mod = AR(x2[i])
        ar_coef.extend(ar_mod.fit(3).params)
        ar_mod = AR(x3[i])
        ar_coef.extend(ar_mod.fit(3).params)

        x_ar.append(ar_coef)

    x_ar = np.vstack(x_ar)
    x_correlation12 = np.vstack(x_correlation12)
    x_correlation13 = np.vstack(x_correlation13)
    x_correlation23 = np.vstack(x_correlation23)
    x_sma = np.sum(x1 + x2 + x3, axis=1) / len(data) * _freq

    rez = np.vstack((rez1, rez2))
    rez = np.vstack((rez, rez3))
    rez = np.vstack((rez, x_ar.T))
    rez = np.vstack((rez, x_correlation12.T))
    rez = np.vstack((rez, x_correlation13.T))
    rez = np.vstack((rez, x_correlation23.T))
    rez = np.vstack((rez, x_sma))
    return rez


# features 8 + sma + 4 (AR) = 13
def process_mag_signal(data):
    data = split_data(data, _window, _overlap)
    rez = process_1d_signal(data)
    x_sma = np.sum(data) / len(data) * _freq
    np.insert(rez, 5, x_sma)

    # insert AR 4 coef
    x_ar = []
    for i in range(0, data.shape[0]):
        ar_mod = AR(data[i])
        x_ar.append(ar_mod.fit(3).params)

    np.append(rez, x_ar)
    return rez


def process_raw_acc_data(data):
    """ Calculate 274 features
    :param data: num.examples * nuum_col. Num_cal is 3 axes and optionally number of seconds
    :return:
    """
    dt = 1 / _freq

    # remove noise
    body_acc = util.butter_lowpass(data, 20, 50, 3)

    # split into gravity and body signal with lowpass filter
    grav_acc = util.butter_lowpass(body_acc, 0.3, 50, 3)
    body_acc = body_acc - grav_acc

    # print(data.shape)
    j1 = np.insert(body_acc, body_acc.shape[0], np.zeros(3), 0)
    j2 = np.insert(body_acc, 0, np.zeros(3), 0)
    body_acc_jerk = np.delete((j2 - j1) / dt, -1, axis=0)
    body_acc_mag = np.sqrt(np.sum(body_acc ** 2, axis=1))
    grav_acc_mag = np.sqrt(np.sum(grav_acc ** 2, axis=1))
    body_acc_jerk_mag = np.sqrt(np.sum(body_acc_jerk ** 2, axis=1))

    # shuld be 40 features each but is 40 * 3 = 120
    features = process_3d_signal(body_acc)
    features = np.vstack((features, process_3d_signal(grav_acc)))
    features = np.vstack((features, process_3d_signal(body_acc_jerk)))

    # should be 13 features but is 9 * 3 = 33
    features = np.vstack((features, process_mag_signal(body_acc_mag)))
    features = np.vstack((features, process_mag_signal(grav_acc_mag)))
    features = np.vstack((features, process_mag_signal(body_acc_jerk_mag)))

    # FFT 71 + 71 + 12 + 12 = 166
    features = np.vstack((features, process_fft_3D_signal(body_acc)))
    features = np.vstack((features, process_fft_3D_signal(body_acc_jerk)))
    features = np.vstack((features, process_fft_1d_signal(body_acc_mag)))
    features = np.vstack((features, process_fft_1d_signal(body_acc_jerk_mag)))

    #    print("Acc features extracted" + str(features.shape))
    return features, body_acc, grav_acc


def process_raw_gyro_data(data):
    """ Calculate 160 features
    :param data:
    :return:
    """
    dt = 1 / _freq

    filt_gyro = util.butter_lowpass(data, 20, 50, 3)
    j2 = np.insert(filt_gyro, filt_gyro.shape[0], np.zeros(3), 0)
    j1 = np.insert(filt_gyro, 0, np.zeros(3), 0)
    gyro_jerk = np.delete((j2 - j1) / dt, -1, axis=0)
    gyro_mag = np.sqrt(np.sum(filt_gyro ** 2, axis=1))
    gyro_jerk_mag = np.sqrt(np.sum(gyro_jerk ** 2, axis=1))

    # 24 features x 2 = 48
    features = process_3d_signal(filt_gyro)
    features = np.vstack((features, process_3d_signal(gyro_jerk)))

    # 8 features x 2 = 16
    features = np.vstack((features, process_mag_signal(gyro_mag)))
    features = np.vstack((features, process_mag_signal(gyro_jerk_mag)))

    # FFT 71 + 12 + 12 = 96
    features = np.vstack((features, process_fft_3D_signal(filt_gyro)))
    features = np.vstack((features, process_fft_1d_signal(gyro_mag)))
    features = np.vstack((features, process_fft_1d_signal(gyro_jerk_mag)))

    #    print("Gyro features extracted" + str(features.shape))
    #    np.savetxt("..\\Tmp\\features.txt", features.T)
    return features, filt_gyro


def process_raw_compas_data(data):
    """ Calculate 160 features
    :param data:
    :return:
    """
    # TO DO
    dt = 1 / _freq
    filt_comp = util.butter_lowpass(data, 20, 50, 1)

    # 8 features
    features = process_mag_signal(filt_comp)

    # FFT 12
    features = np.vstack((features, process_fft_1d_signal(filt_comp)))

    return features, filt_comp


def split_3d_data(data):
    a = data[:, 0]
    b = data[:, 1]
    c = data[:, 2]
    return a, b, c


def split_3d_with_mean(data):
    """
    Splits 3D data into windows and calculates mean 3 axis
    :param data:
    :return:
    """
    x, y, z = split_3d_data(data)
    x = split_data(x, _window, _overlap)
    x = np.mean(x, axis=1)
    y = split_data(y, _window, _overlap)
    y = np.mean(y, axis=1)
    z = split_data(z, _window, _overlap)
    z = np.mean(z, axis=1)

    rez = np.vstack((x, y, z))
    return rez.T


def pre_process_data(data, freq, window, overlap):
    """
    Corrects measured data by adding or removing samples so that each window will have exactly window samples.
    It duplicats las sample in the window, or removes last sample from the window.
    :param data:
    :param freq:
    :param window:
    :param overlap:
    :return:
    """
    dt = 1 / freq
    t = dt * (window - overlap)
    n = window - overlap

    m = data.shape[0]
    # window_num = math.floor(m / n)
    start = data[0][3]
    end = start + t
    window_num = int(math.floor((data[-1][3] - start) / t))

    x = []
    for i in range(window_num):
        x.append([])

    nwin = 0
    for i in range(0, m):
        if start <= data[i][3] <= end and nwin < window_num:
            x[nwin].append(data[i])
        else:
            nwin = nwin + 1
            start = start + t
            end = end + t
            if start <= data[i][3] <= end and nwin < window_num:
                x[nwin].append(data[i])  # add to next window

    num_add = 0
    num_removied = 0
    print("Number of windows: ", len(x))
    for i in range(len(x)):
        while len(x[i]) < n:
            if i == 228:
                print("STOP")

            x[i].append(x[i][-1])
            num_add = num_add + 1
            print("added to window: " + str(i))

        while len(x[i]) > n:
            x[i].pop()
            num_removied = num_removied + 1
            print("removed from window: " + str(i))

    print("Total added and removed: " + str(num_add) + "   " + str(num_removied))
    print("Num. Windows: " + str(window_num) + " Window: " + str(window) + " Overlap: " + str(overlap))
    x = np.array(x)
    data = np.array(data)
    # print(x.shape)
    # x = x.flatten()
    # print(x.shape)
    x = np.reshape(x, [int(x.size / 4), 4])
    print("Data: " + str(data.shape))
    print("Result: " + str(x.shape))
    return x


def split_data(data, w, o):
    """ Split data to windows. Splits one axis.
    :param data: numpy array
    :param w: widow size
    :param o: window distance used for overlap. If o>=w there is no overlap
    :return: splitted np array of dim n x w
    """
    n = int(data.shape[0] / (w - o))
    n = n - 1  # last windows does not have all 128 values so i discard it
    rez = np.array([])
    for i in range(0, n):
        a = data[i * o:i * o + w]
        rez = np.append(rez, a)

    return rez.reshape(n, w)
