import numpy as np
import src.util as util
import scipy.stats
from statsmodels import robust
import math
from statsmodels.tsa.ar_model import AR

from scipy.signal import butter, lfilter, freqz


# ## The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled
# in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal,
# which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body
# acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a
# filter with 0.3 Hz cutoff frequency was used. From each window, a vector of 561 features was obtained by
# calculating variables from the time and frequency domain. See 'features_info.txt' for more details. From 2000
# measures you get 31 labels

# accc time data were processed with median filter and a 3rd order low pass Butterworth filter with a
# corner frequency of 20 Hz to remove noise
# One gotcha is that Wn is a fraction of the Nyquist frequency (half the sampling frequency). So if the sampling rate is
# 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5.
# acc_exp01_user01.txt
# gyro_exp38_user19.txt
###

class SignalProcessing:
    _freq = 50  # sample frequency
    _window = 128  # size pf the window - umber of samples from which we calculate features
    _overlap = 11  # overlap of the windows
    _dt = 0.02
    _tr_test_split_coef = 0.8  # split to train and test files

    def __init__(self, freq, window, overlap):
        self._freq = freq
        self._window = window
        self._overlap = overlap
        self._dt = 1 / self._freq
        print(f"Signal Processing intilized to freq: {freq} window: {window} overlap: {overlap}")

    # noinspection DuplicatedCode
    def process_raw_data(self, acc_raw_data, gyro_raw_data):
        """

        :param acc_raw_data:
        :param gyro_raw_data:
        :return: windows x num_features
        """
        # return 272 features (error returns 234 features)
        acc_features, acc, grav = self.process_raw_acc_data(acc_raw_data)

        # return 165 gyro features (error returns 150)
        gyro_features, gyro = self.process_raw_gyro_data(gyro_raw_data)
        features = np.vstack((acc_features, gyro_features))

        # calculate data for angle
        acc_mean = self.split_3d_with_mean(acc)
        grav_mean = self.split_3d_with_mean(grav)
        gyro_mean = self.split_3d_with_mean(gyro)

        j1 = np.insert(acc, acc.shape[0], np.zeros(3), 0)
        j2 = np.insert(acc, 0, np.zeros(3), 0)
        body_accJerk = np.delete((j2 - j1) / self._dt, -1, axis=0)
        accJerk_mean = self.split_3d_with_mean(body_accJerk)

        j2 = np.insert(gyro, gyro.shape[0], np.zeros(3), 0)
        j1 = np.insert(gyro, 0, np.zeros(3), 0)
        gyro_jerk = np.delete((j2 - j1) / self._dt, -1, axis=0)
        gyro_jerk_mean = self.split_3d_with_mean(gyro_jerk)

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

        ang_features = [ang_acc_gravity, ang_acc_jerk_gravity, ang_gyro_gravity, ang_gyro_jerk_gravity,
                        ang_x_acc_gravity,
                        ang_y_acc_gravity, ang__zacc_gravity]

        ang_features = np.vstack(ang_features)
        features = np.vstack((features, ang_features))
        print(f"Acc: {acc_features.shape} Gyro: {gyro_features.shape}  Ang: {ang_features.shape}")
        features = features.T
        return features

    # noinspection DuplicatedCode
    def process_1d_signal(self, x):
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

    def process_fft_1d_signal(self, x):
        """ Calculate 12 features
        :param x: 1D measured raw data
        :return:
        """
        rez = []
        x = self.split_data_to_windows(x)
        x = np.fft.fft(x)
        freq = np.fft.fftfreq(self._window, d=0.02)
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

        # it is -inf so it breaks everything
        # x_entropy = scipy.stats.entropy(x.T)
        # rez.append(x_entropy)

        rez = np.array(rez)
        return rez

    def process_fft_3D_signal(self, data):
        """ Calculate 36 + 1 + 42 features = 73
        :param data:
        :return:
        """
        a1, a2, a3 = self.split_3d_data(data)
        features = np.array(np.ones(int(data.shape[0] / (self._window - self._overlap)) - 1))

        # 11 * 3 = 33 features
        for a in [a1, a2, a3]:
            features = np.vstack((features, self.process_fft_1d_signal(a)))

        x1 = self.split_data_to_windows(a1)
        x2 = self.split_data_to_windows(a2)
        x3 = self.split_data_to_windows(a3)
        x1 = np.abs(np.fft.fft(x1)) ** 2
        x2 = np.abs(np.fft.fft(x2)) ** 2
        x3 = np.abs(np.fft.fft(x3)) ** 2

        # Calculate SMA
        x_sma = np.sum(x1 + x2 + x3, axis=1) / len(data) * self._freq
        features = np.vstack((features, x_sma))

        for x in [x1, x2, x3]:
            # Energy bands w/3 features in total. For w = 128 we get 42 features    #Energy 42 bands
            # for i in range(0, 41):
            #     f = (x[:, 3 * i] + x[:, 3 * i + 1] + x[:, 3 * i + 2]) / 3
            #     features = np.vstack((features, f))
            band_num = int(self._window / 3) - 1
            f_eb = []
            for i in range(0, band_num):
                f = (x[:, 3 * i] + x[:, 3 * i + 1] + x[:, 3 * i + 2]) / 3
                f_eb.append(f)

            f_eb = np.array(f_eb).reshape(band_num, x1.shape[0])
            features = np.vstack((features, f_eb))

        return features[1:]

    def process_3d_signal(self, data):
        """ Calculate 8*4+3+1+12 = 40 features
        :param data:
        :return:
        """
        a1, a2, a3 = self.split_3d_data(data)
        # a1 = util.normalize(a1)
        # a2 = util.normalize(a1)
        # a3 = util.normalize(a1)

        x1 = self.split_data_to_windows(a1)
        x2 = self.split_data_to_windows(a2)
        x3 = self.split_data_to_windows(a3)

        # 8x3 = 24 in total
        rez1 = self.process_1d_signal(x1)
        rez2 = self.process_1d_signal(x2)
        rez3 = self.process_1d_signal(x3)

        # for all 3 axis - 3 features for correlation, 1 for sma and 12 for AR
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
        x_sma = np.sum(x1 + x2 + x3, axis=1) / len(data) * self._freq

        rez = np.vstack((rez1, rez2))
        rez = np.vstack((rez, rez3))
        rez = np.vstack((rez, x_ar.T))
        rez = np.vstack((rez, x_correlation12.T))
        rez = np.vstack((rez, x_correlation13.T))
        rez = np.vstack((rez, x_correlation23.T))
        rez = np.vstack((rez, x_sma))
        return rez

    # features 8 + sma + 4 (AR) = 13
    def process_mag_signal(self, data):
        data = self.split_data_to_windows(data)
        rez = self.process_1d_signal(data)
        x_sma = np.sum(data) / len(data) * self._freq
        np.insert(rez, 5, x_sma)

        # insert AR 4 coef
        x_ar = []
        for i in range(0, data.shape[0]):
            ar_mod = AR(data[i])
            x_ar.append(ar_mod.fit(3).params)

        np.append(rez, x_ar)
        return rez

    # noinspection DuplicatedCode
    def process_raw_acc_data(self, data):
        """ Calculate 274 features
        :param data: num.examples * nuum_col. Num_cal is 3 axes and optionally number of seconds
        :return:
        """
        # remove noise
        body_acc = util.butter_lowpass(data, 20, 50, 3)

        # split into gravity and body signal with lowpass filter
        grav_acc = util.butter_lowpass(body_acc, 0.3, 50, 3)
        body_acc = body_acc - grav_acc

        # print(data.shape)
        j1 = np.insert(body_acc, body_acc.shape[0], np.zeros(3), 0)
        j2 = np.insert(body_acc, 0, np.zeros(3), 0)
        body_acc_jerk = np.delete((j2 - j1) / self._dt, -1, axis=0)
        body_acc_mag = np.sqrt(np.sum(body_acc ** 2, axis=1))
        grav_acc_mag = np.sqrt(np.sum(grav_acc ** 2, axis=1))
        body_acc_jerk_mag = np.sqrt(np.sum(body_acc_jerk ** 2, axis=1))

        # should be 40 features each but is 40 * 3 = 120
        features = self.process_3d_signal(body_acc)
        features = np.vstack((features, self.process_3d_signal(grav_acc)))
        features = np.vstack((features, self.process_3d_signal(body_acc_jerk)))

        # should be 13 features but is 9 * 3 = 33
        features = np.vstack((features, self.process_mag_signal(body_acc_mag)))
        features = np.vstack((features, self.process_mag_signal(grav_acc_mag)))
        features = np.vstack((features, self.process_mag_signal(body_acc_jerk_mag)))

        # FFT 71 + 71 + 12 + 12 = 166
        features = np.vstack((features, self.process_fft_3D_signal(body_acc)))
        features = np.vstack((features, self.process_fft_3D_signal(body_acc_jerk)))

        features = np.vstack((features, self.process_fft_1d_signal(body_acc_mag)))
        features = np.vstack((features, self.process_fft_1d_signal(body_acc_jerk_mag)))

        #    print("Acc features extracted" + str(features.shape))
        return features, body_acc, grav_acc

    # noinspection DuplicatedCode
    def process_raw_gyro_data(self, data):
        """ Calculate 160 features
        :param data:
        :return:
        """
        filt_gyro = util.butter_lowpass(data, 20, 50, 3)
        j2 = np.insert(filt_gyro, filt_gyro.shape[0], np.zeros(3), 0)
        j1 = np.insert(filt_gyro, 0, np.zeros(3), 0)
        gyro_jerk = np.delete((j2 - j1) / self._dt, -1, axis=0)
        gyro_mag = np.sqrt(np.sum(filt_gyro ** 2, axis=1))
        gyro_jerk_mag = np.sqrt(np.sum(gyro_jerk ** 2, axis=1))

        # 24 features x 2 = 48
        features = self.process_3d_signal(filt_gyro)
        features = np.vstack((features, self.process_3d_signal(gyro_jerk)))

        # 8 features x 2 = 16
        features = np.vstack((features, self.process_mag_signal(gyro_mag)))
        features = np.vstack((features, self.process_mag_signal(gyro_jerk_mag)))

        # FFT 71 + 12 + 12 = 96
        features = np.vstack((features, self.process_fft_3D_signal(filt_gyro)))
        features = np.vstack((features, self.process_fft_1d_signal(gyro_mag)))
        features = np.vstack((features, self.process_fft_1d_signal(gyro_jerk_mag)))

        #    print("Gyro features extracted" + str(features.shape))
        #    np.savetxt("..\\Tmp\\features.txt", features.T)
        return features, filt_gyro

    def process_raw_compas_data(self, data):
        """ Calculate 160 features
        :param data:
        :return:
        """
        filt_comp = util.butter_lowpass(data, 20, 50, 1)

        # 8 features
        features = self.process_mag_signal(filt_comp)

        # FFT 12
        features = np.vstack((features, self.process_fft_1d_signal(filt_comp)))

        return features, filt_comp

    def split_3d_data(self, data):
        a = data[:, 0]
        b = data[:, 1]
        c = data[:, 2]
        return a, b, c

    def split_3d_with_mean(self, data):
        """
        Splits 3D data into windows and calculates mean 3 axis
        :param data:
        :return:
        """
        x, y, z = self.split_3d_data(data)
        x = self.split_data_to_windows(x)
        x = np.mean(x, axis=1)
        y = self.split_data_to_windows(y)
        y = np.mean(y, axis=1)
        z = self.split_data_to_windows(z)
        z = np.mean(z, axis=1)

        rez = np.vstack((x, y, z))
        return rez.T

    def pre_process_4d_data(self, data):
        """
        Corrects measured data by adding or removing samples so that each window will have exactly window samples.
        It duplicats las sample in the window, or removes last sample from the window.
        :param data: [XAxis:float, YAxis:float, ZAxis:float, Time:float]
        :return: 2d array - [float, float, float, float]
        """

        t = self._dt * (self._window - self._overlap)
        n = self._window - self._overlap

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
                # if i == 228:
                #     print("STOP")

                x[i].append(x[i][-1])
                num_add = num_add + 1
                # print("added to window: " + str(i))

            while len(x[i]) > n:
                x[i].pop()
                num_removied = num_removied + 1
                # print("removed from window: " + str(i))

        x = np.array(x)
        data = np.array(data)
        x = np.reshape(x, [int(x.size / 4), 4])
        print(f"Total added and removed: {num_add} {num_removied} Data: {data.shape} Result: {x.shape} \n")
        return x

    def split_data_to_windows(self, data):
        """ Split data to windows. Splits one axis.
        w: widow size
        o: window distance used for overlap. If o>=w there is no overlap
        :param data: numpy array
        :return: splitted np array of dim n x w
        """
        w = self._window
        o = self._overlap
        n = int(data.shape[0] / (w - o))
        n = n - 1  # last windows does not have all 128 values so i discard it
        rez = np.array([])
        for i in range(0, n):
            a = data[i * o:i * o + w]
            rez = np.append(rez, a)

        return rez.reshape(n, w)
