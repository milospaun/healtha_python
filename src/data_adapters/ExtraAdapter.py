import shutil
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import gzip
import io
import sklearn.linear_model
import os
import util
from SignalProcessing import SignalProcessing
from data.DataAdapter import DataAdapter

_labels_set1 = ["LYING_DOWN", "SITTING", "STANDING_IN_PLACE", "WALKING", "STAIRS_-_GOING_UP", "STAIRS_-_GOING_DOWN",
                "RUNNING", "BICYCLING", "ELEVATOR", "DRIVE_-_I_M_THE_DRIVER"]
_labels_set2 = ["LIFTING_WEIGHTS", "COOKING", "CLEANING", "VACUUMING", "WASHING_DISHES", "SMOKING", "DANCING",
                "JUMPING", "EATING", "SLEEPING"]
_labels_set3 = ["LYING_DOWN", "SITTING", "STANDING_IN_PLACE", "WALKING", "STAIRS_-_GOING_UP", "STAIRS_-_GOING_DOWN",
                "RUNNING", "BICYCLING", "ELEVATOR", "DRIVE_-_I_M_THE_DRIVER", "LIFTING_WEIGHTS", "COOKING",
                "CLEANING", "VACUUMING", "WASHING_DISHES", "SMOKING", "DANCING", "JUMPING", "EATING", "SLEEPING"]

_label_names = ['LYING_DOWN', 'SITTING', 'STANDING_IN_PLACE', 'STANDING_AND_MOVING', 'WALKING', 'RUNNING',
                'BICYCLING',
                'LIFTING_WEIGHTS', 'PLAYING_BASEBALL', 'PLAYING_BASKETBALL', 'PLAYING_LACROSSE', 'SKATEBOARDING',
                'PLAYING_SOCCER', 'PLAYING_FRISBEE', 'EXERCISING', 'STRETCHING', 'YOGA', 'ELLIPTICAL_MACHINE',
                'TREADMILL',
                'STATIONARY_BIKE', 'COOKING', 'CLEANING', 'GARDENING', 'DOING_LAUNDRY', 'MOWING_THE_LAWN',
                'RAKING_LEAVES',
                'VACUUMING', 'WASHING_DISHES', 'WASHING_CAR', 'MANUAL_LABOR', 'DANCING',
                'LISTENING_TO_MUSIC__WITH_EARPHONES_',
                'LISTENING_TO_MUSIC__NO_EARPHONES_', 'LISTENING_TO_AUDIO__WITH_EARPHONES_',
                'LISTENING_TO_AUDIO__NO_EARPHONES_',
                'PLAYING_MUSICAL_INSTRUMENT', 'SINGING', 'WHISTLING', 'PLAYING_VIDEOGAMES', 'PLAYING_PHONE-GAMES',
                'RELAXING',
                'STROLLING', 'HIKING', 'SHOPPING', 'WATCHING_TV', 'TALKING', 'READING_A_BOOK', 'DRINKING__ALCOHOL_',
                'SMOKING',
                'EATING', 'DRINKING__NON-ALCOHOL_', 'SLEEPING', 'TOILET', 'BATHING_-_BATH', 'BATHING_-_SHOWER',
                'GROOMING',
                'DRESSING', 'STAIRS_-_GOING_UP', 'STAIRS_-_GOING_DOWN', 'LIMPING', 'JUMPING', 'LAUGHING', 'CRYING',
                'USING_CRUTCHES', 'WHEELCHAIR', 'LAB_WORK', 'WRITTEN_WORK', 'DRAWING', 'TEXTING',
                'SURFING_THE_INTERNET',
                'COMPUTER_WORK', 'STUDYING', 'IN_CLASS', 'IN_A_MEETING', 'AT_HOME', 'AT_WORK', 'AT_SCHOOL', 'AT_A_BAR',
                'AT_A_CONCERT', 'AT_A_PARTY', 'AT_A_SPORTS_EVENT', 'AT_THE_BEACH', 'AT_SEA', 'AT_THE_POOL',
                'AT_THE_GYM',
                'AT_A_RESTAURANT', 'OUTSIDE', 'INDOORS', 'ON_A_BUS', 'ON_A_PLANE', 'ON_A_TRAIN', 'ON_A_BOAT',
                'ELEVATOR',
                'MOTORBIKE', 'RIDING_AN_ANIMAL', 'DRIVE_-_I_M_THE_DRIVER', 'DRIVE_-_I_M_A_PASSENGER', 'IN_A_CAR',
                'PHONE_IN_POCKET', 'PHONE_IN_HAND', 'PHONE_IN_BAG', 'PHONE_ON_TABLE', 'PHONE_AWAY_FROM_ME',
                'PHONE_-_SOMEONE_ELSE_USING_IT', 'PHONE_STRAPPED', 'TRANSFER_-_BED_TO_WHEELCHAIR',
                'TRANSFER_-_BED_TO_STAND', 'TRANSFER_-_WHEELCHAIR_TO_BED', 'TRANSFER_-_STAND_TO_BED',
                'ON_A_DATE', 'WITH_CO-WORKERS', 'WITH_FAMILY', 'WITH_FRIENDS', 'WITH_KIDS', 'TAKING_CARE_OF_KIDS',
                'WITH_A_PET']


class ExtraAdapter(DataAdapter):
    _group_size = 15  # size of group in which data is recorded per label. 800 raw measurments produces 15 examples with
    # window 100 and overlap 50
    _freq = 40  # sample frequency
    _window = 100  # size pf the window - umber of samples from which we calculate features
    _overlap = 50  # overlap of the windows
    _tr_test_split_coef = 0.8  # split to train and test files
    _feature_number = 661

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        super().__init__()
        self._raw_dir = raw_dir
        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        self._processed_dir = "{0}\\datasets\\Processed\\ExtraS".format(parentDir)
        self._features_dir = "{0}\\datasets\\Features\\ExtraS".format(parentDir)

        if not os.path.isdir(self._processed_dir):
            os.makedirs(self._processed_dir)

        if not os.path.isdir(self._features_dir):
            os.makedirs(self._features_dir)

        self._sp = SignalProcessing(self._freq, self._window, self._overlap)

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

    def build_dir(self, list_labels_of_interest):
        """ Read raw data, filter labels of interest and extract features.
        Function reads labels_timestamp file, and then goes through labels of interest label per label.
        For each label we process timestamps.
        :param list_labels_of_interest:
        :return: features, labels
        """
        acc_dir = self._raw_dir + "\\6.raw\\raw_acc\\"
        gyro_dir = self._raw_dir + "\\6.raw\\proc_gyro\\"
        labels_dir = self._raw_dir + "\\3.labels\\"

        # get uuids
        uuid_array = []
        for f in os.listdir(labels_dir):
            if os.path.isfile(os.path.join(labels_dir, f)):
                uuid = f[0:36]
                uuid_array.append(uuid)

        x = np.empty((1, self._feature_number))
        y = np.empty(1)

        # load labels
        label_timestamp = read_label_timestamp_list(self._raw_dir + "\\..\\timestamps_labels.txt")
        list_index = labels_to_index(list_labels_of_interest)
        print(list_index)

        # list_index = [92]
        for lab in list_index:
            print(" ##########Processing label %s with index %d ##########" % (_label_names[lab], lab))

            # timestamps = timestamps[0:10]
            tuples = label_timestamp[lab]

            # limit tuples to 800, that is not more then 12000 examples per label
            tuples = tuples[0:800]
            print("Number of timestamps for label: ", str(len(tuples)))

            count = -1
            for (uuid, t) in tuples:
                # (uuid, t) = tuples[123]
                count = count + 1
                acc_file = acc_dir + uuid + "\\" + str(t) + ".m_raw_acc.dat"
                gyro_file = gyro_dir + uuid + "\\" + str(t) + ".m_proc_gyro.dat"
                # raw_magnet = np.loadtxt(acc_dir + uuid + "\\" + t + ".m_raw_magnet.dat")
                # proc_gravity = np.loadtxt(acc_dir + uuid + "\\" + t + ".m_proc_gravity.dat")
                # watch_acc_file = watch_acc_dir + uuid + "\\" + str(t) + ".m_watch_acc.dat"
                # watch_compass_file = watch_compass_dir + uuid + "\\" + str(t) + ".m_watch_compass.dat"

                if os.path.isfile(acc_file):
                    acc = np.loadtxt(acc_file)
                else:
                    print("SKIPPED: Acc_mobile File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                # read raw data
                # print(" File %s" % (acc_dir + uuid + "\\" + str(t) + ".m_raw_acc.dat"))
                if os.path.isfile(gyro_file):
                    proc_gyro = np.loadtxt(gyro_file)
                else:
                    print(
                        "SKIPPED: Gyro_mobile File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                # process raw data - extract features and split into windows
                acc = acc[:, 1:4]
                proc_gyro = proc_gyro[:, 1:4]
                if acc.shape[0] != proc_gyro.shape[0]:
                    print("SKIPPED: File %d for timestamp %d and uuid %s has acc-gyto examples mismatch %d %d" %
                          (count, t, uuid, acc.shape[0], proc_gyro.shape[0]))
                    continue

                try:
                    file_features = self._sp.process_raw_data(acc, proc_gyro)
                    if file_features.shape[0] != self._feature_number:
                        print("SKIPPED: File %d for timestamp %d and uuid %s has %d features" %
                              (count, t, uuid, file_features.shape[0]))
                        continue

                    if count % 100 == 0:
                        print(" File %d for timestamp %d and uuid %s has: %s" % (count, t, uuid, file_features.shape))

                    # create labels for all examples
                    file_labels = np.zeros(file_features.shape[1]) + lab

                    # append all examples
                    file_features = np.array(file_features).transpose()
                    x = np.vstack((x, file_features))
                    y = np.hstack((y, file_labels))
                except:
                    e = sys.exc_info()[0]
                    print("An exception occurred")
                    print(e)

        # remove first row made by np.empty
        x = x[1:]
        y = y[1:]
        x = util.normalize(x)

        print(f"Shape of X: {x.shape} Shape of Y: {y.shape}")

        y = util.convert_labels_to_norm(y, list_index)
        np.savetxt(self._features_dir + "\\labels.txt", y, "%d")
        np.savetxt(self._features_dir + "\\features.txt", x, "%.9f")
        return x, y


def es_main():
    uuid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13'

    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid)
    feat_sensor_names = get_sensor_names_from_features(feature_names)

    sensors_to_use = ['Acc', 'WAcc']
    target_label = 'STAIRS_-_GOING_UP'
    model = train_model(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_label)

    test_model(X, Y, M, timestamps, feat_sensor_names, label_names, model)

    uuid = '11B5EC4D-4133-4289-B475-4E737182A406'
    (X2, Y2, M2, timestamps2, feature_names2, label_names2) = read_user_data(uuid)

    # All the user data files should have the exact same columns. We can validate it:
    validate_column_names_are_consistent(feature_names, feature_names2)
    validate_column_names_are_consistent(label_names, label_names2)
    test_model(X2, Y2, M2, timestamps2, feat_sensor_names, label_names, model)
    return


def show_user_data(uuid='1155FF54-63D3-4AB2-9863-8385D0BD0A13'):
    print("Preprocessing ExtraSensory Dataset: " + uuid)

    (X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid)

    print("The parts of the user's data (and their dimensions):")
    print("Every example has its timestamp, indicating the minute when the example was recorded")
    print("User %s has %d examples (~%d minutes of behavior)" % (uuid, len(timestamps), len(timestamps)))
    print("The primary data files have %d different sensor-features" % len(feature_names))
    print("X is the feature matrix. Each row is an example and each column is a sensor-feature:", X.shape)
    print("The primary data files have %s context-labels" % len(label_names))
    print("Y is the binary label-matrix. Each row represents an example and each column represents a label.")
    print("Value of 1 indicates the label is relevant for the example:", Y.shape)
    print("Y is accompanied by the missing-label-matrix, M.")
    print("Value of 1 indicates that it is best to consider an entry (example-label pair) as 'missing':", M.shape)

    n_examples_per_label = np.sum(Y, axis=0)
    labels_and_counts = zip(label_names, n_examples_per_label)
    sorted_labels_and_counts = sorted(labels_and_counts, reverse=True, key=lambda pair: pair[1])
    print("How many examples does this user have for each context-label:")
    print("-" * 20)
    for (label, count) in sorted_labels_and_counts:
        # print ("label %s - %d minutes" % (label, count))
        print("%s - %d minutes" % (get_label_pretty_name(label), count))
        pass

    print("Since the collection of labels relied on self-reporting in-the-wild, the labeling may be incomplete.")
    print("For instance, the users did not always report the position of the phone.")
    fig = plt.figure(figsize=(15, 5), facecolor='white')

    ax1 = plt.subplot(1, 2, 1)
    labels_to_display = ['LYING_DOWN', 'SITTING', 'OR_standing', 'FIX_walking', 'FIX_running']
    figure__pie_chart(Y, label_names, labels_to_display, 'Body state', ax1)

    ax2 = plt.subplot(1, 2, 2)
    labels_to_display = ['PHONE_IN_HAND', 'PHONE_IN_BAG', 'PHONE_IN_POCKET', 'PHONE_ON_TABLE']
    figure__pie_chart(Y, label_names, labels_to_display, 'Phone position', ax2)
    plt.show()

    feat_sensor_names = get_sensor_names_from_features(feature_names)

    for (fi, feature) in enumerate(feature_names):
        print("%3d) %s %s" % (fi, feat_sensor_names[fi].ljust(10), feature))
        pass

    feature_inds = [0, 102, 133, 148, 157, 158]
    figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds)
    plt.show()

    print("The phone-state (PS) features are represented as binary indicators:")
    feature_inds = [205, 223]
    figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds)
    plt.show()

    feature1 = 'proc_gyro:magnitude_stats:time_entropy'  # raw_acc:magnitude_autocorrelation:period';
    feature2 = 'raw_acc:3d:mean_y'
    label2color_map = {'PHONE_IN_HAND': 'b', 'PHONE_ON_TABLE': 'g'}
    figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map)
    plt.show()


def read_label_timestamp_list(src_dir):
    label_timestamp = [[] for _ in range(len(_label_names))]
    f = open(src_dir, 'r')
    count = -1
    for line in f:
        count = count + 1
        line = "*" + line
        tuple_list = line.split(')')
        tuple_list.pop()
        for tup in tuple_list:
            uuid = tup[4:40]
            timestamp = int(tup[42:55])
            label_timestamp[count].append((uuid, timestamp))
    f.close()
    # for i in range(len(label_timestamp)):
    #     print("%s:  %d" % (_label_names[i], len(label_timestamp[i])))

    return label_timestamp


def filter_timestamps_per_labels(src_dir, dst_dir):
    count = 0
    labels_dir = src_dir + "Raw\\labels\\"

    # get uuids
    uuid_array = []
    for f in os.listdir(labels_dir):
        if os.path.isfile(os.path.join(labels_dir, f)):
            uuid = f[0:36]
            uuid_array.append(uuid)

    # read labels for user with uuid
    uuid = "00EABED2-271D-49D8-B599-1D4A09240601",

    timestamps_list = [[] for _ in range(len(_label_names))]
    dict_labels_timestamps = dict()
    for uuid in uuid_array:
        print(" ##########Processing user with uuid %s##########" % (uuid))
        timestamps: object
        timestamps, label_names, all_labels, label_source = read_label_file(labels_dir, uuid)
        # labels is num_timestamps * num_labels
        tmp = np.where(all_labels == 1)
        x_ind = tmp[0]
        y_ind = tmp[1]

        for (x, y) in list(zip(x_ind, y_ind)):
            timestamps_list[y].append((uuid, timestamps[x]))

        print("User uuid: %s has labels %s" % (uuid, str(util.count_occurance(y_ind))))

    for i in range(len(timestamps_list)):
        print("%s:  %d" % (_label_names[i], len(timestamps_list[i])))

    timestamps_list = np.array(timestamps_list)
    np.savetxt(dst_dir + "timestamps_labels.txt", timestamps_list, "%s")


def read_raw_data(uuid):
    """ Read the raw senzor data for user. Function reads all files for user with uuid
    and joins them in one file.
    :param uuid:
    :return: data
    """
    src_dir = _dir_raw + 'Raw\\raw_acc\\' + uuid + "\\"
    print(src_dir)

    count = 0
    data = []
    for filename in os.listdir(src_dir):
        # print("\n Processing file: " + filename)
        tmp = np.loadtxt(src_dir + filename)
        data.append(tmp)
        count = count + 1

    data = np.vstack(data)
    print("Total files processed: " + str(count))
    print("DAta shape: ", data.shape)
    return data


def read_label_feature_file(uuid):
    """ Read labels data for a user
    :param uuid:
    :return:
    """
    user_data_file = _dir_raw + 'original\\1.labels_features\\%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass;

    csv_str = csv_str.decode("utf-8")
    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return X, Y, M, timestamps, feature_names, label_names


def read_label_file(dirpath, uuid):
    """ Read labels data in csv format for a user
    :param dirpath: directory of the labels
    :param uuid:
    :return:timestamps - list of timestamps
    :return:dict_label_names - dictionary of label names. key:label name, value:int
    :return:labels - matrix of labels.
    :return:label_source -
    """
    label_file = dirpath + '%s.original_labels.csv' % uuid
    file = open(label_file, "r")
    csv_str = file.read()

    # csv_str = csv_str.decode("utf-8")
    # (feature_names, label_names) = parse_header_of_csv(csv_str)
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')
    n_labels = len(columns)
    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';

    # Then come the labels, till the one-before-last column:
    label_names = columns[1:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('original_label:')
        label_names[li] = label.replace('original_label:', '')
        pass

    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(io.StringIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps: object = full_table[:, 0].astype(int)

    # Read the labels:
    labels = full_table[:, 1:(n_labels - 2)]
    label_source = full_table[:, n_labels - 1]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_labels + 1):-1]  # This should have values of either 0., 1. or NaN
    # M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    # Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    # create dictinary for label names
    return timestamps, label_names, labels, label_source


def read_user_data(uuid):
    """ Read the data (precomputed sensor-features and labels) for a user.
    This function assumes the user's data file is present.
    :param uuid:
    :return:
    """
    user_data_file = _dir_raw + 'original\\1.labels_features\\%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass;

    csv_str = csv_str.decode("utf-8")
    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return X, Y, M, timestamps, feature_names, label_names


def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:', '')
        pass;

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(io.StringIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return X, Y, M, timestamps


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking'
    if label == 'FIX_running':
        return 'Running'
    if label == 'LOC_main_workplace':
        return 'At main workplace'
    if label == 'OR_indoors':
        return 'Indoors'
    if label == 'OR_outside':
        return 'Outside'
    if label == 'LOC_home':
        return 'At home'
    if label == 'FIX_restaurant':
        return 'At a restaurant'
    if label == 'OR_exercise':
        return 'Exercise'
    if label == 'LOC_beach':
        return 'At the beach'
    if label == 'OR_standing':
        return 'Standing'
    if label == 'WATCHING_TV':
        return 'Watching TV'

    if label.endswith('_'):
        label = label[:-1] + ')'
        pass

    label = label.replace('__', ' (').replace('_', ' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m', 'I\'m')
    return label


def figure__pie_chart(Y, label_names, labels_to_display, title_str, ax):
    portion_of_time = np.mean(Y, axis=0)
    portions_to_display = [portion_of_time[label_names.index(label)] for label in labels_to_display]
    pretty_labels_to_display = [get_label_pretty_name(label) for label in labels_to_display]

    ax.pie(portions_to_display, labels=pretty_labels_to_display, autopct='%.2f%%')
    ax.axis('equal')
    plt.title(title_str)
    return


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi, feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass;

    return feat_sensor_names


def figure__feature_track_and_hist(X, feature_names, timestamps, feature_inds):
    seconds_in_day = (60 * 60 * 24)
    days_since_participation = (timestamps - timestamps[0]) / float(seconds_in_day)

    for ind in feature_inds:
        feature = feature_names[ind]
        feat_values = X[:, ind]

        fig = plt.figure(figsize=(10, 3), facecolor='white')

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(days_since_participation, feat_values, '.-', markersize=3, linewidth=0.1)
        plt.xlabel('days of participation')
        plt.ylabel('feature value')
        plt.title('%d) %s\nfunction of time' % (ind, feature))

        ax1 = plt.subplot(1, 2, 2)
        existing_feature = np.logical_not(np.isnan(feat_values))
        ax1.hist(feat_values[existing_feature], bins=30)
        plt.xlabel('feature value')
        plt.ylabel('count')
        plt.title('%d) %s\nhistogram' % (ind, feature))
        pass;
    return


def figure__feature_scatter_for_labels(X, Y, feature_names, label_names, feature1, feature2, label2color_map):
    feat_ind1 = feature_names.index(feature1)
    feat_ind2 = feature_names.index(feature2)
    example_has_feature1 = np.logical_not(np.isnan(X[:, feat_ind1]))
    example_has_feature2 = np.logical_not(np.isnan(X[:, feat_ind2]))
    example_has_features12 = np.logical_and(example_has_feature1, example_has_feature2)

    fig = plt.figure(figsize=(12, 5), facecolor='white')
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 4)

    for label in label2color_map.keys():
        label_ind = label_names.index(label)
        pretty_name = get_label_pretty_name(label)
        color = label2color_map[label]
        style = '.%s' % color

        is_relevant_example = np.logical_and(example_has_features12, Y[:, label_ind])
        count = sum(is_relevant_example)
        feat1_vals = X[is_relevant_example, feat_ind1]
        feat2_vals = X[is_relevant_example, feat_ind2]
        ax1.plot(feat1_vals, feat2_vals, style, markersize=5, label=pretty_name)

        ax2.hist(X[is_relevant_example, feat_ind1], bins=20, density=True, color=color, alpha=0.5,
                 label='%s (%d)' % (pretty_name, count))
        ax3.hist(X[is_relevant_example, feat_ind2], bins=20, density=True, color=color, alpha=0.5,
                 label='%s (%d)' % (pretty_name, count))
        pass

    ax1.set_xlabel(feature1)
    ax1.set_ylabel(feature2)
    ax2.set_title(feature1)
    ax3.set_title(feature2)
    ax2.legend(loc='best')

    return


def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature, is_from_sensor)
        pass
    X = X[:, use_feature]
    return X


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0)
    std_vec = np.nanstd(X_train, axis=0)
    return (mean_vec, std_vec)


def standardize_features(X, mean_vec, std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1, -1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1))
    X_standard = X_centralized / normalizers
    return X_standard


def train_model(X_train, Y_train, M_train, feat_sensor_names, label_names, sensors_to_use, target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train, feat_sensor_names, sensors_to_use)
    print("== Projected the features to %d features from the sensors: %s" % (
        X_train.shape[1], ', '.join(sensors_to_use)))

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train, mean_vec, std_vec)

    # The single target label:
    label_ind = label_names.index(target_label)
    y = Y_train[:, label_ind]
    missing_label = M_train[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label, :]
    y = y[existing_label]

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.

    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))))

    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.

    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    lr_model.fit(X_train, y)

    # Assemble all the parts of the model:
    model = { \
        'sensors_to_use': sensors_to_use, \
        'target_label': target_label, \
        'mean_vec': mean_vec, \
        'std_vec': std_vec, \
        'lr_model': lr_model};

    return model


def test_model(X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use'])
    print("== Projected the features to %d features from the sensors: %s" % (
        X_test.shape[1], ', '.join(model['sensors_to_use'])))

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec'])

    # The single target label:
    label_ind = label_names.index(model['target_label'])
    y = Y_test[:, label_ind]
    missing_label = M_test[:, label_ind]
    existing_label = np.logical_not(missing_label)

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label, :]
    y = y[existing_label]
    timestamps = timestamps[existing_label]

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(model['target_label']), sum(y), sum(np.logical_not(y))))

    # Preform the prediction:
    y_pred = model['lr_model'].predict(X_test)

    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y)

    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred, y))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y))

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)

    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.

    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp + fp)

    print("-" * 10)
    print('Accuracy*:         %.2f' % accuracy)
    print('Sensitivity (TPR): %.2f' % sensitivity)
    print('Specificity (TNR): %.2f' % specificity)
    print('Balanced accuracy: %.2f' % balanced_accuracy)
    print('Precision**:       %.2f' % precision)
    print("-" * 10)

    print(
        '* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print(
        '** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')

    fig = plt.figure(figsize=(10, 4), facecolor='white')
    ax = plt.subplot(1, 1, 1)
    ax.plot(timestamps[y], 1.4 * np.ones(sum(y)), '|g', markersize=10, label='ground truth')
    ax.plot(timestamps[y_pred], np.ones(sum(y_pred)), '|b', markersize=10, label='prediction')

    seconds_in_day = (60 * 60 * 24)
    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day)
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)

    ax.set_ylim([0.5, 5])
    ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels)
    plt.xlabel('days of participation', fontsize=14)
    ax.legend(loc='best')
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']))

    return


def validate_column_names_are_consistent(old_column_names, new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")

    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError(
                "!!! Inconsistent column %d) %s != %s" % (ci, old_column_names[ci], new_column_names[ci]))
        pass
    return


def labels_to_index(labels_list):
    dict_lab = dict(zip(_label_names, range(len(_label_names))))
    list_index = [dict_lab[x] for x in labels_list]
    return list_index


def index_to_label(index_list):
    list_labels = [_label_names[x] for x in index_list]
    return list_labels
