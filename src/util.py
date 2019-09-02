import datetime

import numpy as np
import math
import tensorflow as tf
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import os

_filepath_log = "D:\Projekti\Git\ActivityDetection\Tmp\Report.txt"


def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_md(v1, v2):
    v = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]
    v = v.flatten()
    v = np.clip(v, -1.0, 1.0)
    v = np.arccos(v)
    return v.T


def plot_2(x1, x2, fs=50, t=410, nsamples=20598):
    t = np.linspace(0, t, nsamples, endpoint=False)
    plt.plot(t, x1, label='Noisy signal')
    # plt.plot(t, acc1x_body, label='Filtered body')
    plt.plot(t, x2, "C2", label='Filtered gravity')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.xlabel('time (seconds)')
    plt.show()


def plot_3D(x, y, z, fs=50, nsamples=20598):
    t = np.linspace(0, nsamples, nsamples, endpoint=False)
    plt.legend(loc='upper left')
    plt.subplot(311)
    plt.plot(t, x, label='X')

    plt.subplot(312)
    plt.plot(t, y, label='Y', color='r')

    plt.subplot(313)
    plt.plot(t, z, label='Z', color='g')

    plt.grid(True)
    plt.axis('tight')
    plt.xlabel('time (seconds)')
    plt.show()


def normalize(x):
    new_x = (x - np.average(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    a = np.where(np.isnan(new_x), x, new_x)
    return a


def butter_lowpass(data, cutoff, fs, order):
    # cutOff = 20 hz  # cutoff frequency in rad/s
    # fs = 50 hz  # sampling frequency in rad/s
    # order = 3  # order of filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b, a, data)
    return y


def running_mean(data, w, o):
    """
    :param data: numpy array
    :param w: widow size
    :param o: window distance used for overlap. If o>=w there is no overlap
    :return: filtered numpy array
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    c = (cumsum[w:] - cumsum[:-w]) / w
    c = c[::o]
    return c


def shuffle(x, y):
    """
    :param x: 2D numpy array
    :param y: 1D numpy array, one hot representation of y
    :return:
    """
    m = x.shape[0]
    permutation = list(np.random.permutation(m))
    y = np.reshape(y, (-1, 1))
    # print(x.shape)
    # print(y.shape)

    shuffled_X = x[permutation, :]
    shuffled_Y = y[permutation, :]

    # print(shuffled_X.shape)
    # print(shuffled_Y.shape)
    return shuffled_X, shuffled_Y


def shuffle_with_onehot(x, y):
    """
    :param x: 2D numpy array
    :param y: 2D numpy array, one hot representation of y
    :return:
    """
    m = x.shape[0]
    permutation = list(np.random.permutation(m))
    # print(permutation)
    # print(m)
    # print(x.shape)
    # print(y.shape)

    shuffled_X = x[permutation, :]
    shuffled_Y = y[permutation, :]

    # print(shuffled_X.shape)
    # print(shuffled_Y.shape)
    return shuffled_X, shuffled_Y


def shuffle_group(x, y, group_size):
    """
    Shuffles dataset, but keeps num_examples together in a group to preserve seqence
    :param x:
    :param y:
    :param group_size: how many examples should be in a group
    :return:
    """
    # print(x.shape)
    # print(y.shape)
    m = x.shape[0]
    f_num = x.shape[1]
    row_num = m / group_size
    x = np.reshape(x, (-1, group_size, f_num))
    y = np.reshape(y, (-1, group_size))

    permutation = list(np.random.permutation(int(row_num)))
    # print(x.shape)
    # print(y.shape)

    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]
    shuffled_x = np.array(shuffled_x).reshape(m, f_num)
    shuffled_y = np.array(shuffled_y).reshape(m, )
    # print(shuffled_x.shape)
    # print(shuffled_y.shape)
    return shuffled_x, shuffled_y


def shuffle_test_train(train_x, train_y, test_x, test_y, split_coef=0.8):
    x = np.vstack((test_x, train_x))
    y = np.vstack((test_y, train_y))

    x, y = shuffle(x, y)

    tmp = int(y.shape[0] * split_coef)
    train_x = x[0:tmp]
    train_y = y[0:tmp]
    test_x = x[tmp + 1:]
    test_y = y[tmp + 1:]

    return train_x, train_y, test_x, test_y,


def random_mini_batches(X, Y, mini_batch_size=64, group_size=1):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y) but in a groups
    shuffled_X, shuffled_Y = shuffle_with_onehot(X, Y)
    # shuffled_X, shuffled_Y = shuffle_group(X, Y, group_size)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_onehot(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1. Index begins from 0 in labels
   """

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def convert_from_comma_to_space(load_path, filename):
    """
    Converts comma separated file to space separated file
   """
    file = open(load_path + "\\" + filename, "r")
    parent_dir = os.path.abspath(os.path.join(load_path, os.pardir))
    save_path = os.path.join(parent_dir, "formated")

    nl = 0
    data = []
    for line in file:
        nl = nl + 1
        tmp = line.split(',')
        for x in tmp:
            data.append(np.float32(x))

    # RESHAPE n is num of features, m is num of examples
    data = np.reshape(data, (nl, 3))

    np.savetxt(save_path + "\\" + filename, data, fmt="%.9f")
    file.close()
    return data


def conf_matrix_stas(y, ycapa):
    """
    :param y: labels
    :param ycapa: predicions
    :return: matrix
    """
    mat = None
    acc = None
    return mat, acc


# print metrics
def print_metrics(labels, predictions):
    acc, acc_op = tf.metrics.accuracy(labels, predictions)
    pr, pr_op = tf.metrics.precision(labels, predictions)
    rec, rec_op = tf.metrics.recall(labels, predictions)
    #    mean, mean_op = tf.metrics.mean_iou(labels, predictions, 7)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    print("TF Acc: ", sess.run([acc, acc_op]))
    print("TF Precission", sess.run([pr, pr_op]))
    print("TF Recall:", sess.run([rec, rec_op]))
    #    print(sess.run([mean, mean_op]))

    # confusion matrix
    conf_mat = tf.confusion_matrix(labels, predictions)
    print(sess.run(conf_mat))
    sess.close();

    # # Confusion matrix old
    # print("Test Confusion matrix:")
    # print(tf.contrib.metrics.confusion_matrix(Z3capa_test, Ycapa).eval())
    # print_metrics_per_class(labels, predictions, filepath_report)
    return 0


def print_metrics_per_class(labels, predictions):
    """
    :param labels:
    :param predictions:
    :return:
    """
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    acc, acc_op = tf.metrics.accuracy(labels, predictions)
    pr, pr_op = tf.metrics.precision(labels, predictions)
    rec, rec_op = tf.metrics.recall(labels, predictions)
    # tf.metrics.sensitivity_at_specificity(labels, predictions)
    #    mean, mean_op = tf.metrics.mean_iou(labels, predictions, 7)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    acc, acc_op = sess.run([acc, acc_op])
    pr, pr_op = sess.run([pr, pr_op])
    rec, rec_op = sess.run([rec, rec_op])

    # TensorFlow calculates accuracy as number of true positives / num_labels
    print("TF Acc Total: ", acc, acc_op)
    print("TF Precission Total", pr, pr_op)
    print("TF Recall Total:", rec, rec_op)

    # confusion matrix
    conf_mat = tf.confusion_matrix(labels, predictions)
    a_conf_mat = sess.run(conf_mat)
    sess.close()

    print(a_conf_mat)
    lab_counts = np.sum(a_conf_mat, axis=0)
    print("Lab_Count: ", lab_counts)
    pred_counts = np.sum(a_conf_mat, axis=1)
    print("Prediction_Count: ", pred_counts)
    total_count = np.sum(a_conf_mat)

    true_positives = np.diagonal(a_conf_mat)
    true_negatives = total_count - pred_counts - lab_counts + true_positives
    false_negatives = lab_counts - true_positives
    false_positives = pred_counts - true_positives
    print("TP: ", true_positives)
    print("TN: ", true_negatives)
    print("FN: ", false_negatives)
    print("FP: ", false_positives)

    acc_k = true_positives / lab_counts
    prec_k = true_positives / (true_positives + false_positives)
    recall_k = true_positives / (true_positives + false_negatives)
    f1 = 2 * prec_k * recall_k / (prec_k + recall_k)
    spec = true_negatives / (total_count - lab_counts)
    print("Acc: ", np.around(acc_k, 2))
    print("Prec: ", np.around(prec_k, 2))
    print("Rec: ", np.around(recall_k, 2))
    print("F1: ", np.around(f1, 2))
    print("Spec: ", np.around(spec, 2))

    # save to file
    if _filepath_log is not None:
        file = open(_filepath_log, 'a')

        file.write("Lab_counts:       " + str(lab_counts.sum()) + "  " + str(count_occurance(labels)) + "\n")
        file.write("Prediction_counts: " + str(pred_counts.sum()) + "  " + str(count_occurance(predictions)) + "\n")
        file.write("Total Stats ACC:" + str(acc_op) + "    PREC: " + str(pr) + "    REC: " + str(rec))
        file.write("\nAcc: " + str(np.around(acc_k, 2)))
        file.write("\nPrec: " + str(np.around(prec_k, 2)))
        file.write("\nRec: " + str(np.around(recall_k, 2)))
        file.write("\nF1: " + str(np.around(f1, 2)))
        file.write("\nSpec: " + str(np.around(spec, 2)))
        file.write("\nConfusion matrix: \n")
        file.write(str(a_conf_mat))
        file.close()

    return a_conf_mat


def count_occurance(np_list):
    """
    Counts occurance of each elelment in numpy array and retruns it as dictionary
    :param np_list:
    :return: dictionary (element: numbee_of_occurance)
    """
    unique, counts = np.unique(np_list, return_counts=True)
    return dict(zip(unique, counts))


def filter_lower_then(x, y, num):
    """
    :param x: numpy array
    :param y:  numpy array
    :param num:  integer value
    :return: numpy array, numpy array
    """
    filt = y < num
    fy = y[filt]
    fy = fy.astype(int)

    y[filt] = y
    # y[np.logical_not(filt)] = 7

    f = filt.flatten()
    fx = x[f]

    return fx, fy


def convert_labels_to_norm(y, labels_of_interest):
    """
    Converts original labels to the range[0, num_labels]
    :param labels_of_interest: original index of labels
    :param y:
    :return:
    """

    # change labels to be from 0 to labels_num
    label_num = len(labels_of_interest)

    # change labels to range [0, num_lab]
    dict_lab = dict(zip(labels_of_interest, range(label_num)))
    y = [dict_lab[i] for i in y.flatten().tolist()]
    y = np.array(y).reshape(len(y), 1)

    return y


def filter_greater_then(x, y, num):
    """
    :param x: numpy array
    :param y:  numpy array
    :param num:  integer value
    :return: numpy array, numpy array
    """
    filt = y > num
    fy = y[filt]
    fy = fy.astype(int)

    f = filt.flatten()
    fx = x[f]

    return fx, fy


def filter_labels(x, y, labels_list):
    """
    :param x: numpy array
    :param y:  numpy array
    :param labels_list:  list of labels of interest
    :return: numpy array, numpy array
    """
    filt = np.isin(y, labels_list)
    fy = y[filt]
    fy = fy.astype(int)

    f = filt.flatten()
    fx = x[f]

    return fx, fy


def filter_shuffle_n_examples(x, y, num, shuffle=False):
    """
    Takes all examples. Shuffles them and then select max num per each class
    :param shuffle: true for shuffle, false if there is no shuffling
    :param x: numpy array
    :param y:  numpy array
    :param num:  number of examples per class
    :return: numpy array, numpy array
    """
    y = np.array(y)
    num_class = len(np.unique(y))

    # must reshape y so that I can stack it to x
    y = y.reshape((len(y), 1))
    xy = np.hstack((y, x))

    # shuffle examples
    if shuffle:
        np.random.shuffle(xy)

    fxy = []
    for i in range(num_class):
        # select all examples with class i
        filt = y == i
        filt = filt.flatten()
        pom = xy[filt]

        # slelect only first num examples after shuffeling
        pom = pom[0:num]

        # add to return result
        fxy.append(pom)

    fxy = np.vstack(fxy)
    fx = fxy[:, 1:]
    fy = fxy[:, 0]
    fy = fy.reshape((len(fy), 1))
    return fx, fy


def log(msg):
    if _filepath_log is not None:
        file = open(_filepath_log, 'a')

        file.write(msg + "\n")
        file.close()

    print(msg)


def log_table_value(msg):
    filepath_table = "D:\Projekti\Git\ActivityDetection\Tmp\Report_epoch.txt"
    if filepath_table is not None:
        file = open(filepath_table, 'a')

        file.write(msg + "\n")
        file.close()


def split_train_test(x, y, train_coef=0.8):
    # save and print results, split into test and train sets
    tmp = int(y.shape[0] * train_coef)

    x_train = x[0:tmp]
    y_train = y[0:tmp]
    x_test = x[tmp:]
    y_test = y[tmp:]

    return x_train, y_train, x_test, y_test


def split_train_test_dev(x, y, train_coef=0.8):
    # save and print results, split into test and train sets
    # m = 100, train_coef = 0.8, index_test = 80, index_dev = 80 + (100 - 80)/2 = 90
    index_test = int(y.shape[0] * train_coef)
    index_dev = index_test + int((y.shape[0] - index_test) / 2)

    x_train = x[0:index_test]
    y_train = y[0:index_test]
    x_test = x[index_test:index_dev]
    y_test = y[index_test:index_dev]
    x_dev = x[index_dev:]
    y_dev = y[index_dev:]

    return x_train, y_train, x_test, y_test, x_dev, y_dev
