import numpy as np
import os
import logging
import util

from data.DataAdapter import DataAdapter
from data.ExtraAdapter import ExtraAdapter, read_label_timestamp_list, labels_to_index, _label_names

logger = logging.getLogger()


class ExtraFusionAdapter(ExtraAdapter):

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        super().__init__(raw_dir)
        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        self._raw_dir = raw_dir
        self._processed_dir = "{0}\\datasets\\Processed\\ExtraFusion".format(parentDir)
        self._features_dir = "{0}\\datasets\\Features\\ExtraFusion".format(parentDir)

        logger.info(f"Created ExtraFusionAdapter raw dir: {self._raw_dir} features_dir: {self._features_dir}")

    def build_dir(self, list_labels_of_interest):
        """ Read raw data, filter labels of interest and extract features.
        Function reads labels_timestamp file, and then goes through labels of interest label per label.
        For each label we process timestamps.
        :param list_labels_of_interest:
        :return: features, labels
        """
        acc_dir = self._raw_dir + "\\6.raw\\raw_acc\\"
        gyro_dir = self._raw_dir + "\\6.raw\\proc_gyro\\"
        acc_watch_dir = self._raw_dir + "\\6.raw\\watch_acc\\"
        gyro_watch_dir = self._raw_dir + "\\6.raw\\watch_compass\\"
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
                acc_watch_file = acc_watch_dir + uuid + "\\" + str(t) + ".m_watch_acc.dat"
                gyro_watch_file = gyro_watch_dir + uuid + "\\" + str(t) + ".m_watch_compass.dat"

                if os.path.isfile(acc_file):
                    mob_acc = np.loadtxt(acc_file)
                else:
                    print("SKIPPED: Acc_mobile File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                if os.path.isfile(gyro_file):
                    mob_gyro = np.loadtxt(gyro_file)
                else:
                    print(
                        "SKIPPED: Gyro_mobile File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                if os.path.isfile(acc_watch_file):
                    watch_acc = np.loadtxt(acc_watch_file)
                else:
                    print(
                        "SKIPPED: Acc_watch File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                if os.path.isfile(gyro_watch_file):
                    watch_gyro = np.loadtxt(gyro_watch_file)
                else:
                    print(
                        "SKIPPED: Gyro_watch File %d for timestamp %d and uuid %s does not exist!" % (count, t, uuid))
                    continue

                # process raw data - extract features and split into windows
                mob_acc = mob_acc[:, 1:4]
                mob_gyro = mob_gyro[:, 1:4]
                watch_acc = watch_acc[:, 1:4]
                watch_gyro = watch_gyro[:, 1:4]

                if mob_acc.shape[0] != mob_gyro.shape[0] or watch_acc.shape[0] != watch_gyro.shape[0]:
                    print("SKIPPED: File %d for timestamp %d and uuid %s has acc-gyro examples mismatch %d %d %d %d" %
                        (count, t, uuid, mob_acc.shape[0], mob_gyro.shape[0],  watch_acc.shape[0], watch_gyro.shape[0]))
                    continue

                file_mob_features = self._sp.process_raw_data(mob_acc, mob_gyro)
                # if file_mob_features.shape[0] != self._feature_number:
                #     print("SKIPPED: File %d for timestamp %d and uuid %s has %d features" %
                #           (count, t, uuid, file_mob_features.shape[0]))
                #     continue
                #

                file_watch_features = self._sp.process_raw_data(mob_acc, mob_gyro)
                # if file_watch_features.shape[0] != self._feature_number:
                #     print("SKIPPED: File %d for timestamp %d and uuid %s has %d features" %
                #           (count, t, uuid, file_watch_features.shape[0]))
                #     continue

                if count % 100 == 0:
                    print(" File %d for timestamp %d and uuid %s has: %s" % (count, t, uuid, file_features.shape))

                # create labels for all examples
                file_labels = np.zeros(file_features.shape[1]) + lab

                # append all examples
                file_features = np.array(file_features).transpose()
                x = np.vstack((x, file_mob_features, file_watch_features))
                y = np.hstack((y, file_labels))

                print(f"Shape of mob x: {file_mob_features.shape} "
                      f"shape of watch x: {file_watch_features.shape} shape of Y: {y.shape}")

        # remove first row made by np.empty
        x = x[1:]
        y = y[1:]
        x = util.normalize(x)

        print(f"Shape of X: {x.shape} Shape of Y: {y.shape}")

        y = util.convert_labels_to_norm(y, list_index)
        np.savetxt(self._features_dir + "\\labels.txt", y, "%d")
        np.savetxt(self._features_dir + "\\features.txt", x, "%.9f")
        return x, y
