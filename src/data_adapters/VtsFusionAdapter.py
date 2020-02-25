import numpy as np
import os
import logging

import pandas as pd

from src.data_adapters.DataAdapter import DataAdapter
from src.data_adapters.VtsAdapterMobile import VtsAdapterMobile
from src.data_adapters.VtsAdapterWatch import VtsAdapterWatch

logger = logging.getLogger()


class VtsFusionAdapter(DataAdapter):
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

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        _raw_dir = raw_dir
        _processed_dir = "{0}\\datasets\\Processed\\VtsFusion".format(parentDir)
        _features_dir = "{0}\\datasets\\Features\\VtsFusion".format(parentDir)

        logger.info(f"Created VtsFusionAdapter raw dir: {self._raw_dir} features_dir: {self._features_dir}")

    def build_dir(self):
        adapter_mob = VtsAdapterMobile(self._raw_dir + "VtsApps\\2.RAW\\Mobile")
        adapter_mob.preprocess_dir()
        x_mob, y_mob = adapter_mob.build_dir()

        adapter_watch = VtsAdapterWatch(self._raw_dir + "VtsApps\\2.RAW\\Watch")
        adapter_watch.preprocess_dir()
        x_watch, y_watch = adapter_watch.build_dir()

        x_mob, y_mob, x_watch, y_watch = pd.DataFrame(x_mob), pd.DataFrame(y_mob), pd.DataFrame(x_watch), pd.DataFrame(
            y_watch)
        data, labels = self.__build_fused_features(x_mob, y_mob, x_watch, y_watch)

        return data, labels

    def __build_fused_features(self, x_mob, y_mob, x_watch, y_watch):
        """
        Takes watch and mobile features and combines them into one file
        :param x_mob: DataFrame
        :param y_mob:  DataFrame
        :param x_watch: DataFrame
        :param y_watch: DataFrame
        :return: DataFrame, Series
        """
        y_mob.columns = ['M_Label', 'M_Time', 'M_Experiment', ]
        y_watch.columns = ['W_Label', 'L_Time', 'W_Experiment']

        mob = pd.concat([x_mob, y_mob], axis=1)
        watch = pd.concat([x_watch, y_watch], axis=1)

        data = pd.merge(mob, watch, left_on=['M_Experiment', 'M_Time'], right_on=['W_Experiment', 'L_Time'])

        labels = data.pop('M_Label')
        data.drop(['M_Experiment', 'M_Time', 'W_Experiment', 'W_Label', 'L_Time'], axis=1)

        fusion_dir = self._features_dir + "\\..\\VtsFusion\\"
        if not os.path.isdir(fusion_dir):
            os.makedirs(fusion_dir)

        np.savetxt(fusion_dir + "vts_x.txt", data, "%.9f")
        np.savetxt(fusion_dir + "vts_y.txt", labels, "%.2f")
        print(f"Num examples before mob:{x_mob.shape} watch:{x_watch.shape}, and after processing: {data.shape}")

        return data, labels

    def build_data(self, acc_mob, gyro_mob, acc_watch, gyro_watch):
        """

        :param acc_mob:
        :param gyro_mob:
        :param acc_watch:
        :param gyro_watch:
        :return:
        """
        adapter_mob = VtsAdapterMobile("")
        features_mob = adapter_mob.build_data(acc_mob, gyro_mob)

        adapter_watch = VtsAdapterWatch("")
        features_watch = adapter_watch.build_data(acc_watch, gyro_watch)
        print(f"Num examples before mob:{features_mob.shape} watch:{features_watch.shape}")

        # check if they are the same size and correct
        if features_mob.shape[0] > features_watch.shape[0]:
            print(f"Gyro file has more samples, deleting {features_mob.shape[0] - features_watch.shape[0]}")
            features_mob = features_mob[:features_watch.shape[0], :]

        elif features_watch.shape[0] > features_mob.shape[0]:
            print(f"Acc file has more samples, deleting {features_watch.shape[0] - features_mob.shape[0]}")
            features_watch = features_watch[:features_mob.shape[0], :]

        features = np.hstack((features_mob, features_watch))

        return features
