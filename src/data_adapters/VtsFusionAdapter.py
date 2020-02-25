import numpy as np
import os
import logging

import pandas as pd

from src.data_adapters.DataAdapter import DataAdapter
from src.data_adapters.VtsAdapterMobile import VtsAdapterMobile
from src.data_adapters.VtsAdapterWatch import VtsAdapterWatch

logger = logging.getLogger()


class VtsFusionAdapter(DataAdapter):

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
