import os
import logging
from src.data_adapters.VtsAdapterMobile import VtsAdapterMobile
logger = logging.getLogger()


class VtsAdapterWatch(VtsAdapterMobile):

    # Initializer / Instance attributes
    def __init__(self, raw_dir):
        super().__init__(raw_dir)

        cwd = os.getcwd()
        parentDir = os.path.abspath(os.path.join(cwd, os.pardir))
        processed_dir = "{0}\\datasets\\Processed\\VtsWatch".format(parentDir)
        features_dir = "{0}\\datasets\\Features\\VtsWatch".format(parentDir)
        super().set_parameters(raw_dir, processed_dir, features_dir,
                               freq=5, window=10, overlap=5, num_exp=15)
        logger.info(f"Created VtsWatchAdapter with freq: {self._freq} window: {self._window} "
                    f"overlap: {self._overlap} raw dir: {self._raw_dir} features_dir: {self._features_dir}")



