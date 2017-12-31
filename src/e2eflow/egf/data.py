import os
import sys
import re

import numpy as np
from PIL import Image

from ..core.data import Data
from ..util import tryremove
from shutil import copyfile, rmtree
from urllib.request import urlretrieve


class EGFData(Data):

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)


    def get_raw_files(self):
        return self.data_dir

    def _fetch_if_missing(self):
        """A call to this must make subsequent calls to get_raw_files succeed.
        All subdirs of data_dir listed in self.dirs must exist after this call.
        """
        pass

    def _download_and_extract(self, url, extract_to, ext='zip'):
        raise NotImplementedError()

    def compute_statistics(self, files):
        raise NotImplementedError()
