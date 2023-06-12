from typing import Callable

import logging
import os
from enum import Enum, auto

import pandas as pd

from ..utils.hash import dir_hash

read_function = Callable[[str], pd.DataFrame]


class DatasetMode(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class FileDataset:
    """Dataset class for file control functions."""

    supported_extensions = ["csv", "xlsx"]

    def __init__(self, dpath: str, file_extension: str = "csv"):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(dpath):
            raise ValueError("Dataset is not initialized - create it first!")

        if file_extension not in self.supported_extensions:
            raise ValueError(f"'{file_extension}' is not supported.")

        self._dpath = dpath
        self._file_extension = file_extension

        self._logger.info(f"Dataset directory: {self._dpath}")
        self._logger.info(f"Computed dataset hash: {self.hash}")

        self._train_fpath = os.path.join(self._dpath, f"train.{self._file_extension}")
        self._logger.info(f"Train data path: {self._train_fpath}")

        self._valid_fpath = os.path.join(self._dpath, f"val.{self._file_extension}")
        self._logger.info(f"Val data path: {self._valid_fpath}")

        self._test_fpath = os.path.join(self._dpath, f"test.{self._file_extension}")
        self._logger.info(f"Test data path: {self._test_fpath}")

    @property
    def hash(self) -> str:
        return dir_hash(self._dpath)

    def _read_file(self, fpath: str) -> pd.DataFrame:
        if fpath.lower().endswith(".csv"):
            read_func: read_function = pd.read_csv
        elif fpath.lower().endswith(".xlsx"):
            read_func: read_function = pd.read_excel

        return read_func(fpath, index_col=0)

    def get_data(self, mode: DatasetMode) -> pd.DataFrame:
        if mode.value == DatasetMode.TRAIN.value:
            fpath = self._train_fpath
        elif mode.value == DatasetMode.VALID.value:
            fpath = self._valid_fpath
        elif mode.value == DatasetMode.TEST.value:
            fpath = self._test_fpath
        else:
            raise ValueError(f"Invalid mode value: {mode}")

        df = self._read_file(fpath)
        self._logger.info(f"{mode.value} Data Shape: {df.shape}")

        return df
