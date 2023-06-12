from typing import Union

import hashlib
import json
from pathlib import Path

import checksumdir
import pandas as pd


def dict_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg)

    hash_str = hashlib.sha224(cfg_str.encode("utf-8")).hexdigest()

    return hash_str


def file_hash(fpath: Union[str, Path]) -> str:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    hash_str = hashlib.sha224(fpath.read_bytes()).hexdigest()

    return hash_str


def dir_hash(dpath: str) -> str:
    hash_str = checksumdir.dirhash(dpath)

    return hash_str


def dataframe_hash(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df).sum())
