from typing import Union

import hashlib
import json
from pathlib import Path

import checksumdir
import pandas as pd


def dict_hash(cfg: dict) -> str:
    cfg_str = json.dumps(cfg)
    return hashlib.sha224(cfg_str.encode("utf-8")).hexdigest()


def file_hash(fpath: Union[str, Path]) -> str:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    return hashlib.sha224(fpath.read_bytes()).hexdigest()


def dir_hash(dpath: str) -> str:
    return checksumdir.dirhash(dpath)


def dataframe_hash(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df).sum())
