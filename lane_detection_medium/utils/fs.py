from typing import Union

import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..types.image_types import Image, is_gray


def get_date_string() -> str:
    return date.today().strftime("%Y-%m-%d")


# --- Image related --- #


def read_image(fpath: Union[str, Path], gray_scale: bool = False) -> Image:
    """Read image from filesystem."""

    if gray_scale:
        return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def write_image(fpath: str, img: Image):
    if not is_gray(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(fpath, img)


# --- YAML related --- #


def read_yaml(fpath: str) -> dict:
    with open(fpath) as fp:
        data = yaml.safe_load(fp)
    return data


def save_yaml(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        yaml.safe_dump(data, fp)


# --- JSON related --- #


def read_json(fpath: str) -> dict:
    with open(fpath) as fp:
        data = json.load(fp)
    return data[0]


def save_json(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)


# --- TXT related --- #


def read_txt(fpath: str) -> list[str]:
    with open(fpath) as fp:
        data = fp.readlines()
    return data


# --- YOLO related --- #


def read_yolo_labels(fpath: str) -> np.ndarray:
    txt_data = read_txt(fpath)

    np_data = []
    for line in txt_data:
        np_line = np.fromstring(line, dtype=np.float32, sep=" ")
        np_data.append(np_line)

    np_data = np.array(np_data)

    return np_data


def write_yolo_labels(fpath: str, bboxes):
    with open(fpath, "w") as file:
        file.write("\n".join([" ".join([str(v) for v in bbox]) for bbox in bboxes]))
