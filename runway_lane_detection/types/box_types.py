# flake8: noqa
# TODO: refactor for nasty linter

from typing import Optional, Union

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class BoxFormat(Enum):
    xywh = auto()
    xyxy = auto()


@dataclass
class Bbox:
    coords: Union[list[float], np.ndarray[float, np.dtype[np.float32]]]
    dformat: BoxFormat = BoxFormat.xywh

    def __post_init__(self):
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords)

        self.coords = self.coords.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        return self._as_dformat(BoxFormat.xywh)

    @property
    def xyxy(self) -> np.ndarray:
        return self._as_dformat(BoxFormat.xyxy)

    @property
    def center(self) -> tuple[float, float]:
        w, h = self.xywh[2:]
        return (w / 2, h / 2)

    @property
    def area(self) -> float:
        w, h = self.xywh[2:]
        return w * h

    @property
    def perimeter(self) -> float:
        w, h = self.xywh[2:]
        return 2 * (w + h)

    @property
    def points(self) -> np.ndarray:
        # NOTE: from left to right
        pt0, pt2 = self.xyxy.reshape(-1, 2)
        pt1 = [pt2[0], pt0[1]]
        pt3 = [pt0[0], pt2[1]]
        return np.array([pt0, pt1, pt2, pt3])

    @classmethod
    def from_xyxy(cls, coords: Union[list[float], np.ndarray]):
        return cls(coords=coords)

    @classmethod
    def from_xywh(cls, coords: Union[list[float], np.ndarray]):
        return cls(coords=coords)

    def __getitem__(self, idx: int) -> float:
        return self.coords[idx]

    @staticmethod
    def _to_xywh(data: np.ndarray) -> np.ndarray:
        data[2:4] = data[2:4] - data[:2]
        return data

    @staticmethod
    def _to_xyxy(data: np.ndarray) -> np.ndarray:
        data[2:4] = data[2:4] + data[:2]
        return data

    @staticmethod
    def _convert_format(data: np.ndarray, dst_dfrmt: BoxFormat) -> np.ndarray:
        if dst_dfrmt == BoxFormat.xywh:
            return Bbox._to_xywh(data)

        if dst_dfrmt == BoxFormat.xyxy:
            return Bbox._to_xyxy(data)

        raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")

    def _as_dformat(self, dformat: BoxFormat) -> np.ndarray:
        data = self.coords.copy()
        if self.dformat != dformat:
            data = self._convert_format(data, dformat)
        return data


@dataclass
class YoloBbox(Bbox):
    @classmethod
    def from_bbox(cls, bbox: Bbox, height: int, width: int):
        bbox_np = bbox.xywh.copy()
        bbox_np[:2] += bbox_np[2:] / 2  # xy top-left corner to center
        bbox_np[[0, 2]] /= width  # normalize x
        bbox_np[[1, 3]] /= height  # normalize y

        return cls(bbox_np, dformat=BoxFormat.xywh)

    @classmethod
    def from_yolo(cls, coord: np.ndarray, height: int, width: int):
        bbox_np = coord.copy()

        bbox_np[[1, 3]] *= height  # reverse normalize y
        # reverse normalize x
        bbox_np[[0, 2]] *= width  # noqa: WPS362
        # xy top-left corner to center
        bbox_np[:2] -= bbox_np[2:] / 2  # noqa: WPS362

        return cls(bbox_np.reshape(-1), dformat=BoxFormat.xywh)


@dataclass
class BboxList:  # noqa: WPS306
    data: np.ndarray
    dformat: BoxFormat = BoxFormat.xywh

    def __post_init__(self):
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def append(self, vals):
        if self.data.shape[0]:
            self.data = np.vstack((self.data, vals))
        else:
            self.data = np.array(vals, dtype=np.float32)

    def insert(self, index, vals):
        if self.data.shape[0]:
            self.data = np.insert(self.data, index, vals, axis=0)
        else:
            self.data = np.array([vals], dtype=np.float32)

    def delete(self, indices):
        self.data = np.delete(self.data, indices, axis=0)

    def pop(self, ind: int):
        self.data = np.delete(self.data, ind, 0)

    def _as_dformat(self, dformat: BoxFormat):
        if self.data.shape[0] == 0:
            return self.data

        data = self.data.copy()
        if self.dformat != dformat:
            data = self._convert_format(data, self.dformat, dformat)
        return data

    def as_numpy(self, dformat: BoxFormat) -> np.ndarray:
        return self._as_dformat(dformat)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, index, value):
        self.data[index] = value

    def filtered(self, mask):
        return self.__class__(self.data[mask])

    def contains_xywh_coords(self, coords):
        coords = np.array(coords, dtype=np.float32)
        return coords in self.data

    @property
    def xywh(self) -> np.ndarray:
        return self.as_numpy(dformat=BoxFormat.xywh)

    @property
    def xyxy(self) -> np.ndarray:
        return self.as_numpy(dformat=BoxFormat.xyxy)

    @property
    def enclosing_bbox(self) -> Optional[Bbox]:
        # NOTE: always returns xywh format
        data = self.as_numpy(BoxFormat.xyxy)
        if data.shape[0] == 0:
            return None

        tl = np.amin(data[:, 0:2], axis=0)
        br = np.amax(data[:, 2:4], axis=0)

        bbox = np.concatenate((tl, br), axis=0)
        bbox[2:4] -= bbox[:2]  # noqa: WPS362
        return Bbox(bbox, dformat=BoxFormat.xywh)

    @property
    def area(self) -> np.ndarray:
        bboxes_wh = self.xywh[:, 2:4]
        return np.product(bboxes_wh, axis=1)

    @classmethod
    def from_list(
        cls,
        coords: list[tuple[float, float, float, float]],
        dformat: BoxFormat = BoxFormat.xywh,
    ):
        data = np.array(coords, dtype=np.float32)
        return cls(data, dformat=dformat)

    @classmethod
    def from_bbox_list(cls, bboxes: list[Bbox]):
        # NOTE: always returns xywh format
        if not len(bboxes):
            return cls.from_list([], dformat=BoxFormat.xywh)

        if not isinstance(bboxes[0], Bbox):
            raise TypeError("Invalid type")

        coords = [bbox.xywh for bbox in bboxes]
        return cls.from_list(coords, dformat=BoxFormat.xywh)

    @classmethod
    def from_xywh(cls, coords: list[tuple[float, float, float, float]]):
        return cls.from_list(coords, dformat=BoxFormat.xywh)

    @classmethod
    def from_xyxy(cls, coords: list[tuple[float, float, float, float]]):
        return cls.from_list(coords, dformat=BoxFormat.xyxy)

    @staticmethod
    def _convert_format(data: np.ndarray, src_dfrmt: BoxFormat, dst_dfrmt: BoxFormat):
        if src_dfrmt == BoxFormat.xyxy:
            if dst_dfrmt == BoxFormat.xywh:
                data[:, 2:4] = data[:, 2:4] - data[:, :2]  # noqa: WPS221
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")
        elif src_dfrmt == BoxFormat.xywh:
            if dst_dfrmt == BoxFormat.xyxy:
                data[:, 2:4] = data[:, 2:4] + data[:, :2]  # noqa: WPS221
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")

        raise ValueError(f"Unknown source dformat: {src_dfrmt}")
