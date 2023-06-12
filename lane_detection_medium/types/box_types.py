from typing import Union

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from shapely import Point

from .image_types import Image


class BoxFormat(Enum):
    """List of formats."""

    xywh = auto()
    xyxy = auto()


@dataclass
class Bbox:
    def __init__(
        self,
        coords: Union[list[float], tuple[float], np.ndarray],
        dformat: BoxFormat = BoxFormat.xywh,
    ):
        self.coords = np.array(coords, dtype=np.float32)
        self.dformat = dformat

    @property
    def xywh_numpy(self) -> np.ndarray:
        data = self._as_dformat(BoxFormat.xywh)
        return data

    @property
    def xyxy_numpy(self) -> np.ndarray:
        data = self._as_dformat(BoxFormat.xyxy)
        return data

    @property
    def center(self) -> Point:
        w, h = self._as_dformat(BoxFormat.xywh)[2:]
        return Point(w / 2, h / 2)

    @property
    def area(self) -> float:
        w, h = self._as_dformat(BoxFormat.xywh)[2:]
        return w * h

    @property
    def perimeter(self) -> float:
        w, h = self._as_dformat(BoxFormat.xywh)[2:]
        return 2 * (w + h)

    @property
    def points(self) -> np.ndarray:
        # NOTE: from left to right
        pt0, pt2 = self._as_dformat(BoxFormat.xyxy).reshape(-1, 2)
        pt1 = [pt2[0], pt0[1]]
        pt3 = [pt0[0], pt2[1]]
        points = np.array([pt0, pt1, pt2, pt3])
        return points

    @classmethod
    def from_xyxy(cls, coords):
        return cls(coords=coords)

    @classmethod
    def from_xywh(cls, coords):
        return cls(coords=coords)

    def get_roi(self, img: Image) -> Image:
        bbox_np = self._as_dformat(BoxFormat.xywh)
        x, y, w, h = bbox_np.astype(int)
        return img[y : y + h, x : x + w]  # noqa: E203

    def __getitem__(self, idx: int) -> float:
        return self.coords[idx]

    def apply_inverse_offset(self, coords):
        self.coords[:2] -= np.array(coords[:2])

    def apply_offset(self, coords):
        self.coords[:2] += np.array(coords[:2])

    @staticmethod
    def _convert_format(data: np.ndarray, src_dfrmt: BoxFormat, dst_dfrmt: BoxFormat):
        if src_dfrmt == BoxFormat.xyxy:
            if dst_dfrmt == BoxFormat.xywh:
                data[2:4] = data[2:4] - data[:2]
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")
        elif src_dfrmt == BoxFormat.xywh:
            if dst_dfrmt == BoxFormat.xyxy:
                data[2:4] = data[2:4] + data[:2]
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")

        raise ValueError(f"Unknown source dformat: {src_dfrmt}")

    def _as_dformat(self, dformat: BoxFormat):
        data = self.coords.copy()
        if self.dformat != dformat:
            data = self._convert_format(data, self.dformat, dformat)
        return data

    def __repr__(self):
        return str(self.coords)

    def letterbox(self, params) -> np.ndarray:
        # NOTE: always returns xywh format
        bbox = self._as_dformat(BoxFormat.xywh)
        if len(bbox):
            bbox[[1, 3]] *= params.scale[0]
            bbox[[0, 2]] *= params.scale[1]
            bbox[[1, 0]] += params.padding
        return bbox


class YoloBbox(Bbox):
    @classmethod
    def from_bbox(cls, bbox: Bbox, width: int, height: int):
        bbox_np = bbox.xywh_numpy.copy()
        bbox_np[:2] += bbox_np[2:] / 2  # xy top-left corner to center
        bbox_np[[0, 2]] /= width  # normalize x
        bbox_np[[1, 3]] /= height  # normalize y

        return cls(bbox_np, dformat=BoxFormat.xywh)


class BboxList:
    def __init__(
        self,
        data: np.ndarray,
        dtype=np.float32,
        dformat: BoxFormat = BoxFormat.xywh,
    ):
        self.data = data.astype(dtype)
        self._dformat = dformat

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def area(self):
        bboxes_wh = self.as_xywh_numpy()[:, 2:4]
        bbox_areas = np.product(bboxes_wh, axis=1)
        return bbox_areas

    @classmethod
    def from_bbox_list(cls, bboxes: list[Bbox]):
        # NOTE: always returns xywh format
        if not len(bboxes):
            return cls.from_list([], dformat=BoxFormat.xywh)

        if not isinstance(bboxes[0], Bbox):
            raise TypeError("Invalid type")

        coords = [bbox.xywh_numpy for bbox in bboxes]
        return cls.from_list(coords, dformat=BoxFormat.xywh)

    @classmethod
    def from_list(
        cls,
        coords: list[tuple[float, float, float, float]],
        dformat: BoxFormat = BoxFormat.xywh,
    ):
        data = np.array(coords, dtype=np.float32)
        return cls(data, dformat=dformat)

    @classmethod
    def from_xywh_list(cls, coords: list[tuple[float, float, float, float]]):
        return cls.from_list(coords, dformat=BoxFormat.xywh)

    @classmethod
    def from_xyxy_list(cls, coords: list[tuple[float, float, float, float]]):
        return cls.from_list(coords, dformat=BoxFormat.xyxy)

    @classmethod
    def from_abs_xyxy(cls, coords, height: int, width: int):
        bbox_np = np.array(coords, dtype="float32")

        bbox_np[[0, 2]] /= width
        bbox_np[[1, 3]] /= height

        x = bbox_np[[0, 2]].sum() / 2
        y = bbox_np[[1, 3]].sum() / 2
        w = bbox_np[2] - bbox_np[0]
        h = bbox_np[3] - bbox_np[1]

        return cls.from_list(coords=[(x, y, w, h)], dformat=BoxFormat.xywh)

    @classmethod
    def get_box_from_file(cls, path: str):
        array = np.load(path, allow_pickle=True)

        def calc_box(frame_index: int, image_shape):
            boxes = array.take(np.where(array[:, 0] == frame_index)[0], axis=0)[:, 1:5]
            boxes[:, [1, 3]] *= image_shape[0]
            boxes[:, [0, 2]] *= image_shape[1]
            return cls.from_list(boxes, dformat=BoxFormat.xyxy)

        def get_box(frame_index: int):
            boxes = array.take(np.where(array[:, 0] == frame_index)[0], axis=0)[:, 1:5]
            return cls.from_list(boxes, dformat=BoxFormat.xyxy)

        return get_box

    @property
    def dformat(self) -> BoxFormat:
        return self._dformat

    def get_enclosing_bbox(self) -> Bbox:
        # NOTE: always returns xywh format
        data = self.as_numpy(BoxFormat.xyxy)
        if data.shape[0] == 0:
            return None

        tl = np.amin(data[:, 0:2], axis=0)
        br = np.amax(data[:, 2:4], axis=0)

        bbox = np.concatenate((tl, br), axis=0)
        bbox[2:4] -= bbox[:2]
        return Bbox(bbox, dformat=BoxFormat.xywh)

    def pop(self, ind):
        self.data = np.delete(self.data, ind, 0)

    def append(self, vals):
        if len(self.data):
            self.data = np.vstack((self.data, vals))
        else:
            self.data = np.array(vals, dtype=np.float32)

    def insert(self, index, vals):
        if len(self.data):
            self.data = np.insert(self.data, index, vals, axis=0)
        else:
            self.data = np.array([vals], dtype=np.float32)

    def delete(self, indices):
        self.data = np.delete(self.data, indices, axis=0)

    @staticmethod
    def _convert_format(data: np.ndarray, src_dfrmt: BoxFormat, dst_dfrmt: BoxFormat):
        if src_dfrmt == BoxFormat.xyxy:
            if dst_dfrmt == BoxFormat.xywh:
                data[:, 2:4] = data[:, 2:4] - data[:, :2]
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")
        elif src_dfrmt == BoxFormat.xywh:
            if dst_dfrmt == BoxFormat.xyxy:
                data[:, 2:4] = data[:, 2:4] + data[:, :2]
                return data

            raise ValueError(f"Unknown destination dformat: {dst_dfrmt}")

        raise ValueError(f"Unknown source dformat: {src_dfrmt}")

    @staticmethod
    def convert_relative_to_absolute(data: np.ndarray, img_shape: tuple):
        data = data.copy()
        data[:, [0, 2]] *= img_shape[1]
        data[:, [1, 3]] *= img_shape[0]
        return data

    def _as_dformat(self, dformat: BoxFormat):
        if self.data.shape[0] == 0:
            return self.data

        data = self.data.copy()
        if self._dformat != dformat:
            data = self._convert_format(data, self._dformat, dformat)
        return data

    def as_numpy(self, dformat: BoxFormat):
        data = self._as_dformat(dformat)
        return data

    def as_xywh_numpy(self):
        return self.as_numpy(dformat=BoxFormat.xywh)

    def as_xyxy_numpy(self):
        return self.as_numpy(dformat=BoxFormat.xyxy)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, index, value):
        self.data[index] = value

    def filtered(self, mask):
        return self.__class__(self.data[mask])

    def __repr__(self):
        return str(self.data)

    def contains_xywh_coords(self, coords):
        coords = np.array(coords, dtype=np.float32)
        return coords in self.data
