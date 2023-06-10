from typing import Optional, Union

import numpy as np

from .box_types import Bbox, BboxList, BoxFormat


class Detection:
    """Single Detection Result Class Implementation."""

    def __init__(
        self, label_id: int, conf: float, coord: np.ndarray, label_name: Optional[str] = None
    ):
        self._label_name = label_name
        self._label_id = label_id
        self._conf = conf
        self._bbox = Bbox(coord, dformat=BoxFormat.xyxy)

    @property
    def bbox(self) -> Bbox:
        return self._bbox

    @property
    def conf(self) -> float:
        return self._conf

    @property
    def label_name(self) -> Optional[str]:
        return self._label_name


class ImageDetections:
    """Single Image Detections Class Implementation."""

    def __init__(self, result: np.ndarray, dformat: BoxFormat = BoxFormat.xyxy):
        self._raw_result = result

        self._bboxes = BboxList(self._raw_result[:, :4], dformat=dformat)

    @property
    def raw_detection(self) -> np.ndarray:
        return self._raw_result

    @property
    def bboxes(self) -> BboxList:
        return self._bboxes

    @property
    def class_ids(self) -> np.ndarray:
        return self._raw_result[:, -2]

    @property
    def class_labels(self) -> np.ndarray:
        return self._raw_result[:, -1]

    @property
    def confs(self) -> np.ndarray:
        return self._raw_result[:, 4]

    def sort(self, ascending: bool = False):
        """Sort detections by confidence values."""

        conf_data = self._raw_result[:, 4]
        if not ascending:
            conf_data = -(self._raw_result[:, 4])

        self._raw_result = self._raw_result[conf_data.argsort()]

    def get_index(self, cls_name: Union[str, int]) -> Optional[np.ndarray]:
        if isinstance(cls_name, str):
            index_pos = -1
        elif isinstance(cls_name, int):
            index_pos = -2
        else:
            raise ValueError(f"cls_name contains wrong value '{cls_name}'")

        bbox_indexes = np.where(self._raw_result[:, index_pos] == cls_name)[0]
        if not len(bbox_indexes):
            bbox_indexes = None
        return bbox_indexes

    def delete(self, index: int | np.ndarray):
        updated_result = np.delete(self._raw_result, index, axis=0)
        return ImageDetections(updated_result)

    def __len__(self):
        return len(self._raw_result)

    @classmethod
    def from_yolo_labels(cls, data: np.ndarray, height: float, width: float):
        # shift label_id to the last column
        result = np.roll(data, -1, axis=1)

        bbox_np = result[:, :4].copy()
        bbox_np[:, [1, 3]] *= height  # reverse normalize y
        bbox_np[:, [0, 2]] *= width  # reverse normalize x
        bbox_np[:, :2] -= bbox_np[:, 2:] / 2  # xy top-left corner to center

        # fill columns of label names with -1
        result = np.c_[bbox_np, result[:, -1], np.ones(result.shape[0]) * -1]
        return cls(result, dformat=BoxFormat.xywh)

    def filter_by_confidence(self, conf_threshold: float) -> np.ndarray:
        idx = np.where(self._raw_result[:, 4] >= conf_threshold)[0]
        return self._raw_result[idx]

    def __getitem__(self, idx) -> Detection:
        res = self._raw_result[idx]

        return Detection(label_name=res[-1], label_id=res[-2], conf=res[4], coord=res[:4])
