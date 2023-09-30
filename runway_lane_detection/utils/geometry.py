from typing import Optional

import numpy as np

from ..types.box_types import Bbox, BboxList, BoxFormat
from ..types.image_types import Image


def get_roi(bbox: Bbox, img: Image) -> Image:
    x, y, w, h = bbox.xywh.astype(int)
    return img[y : y + h, x : x + w]  # noqa: E203


def box_intersect(box_a: BboxList, box_b: BboxList) -> Optional[np.ndarray]:
    """Intersection between boxes.

    Return:
        (ndarray[a,b]): Matrix with intersection [pixels] between boxes
    """
    if box_a.size == 0 or box_b.size == 0:
        return None

    if isinstance(box_a, BboxList):
        box_a = box_a.as_numpy(BoxFormat.xyxy)

    if isinstance(box_b, BboxList):
        box_b = box_b.as_numpy(BoxFormat.xyxy)

    xy2_box_a = np.expand_dims(box_a[:, 2:4], 1)
    xy2_box_b = np.expand_dims(box_b[:, 2:4], 0)
    xy1_box_a = np.expand_dims(box_a[:, :2], 1)
    xy1_box_b = np.expand_dims(box_b[:, :2], 0)

    max_xy = np.minimum(xy2_box_a, xy2_box_b)
    min_xy = np.maximum(xy1_box_a, xy1_box_b)
    return np.clip(max_xy - min_xy, 0, None).prod(2)  # inter


def boxes_iou(box_a: BboxList, box_b: BboxList) -> Optional[np.ndarray]:
    """Compute IoU between boxes.

    Returns
    -------
        (ndarray[a,b]): Matrix with IoU between boxes
    """
    if not len(box_a) or not len(box_b):
        return None

    if isinstance(box_a, BboxList):
        box_a = box_a.xyxy

    if isinstance(box_b, BboxList):
        box_b = box_b.xyxy

    inter = box_intersect(box_a, box_b)

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = np.expand_dims(area_a, 1)

    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = np.expand_dims(area_b, 0)

    union = area_a + area_b - inter
    return inter / union


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    # https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py#L23
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
