from ..types.box_types import Bbox
from ..types.image_types import Image


def get_roi(bbox: Bbox, img: Image) -> Image:
    x, y, w, h = bbox.xywh.astype(int)
    return img[y : y + h, x : x + w]  # noqa: E203
