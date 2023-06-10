import logging

import torch

from .path import YOLO_DPATH
from .types.base_types import Dict
from .types.detection_types import ImageDetections
from .types.image_types import ImageRGB


class DetectionInference:
    """Class Inference implementation for trained YOLOv5 model."""

    model_name = YOLO_DPATH

    def __init__(
        self,
        model,
        img_size: tuple[int, int] = (640, 640),
        batch_size: int = 16,
        device: str = "cpu",
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = model

        self._device = device
        self._batch_size = batch_size
        self._img_size = img_size

    @property
    def names_map(self) -> Dict:
        return Dict(self._model.names)

    @property
    def conf_threshold(self) -> float:
        return self._model.conf

    @conf_threshold.setter
    def conf_threshold(self, val: float):
        self._model.conf = val

    @property
    def iou_threshold(self) -> float:
        return self._model.iou

    @iou_threshold.setter
    def iou_threshold(self, val: float):
        self._model.iou = val

    @classmethod
    def from_file(cls, fpath: str, img_size: tuple[int, int], device=None, **kwargs):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = torch.hub.load(
            DetectionInference.model_name,
            "custom",
            path=fpath,
            force_reload=False,
            autoshape=True,
            device=device,
            source="local",
        )

        obj_ = cls(model=model, img_size=img_size, device=device, **kwargs)
        return obj_

    def detect(
        self, images: list[ImageRGB], conf: float = 0.001, iou: float = 0.3
    ) -> list[ImageDetections]:
        self.conf_threshold = conf
        self.iou_threshold = iou

        output = self._model(images, size=self._img_size, augment=False, profile=False)
        results = output.pandas().xyxy

        img_detections = [ImageDetections(res.to_numpy()) for res in results]

        return img_detections
