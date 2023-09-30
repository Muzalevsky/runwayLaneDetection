import logging

import torch
from tqdm import tqdm

from .path import YOLO_DPATH
from .types.base_types import Dict
from .types.detection_types import ImageDetections
from .types.image_types import ImageRGB


class DetectionInference:
    """Class Inference implementation for trained YOLOv5 model."""

    model_name: str = str(YOLO_DPATH)

    def __init__(  # noqa: WPS211
        self,
        model,
        img_size: tuple[int, int] = (640, 640),
        batch_size: int = 16,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = model

        self._device = device
        self._batch_size = batch_size
        self._img_size = img_size

        self._verbose = verbose

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

        return cls(model=model, img_size=img_size, device=device, **kwargs)

    @staticmethod
    def batch_generator(images: list[str], batch_size: int = 8):
        for index in range(0, len(images), batch_size):
            b_images = images[index : index + batch_size]
            yield b_images

    def detect(
        self,
        images: list[ImageRGB],
        conf: float = 0.001,
        iou: float = 0.3,
    ) -> list[ImageDetections]:
        self.conf_threshold = conf
        self.iou_threshold = iou

        stream = self.batch_generator(images, batch_size=self._batch_size)
        if self._verbose:
            stream = tqdm(
                stream,
                total=round(len(images) / self._batch_size),
                desc="Detection Processing",
            )

        results = []
        for b_images in stream:
            b_output = self._model(b_images, size=self._img_size, augment=False, profile=False)
            b_results = b_output.pandas().xyxy
            results += b_results

        return [ImageDetections(res.to_numpy()) for res in results]
