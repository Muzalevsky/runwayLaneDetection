import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inference import DetectionInference
from .metrics import DetectionMetricCalculator
from .types.detection_types import ImageDetections
from .types.image_types import ImageRGB
from .utils.convert import str_2_points
from .utils.fs import read_image, read_yolo_labels


class DetectionEvaluator:
    def __init__(self, model: DetectionInference, batch_size: int = 8, verbose: bool = False):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = model
        self._verbose = verbose
        self._batch_size = batch_size

    def _get_batch_labels(self, row: pd.Series) -> ImageDetections:
        gt_dets = []
        for label_id, label_name in self._model.names_map.items():
            points_np = str_2_points(row[label_name])

            if points_np is None:
                continue

            gt_data = np.concatenate((points_np, [label_id, -1]))
            gt_dets.append(gt_data)

        gt_dets = np.stack(gt_dets) if len(gt_dets) else np.empty((0, 7))
        gt_dets = ImageDetections(gt_dets)
        return gt_dets

    def _get_batch(
        self, img_fpaths: list[Path], lbl_fpaths: list[Path]
    ) -> tuple[list[ImageRGB], list[ImageDetections]]:
        b_images, b_gt_dets = [], []
        for img_fpath, lbl_fpath in zip(img_fpaths, lbl_fpaths):
            b_image = read_image(str(img_fpath))
            b_images.append(b_image)

            gt_np = read_yolo_labels(str(lbl_fpath))
            # NOTE: add synthetic conf column
            gt_dets = ImageDetections.from_yolo_labels(gt_np, *b_image.shape[:2])
            b_gt_dets.append(gt_dets)

        return b_images, b_gt_dets

    def evaluate(self, data_dpath: Path, conf: float = 0.001, iou: float = 0.3) -> pd.DataFrame:
        img_fpaths = sorted((data_dpath / "images").glob("*.PNG"))
        lbl_fpaths = sorted((data_dpath / "labels").glob("*.txt"))

        metric_calculator = DetectionMetricCalculator(self._model.names_map)

        stream = range(0, len(img_fpaths), self._batch_size)
        if self._verbose:
            stream = tqdm(stream, desc="Evaluation Processing")

        for ind in stream:
            b_img_fpaths = img_fpaths[ind : ind + self._batch_size]  # noqa: E203
            b_lbl_fpaths = lbl_fpaths[ind : ind + self._batch_size]  # noqa: E203

            b_images, b_gt_dets = self._get_batch(b_img_fpaths, b_lbl_fpaths)

            preds = self._model.detect(b_images, conf=conf, iou=iou)

            # --- Stats Collection for Metrics --- #
            for pred_dets, gt_dets in zip(preds, b_gt_dets):
                metric_calculator.update(pred_dets, gt_dets)

        # --- Metric Computation --- #
        metrics_df = metric_calculator.compute_metrics()
        metrics_df["Images"] = len(img_fpaths)
        return metrics_df.round(3)
