import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .inference import DetectionInference
from .metrics import DetectionMetricCalculator
from .types.detection_types import ImageDetections
from .types.image_types import ImageRGB
from .utils.convert import str_2_points
from .utils.fs import read_image


class DetectionEvaluator:
    def __init__(
        self, model: DetectionInference, dpath: str, batch_size: int = 8, verbose: bool = False
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._model = model
        self._dpath = dpath
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

    def _get_batch(self, batch_df: pd.DataFrame) -> tuple[list[ImageRGB], list[ImageDetections]]:
        # TODO: check column name
        batch_df = batch_df[["fpath", *self._model.names_map.values()]]

        b_images, b_gt_dets = [], []
        for _row_ind, row in batch_df.iterrows():
            # --- Image Preprocessing --- #
            img_fpath = os.path.join(self._dpath, row["fpath"])
            b_image = read_image(img_fpath)
            b_images.append(b_image)

            # --- Labels Preprocessing --- #
            gt_dets = self._get_batch_labels(row)
            b_gt_dets.append(gt_dets)

        return b_images, b_gt_dets

    def evaluate(self, df: pd.DataFrame, conf: float = 0.001, iou: float = 0.3) -> pd.DataFrame:
        # TODO: check column name
        img_paths = df["fpath"].values

        metric_calculator = DetectionMetricCalculator(self._model.names_map)

        stream = range(0, img_paths.shape[0], self._batch_size)
        if self._verbose:
            stream = tqdm(stream, desc="Evaluation Processing")

        for ind in stream:
            batch_df = df.iloc[ind : ind + self._batch_size]  # noqa: E203
            b_images, b_gt_dets = self._get_batch(batch_df)
            preds = self._model.detect(b_images, conf=conf, iou=iou)

            # TODO: add data storage

            # --- Stats Collection for Metrics --- #
            for pred_dets, gt_dets in zip(preds, b_gt_dets):
                metric_calculator.update(pred_dets, gt_dets)

        # --- Metric Computation --- #
        metrics_df = metric_calculator.compute_metrics()
        metrics_df["Images"] = img_paths.shape[0]
        return metrics_df.round(3)
