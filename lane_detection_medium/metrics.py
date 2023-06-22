import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .types.detection_types import ImageDetections
from .utils.geometry import boxes_iou, smooth


def clf_report_df(true_data, pred_data) -> pd.DataFrame:
    clf_report = classification_report(true_data, pred_data, output_dict=True)
    df = pd.DataFrame(clf_report).round(3).T
    df.loc[["accuracy"], ["precision", "recall", "support"]] = ""
    return df


def clf_report_extend_specificity(
    y_true: pd.Series, y_pred: pd.Series, report: pd.DataFrame, class_labels: list[str]
) -> pd.DataFrame:
    support = report.loc[class_labels, "support"]

    cm = confusion_matrix(y_true, y_pred)
    tp = cm.diagonal()
    fp = np.sum(cm, axis=0) - tp
    # fn = np.sum(cm, axis=1) - tp
    tn = np.sum(tp) - tp
    specificity = np.round(tn / (tn + fp), decimals=3)
    specificity = pd.Series(specificity, index=class_labels, name="specificity")

    weights = support / np.sum(support)
    # micro_specificity = np.sum(tn) / (np.sum(tn) + np.sum(fp))
    macro_specificity = np.round(np.mean(specificity), decimals=3)
    weighted_specificity = np.round(np.sum(specificity * weights), decimals=3)

    other_cols = pd.Series(
        ["", macro_specificity, weighted_specificity],
        index=["accuracy", "macro avg", "weighted avg"],
        name=specificity.name,
    )
    specificity = specificity.append(other_cols)

    report = pd.concat((specificity, report), axis=1)
    return report


def compute_ap(recall: np.ndarray, precision: np.ndarray, method: str = "interp") -> float:
    """Compute the average precision, given the recall and precision curves.

    Parameters
    ----------
    recall : np.ndarray
        The recall curve
    precision : np.ndarray
        The precision curve
    method : str, optional
        The computation method name, by default "interp"
        * "interp" - Intergate area under curve
        * "continuous"

    Notes
    -----
    Source YOLOv5: https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py#L98

    Returns
    -------
    float
        The average precision value.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def _compute_ap_per_class(
    tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray, eps: float = 1e-16
):
    """Compute the average precision per class.

    Notes
    -----
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    Source YOLOv5: https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py#L31

    Parameters
    ----------
    tp : np.ndarray
        The bool matrix pred boxes per each threshold
    conf : np.ndarray
        The model confidences array
    pred_cls : np.ndarray
        The model predicted class labels array
    gt_cls : np.ndarray
        The ground truth labels array
    eps : float, optional
        epsilon, by default 1e-16

    Returns
    -------
    _type_
        _description_
    """

    # Sort by confidence descending
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, gt_class_counts = np.unique(gt_cls, return_counts=True)

    px = np.linspace(0, 1, 1000)
    ap = np.zeros((unique_classes.shape[0], tp.shape[1]))
    p, r = np.zeros((unique_classes.shape[0], 1000)), np.zeros((unique_classes.shape[0], 1000))

    for cls_idx, cls_id in enumerate(unique_classes):
        mask = pred_cls == cls_id
        # number of labels
        gt_label_n = gt_class_counts[cls_idx]

        n_p = i.sum()  # number of predictions
        if n_p == 0 or gt_label_n == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[mask]).cumsum(0)
        tpc = tp[mask].cumsum(0)

        # Recall: TP / (TP + FN) -> TP / gt_label_n
        recall = tpc / (gt_label_n + eps)  # recall curve
        r[cls_idx] = np.interp(-px, -conf.astype(np.float32)[mask], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        # p at pr_score
        p[cls_idx] = np.interp(-px, -conf.astype(np.float32)[mask], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[cls_idx, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    return p, r, f1, ap


class DetectionMetricCalculator:
    def __init__(self, names):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._stats = []
        self._names = names

        # NOTE: iou vector for mAP@0.5:0.95
        self._iouv = np.linspace(0.5, 0.95, 10)

        self._columns = ["Class", "Instances", "Precision", "Recall", "F1", "mAP50", "mAP50-95"]

    def update(self, pred_dets: ImageDetections, gt_dets: ImageDetections):
        pred_stats = np.zeros(shape=(len(pred_dets), len(self._iouv)), dtype=np.bool_)

        if len(gt_dets) and len(pred_dets):
            iou_mat = boxes_iou(gt_dets.bboxes, pred_dets.bboxes)

            # NOTE: reshape is needed to compare each gt bbox with each pred bbox
            #       output shape = [gt_dets.bboxes.shape[0], pred_dets.bboxes.shape[0]]
            correct_class_mask = gt_dets.class_ids.reshape(-1, 1) == pred_dets.class_ids

            for i in range(len(self._iouv)):
                # IoU > threshold and classes match
                x = np.where((iou_mat >= self._iouv[i]) & correct_class_mask)

                if x[0].shape[0]:
                    # [[row_ind, col_ind, iou_val]]
                    matches = np.column_stack((*x, iou_mat[x[0], x[1]]))

                    if x[0].shape[0] > 1:
                        # sort match-rows iou-descending
                        matches = matches[matches[:, 2].argsort()[::-1]]

                        # choose best matches per class per particular IoU threshold
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    # mask pred detections as successful for specific iou threshold
                    pred_stats[matches[:, 1].astype(int), i] = True

        self._stats.append((pred_stats, pred_dets.confs, pred_dets.class_ids, gt_dets.class_ids))

    def compute_metrics(self):
        # concat stats per field
        stats = [np.concatenate(data) for data in zip(*self._stats)]

        p, r, f1, ap = _compute_ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95

        # mean metrics
        mp, mr, map50, map_ = p.mean(), r.mean(), ap50.mean(), ap.mean()

        # number of targets per class
        nt = np.bincount(stats[3].astype(int), minlength=np.unique(stats[-1]).shape[0])

        df_data = []
        if len(self._names) > 1:
            df_data = [["all", nt.sum(), mp, mr, f1.mean(), map50, map_]]

        for class_id, class_name in self._names.items():
            df_data.append(
                [
                    class_name,
                    nt[class_id],
                    p[class_id],
                    r[class_id],
                    f1[class_id],
                    ap50[class_id],
                    ap[class_id],
                ]
            )

        df = pd.DataFrame(df_data, columns=self._columns)
        return df
