from typing import Union

import cv2
import numpy as np

from ..types.box_types import Bbox, BboxList, BoxFormat
from ..types.image_types import ImageRGB


def put_label(
    canvas: np.ndarray,
    text: str,
    pos: Union[int, int],
    color: Union[int, int, int] = (0, 232, 201),
    font_size: float = 0.7,
    font_width: int = 1,
    box_color: Union[int, int, int] = (0, 232, 201),
):
    font = cv2.FONT_HERSHEY_COMPLEX
    text_wh, baseline = cv2.getTextSize(
        text, fontFace=font, fontScale=font_size, thickness=font_width
    )

    height_margin = 4
    x, y = pos
    y -= baseline * 2 + height_margin
    w, h = text_wh
    h += height_margin * 2
    cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, -1)

    cv2.putText(
        canvas,
        text,
        org=pos,
        fontFace=font,
        fontScale=font_size,
        color=color,
        lineType=cv2.LINE_AA,
        thickness=font_width,
    )


def render_bbox(
    canvas: ImageRGB,
    bbox: Bbox,
    color: tuple[int, int, int] = (0, 232, 201),
    line_thickness: int = 2,
    label: str = None,
    font_color: tuple[int, int, int] = (0, 0, 0),
    font_size: float = 0.8,
    font_width: float = 2,
    only_corners: bool = False,
    radius: int = 10,
    length: int = 40,
):
    x, y, w, h = bbox.xywh.astype(int)

    if only_corners:
        draw_corners(
            canvas,
            (x, y),
            (x + w, y + h),
            color=color,
            thickness=line_thickness,
            radius=radius,
            length=length,
        )
    else:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, line_thickness)

    if label is not None:
        put_label(  # noqa: WPS317
            canvas,
            label,
            (int(x), int(y - 5)),
            color=font_color,
            box_color=color,
            font_size=font_size,
            font_width=font_width,
        )


def render_boxes(  # noqa: WPS211
    canvas: ImageRGB,
    bboxes: BboxList,
    color: tuple[int, int, int] = (0, 232, 201),
    line_thickness: int = 2,
    labels: list[str] = None,
    font_color: tuple[int, int, int] = (0, 232, 201),
    font_size: float = 0.8,
    font_width: float = 2,
    only_corners: bool = True,
    radius: int = 50,
    length: int = 15,
    check_min_box_size: bool = True,
):
    """Bboxes rendering.

    Parameters
    ----------
        canvas (types.ImageRGB): Canvas to draw on it
        bboxes (np.ndarray): Bboxes to draw
        color (Tuple[int, int, int], optional): Color of bboxes lines. Defaults to (0, 0, 255).
        line_thickness (int, optional): Thickness of bboxes line. Defaults to 2.
        labels (List[str], optional): List of labels to draw near bboxes. Defaults to None.
        font_color (Tuple[int, int, int], optional): Color of label
        font_size (float, optional): Size of font
        font_width (float, optional): Width of font
        only_corner (bool, optional): enable rendering of only rectangle corners
        check_min_box_size (bool, optional): increase box size if it less than 2*radius
    """
    if isinstance(bboxes, np.ndarray):
        bboxes = BboxList.from_xyxy_list(bboxes)

    bboxes = bboxes.as_numpy(BoxFormat.xywh).astype(int)
    if check_min_box_size:
        min_radius = 2 * radius
        bboxes[:, 2:4] = np.clip(bboxes[:, 2:4], min_radius, None)

    for i, box in enumerate(bboxes[:, :4]):
        x, y, w, h = box
        if only_corners:
            draw_corners(
                canvas,
                (x, y),
                (x + w, y + h),
                color=color,
                thickness=line_thickness,
                radius=radius,
                length=length,
            )
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, line_thickness)

        if labels is not None:
            label = labels[i]
            put_label(  # noqa: WPS317
                canvas,
                label,
                (int(x), int(y - 5)),
                color=font_color,
                box_color=color,
                font_size=font_size,
                font_width=font_width,
            )


def draw_corners(
    canvas: ImageRGB,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int] = (0, 232, 201),
    thickness: int = 2,
    radius: int = 50,
    length: int = 15,
):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(canvas, (x1 + radius, y1), (x1 + radius + length, y1), color, thickness)
    cv2.line(canvas, (x1, y1 + radius), (x1, y1 + radius + length), color, thickness)
    cv2.ellipse(canvas, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(canvas, (x2 - radius, y1), (x2 - radius - length, y1), color, thickness)
    cv2.line(canvas, (x2, y1 + radius), (x2, y1 + radius + length), color, thickness)
    cv2.ellipse(canvas, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(canvas, (x1 + radius, y2), (x1 + radius + length, y2), color, thickness)
    cv2.line(canvas, (x1, y2 - radius), (x1, y2 - radius - length), color, thickness)
    cv2.ellipse(canvas, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(canvas, (x2 - radius, y2), (x2 - radius - length, y2), color, thickness)
    cv2.line(canvas, (x2, y2 - radius), (x2, y2 - radius - length), color, thickness)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def put_text(img, text, pos, color):
    img = cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return img
