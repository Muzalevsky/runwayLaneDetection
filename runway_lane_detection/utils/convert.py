from typing import Any, Optional, Union

import numpy as np


def str_to_points(str_data: Union[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(str_data, str):
        return None

    points = [
        float(item)
        for item in str_data.strip("()[]").split(", ")  # noqa: WPS221
        if item not in ("()[]")
    ]
    return np.array(points)
