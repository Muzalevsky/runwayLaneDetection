from typing import Any, Optional, Union

import numpy as np


def str_2_points(str_data: Union[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(str_data, str):
        return None

    points = [float(item) for item in str_data.strip("()[]").split(", ") if item not in ("()[]")]
    points = np.array(points)
    return points
