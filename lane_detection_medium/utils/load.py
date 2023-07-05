_LABEL_NAMES = ["solid_white", "break_white", "zebra"]
_LABEL_IDS = [0, 1, 4]


def get_label_names() -> list[str]:
    return _LABEL_NAMES


def get_label_map() -> dict[str, int]:
    return dict(zip(_LABEL_NAMES, _LABEL_IDS))


def get_zebra_id() -> int:
    return get_label_map()["zebra"]
