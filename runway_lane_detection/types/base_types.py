class Dict(dict):  # noqa: WPS600
    def keys(self) -> list:
        return list(super().keys())

    def values(self) -> list:  # noqa: WPS110
        return list(super().values())
