from typing import Protocol

import numpy as np


class DatasetSaver(Protocol):
    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ) -> None: ...

    def save(self, id_map: dict) -> None: ...
    def archive(self) -> str: ...
