from typing import Type

import numpy as np

from dataset_export.base_saver import DatasetSaver
from dataset_export.coco import CocoDatasetSaver
from dataset_export.voc import VocDatasetSaver
from dataset_export.yolo import YoloDatasetSaver


def get_type_save_annotation(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    class_names: list[str],
    type_save: str = 'yolo',
) -> DatasetSaver:
    """Factory"""
    types_saves: dict[str, Type[DatasetSaver]] = {
        'yolo': YoloDatasetSaver,
        'coco': CocoDatasetSaver,
        'voc': VocDatasetSaver,
    }
    SaverClass = types_saves.get(type_save.lower())
    if SaverClass is None:
        raise ValueError(f'Unknown dataset type: {type_save}')

    return SaverClass(images, masks, class_names)
