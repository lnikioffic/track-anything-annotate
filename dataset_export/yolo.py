import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_export.config import SAVE_FOLDER
from dataset_export.helpers import extract_objects
from dataset_export.utils import generate_class_folder_name


class YoloDatasetSaver:
    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ) -> None:
        self.images = images
        self.masks = masks
        self.class_to_idx = {}

        for i, name in enumerate(class_names):
            self.class_to_idx[name] = i

        dataset_name = generate_class_folder_name(class_names)
        dataset_path = Path(SAVE_FOLDER / dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = SAVE_FOLDER / dataset_name

        self.images_dir = self.dataset_dir / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.labels_dir = self.dataset_dir / 'labels'
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def save(self, id_map: dict):
        for idx, (image, mask) in enumerate(zip(self.images, self.masks)):
            image_filename = f'image_{idx + 1:04d}'
            image_path = self.images_dir / f'{image_filename}.jpg'
            label_path = self.labels_dir / f'{image_filename}.txt'

            cv2.imwrite(str(image_path), image)
            self._save_yolo_annotation(image, mask, str(label_path), id_map)

        self._save_class_names(self.dataset_dir / 'classes.txt')

    def archive(self) -> str:
        shutil.make_archive(str(self.dataset_dir), 'zip', str(self.dataset_dir))
        shutil.rmtree(str(self.dataset_dir))
        return f'{self.dataset_dir}.zip'

    def _save_class_names(self, file_path: Path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for class_name, class_id in self.class_to_idx.items():
                file.write(f'{class_id} {class_name}\n')

    def _save_yolo_annotation(
        self,
        images: np.ndarray,
        mask_unique: np.ndarray,
        file_path: str,
        id_mapping,
    ):
        img_height = images.shape[0]
        img_width = images.shape[1]
        with open(file_path, 'w', encoding='utf-8') as file:
            result_objects = extract_objects(mask_unique, id_mapping)

            for obj in result_objects:
                x, y = obj['bbox'][0], obj['bbox'][1]
                w, h = obj['bbox'][2], obj['bbox'][3]

                x_center = x + w / 2
                y_center = y + h / 2

                norm_xc = x_center / img_width
                norm_yc = y_center / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                file.write(f'{obj["order"]} {norm_xc} {norm_yc} {norm_width} {norm_height}\n')
