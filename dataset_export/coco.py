import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from dataset_export.config import SAVE_FOLDER
from dataset_export.helpers import extract_objects
from dataset_export.utils import generate_class_folder_name


class CocoDatasetSaver:
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

    def save(self, id_map: dict):
        self._create_coco_annotations(self.images, self.masks, id_map)

    def archive(self) -> str:
        shutil.make_archive(str(self.dataset_dir), 'zip', str(self.dataset_dir))
        shutil.rmtree(str(self.dataset_dir))
        return f'{self.dataset_dir}.zip'

    def _create_coco_annotations(self, images: list, masks: list, id_mapping):
        coco_data = {
            # 'info': {
            #     'description': 'Custom COCO Dataset',
            #     'version': '1.0',
            #     'year': 2024,
            #     'contributor': '',
            #     'url': ''
            # },
            # 'licenses': [{'id': 1, 'name': 'Academic', 'url': ''}],
            'categories': self._create_categories(),
            'images': [],
            'annotations': [],
        }

        annotation_id = 1
        for img_id, (image, mask) in enumerate(zip(images, masks)):
            img_filename = f'{img_id:012d}.jpg'
            img_path = self.images_dir / img_filename
            cv2.imwrite(str(img_path), image)

            coco_data['images'].append(
                {
                    'id': img_id,
                    'file_name': img_filename,
                    'width': image.shape[1],
                    'height': image.shape[0],
                }
            )

            # Добавляем аннотации (bounding boxes и сегментации)
            annotations = self._create_annotations(mask, img_id, id_mapping)
            coco_data['annotations'].extend(annotations)
            annotation_id += len(annotations)

        # Сохраняем JSON аннотации
        annotations_path = self.dataset_dir / 'annotations.json'
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)

    def _create_categories(self):
        return [
            {'id': class_id, 'name': class_name}
            for class_name, class_id in self.class_to_idx.items()
        ]

    def _create_annotations(self, mask_unique: np.ndarray, image_id: int, id_mapping):
        annotations = []
        result_objects = extract_objects(mask_unique, id_mapping)

        for obj in result_objects:
            x, y = obj['bbox'][0], obj['bbox'][1]
            w, h = obj['bbox'][2], obj['bbox'][3]
            data_images = {
                'image_id': image_id,
                'category_id': obj['order'],
                'bbox': [x, y, w, h],
            }
            annotations.append(data_images)
        return annotations
