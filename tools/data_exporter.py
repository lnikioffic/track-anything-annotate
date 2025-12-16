import json
import shutil
import uuid
from pathlib import Path
from typing import Protocol, Type

import cv2
import numpy as np

from tools.contour_detector import getting_coordinates
from tools.mask_display import mask_map

SAVE_FOLDER = Path.cwd() / 'video-test'


class TypeSave(Protocol):
    def create_dataset(self, id_map) -> None:
        pass

    def create_archive(self) -> str:
        pass


def get_type_save_annotation(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    type_save: str = 'yolo',
) -> TypeSave:
    """Factory"""
    types_saves: dict[str, Type[TypeSave]] = {
        'yolo': YoloDatasetSaver,
        'coco': CocoDatasetSaver,
    }
    SaverClass = types_saves.get(type_save.lower())
    if SaverClass is None:
        raise ValueError(f'Unknown dataset type: {type_save}')

    return SaverClass(images, masks, names_class)


def create_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    id_map,
    type_save: str = 'yolo',
):
    assert len(masks) == len(images)

    send_images = []
    send_masks = []
    for i in range(len(masks)):
        if i % 2 == 0:
            send_images.append(images[i])
            send_masks.append(masks[i])

    saver = get_type_save_annotation(send_images, send_masks, names_class, type_save)
    saver.create_dataset(id_map)
    print(f'Saved archive {saver.create_archive()}')


def generate_class_folder_name(names_class: list[str]):
    combined = ''.join(names_class)
    base_name = combined[:10] if len(combined) > 10 else combined

    folder_name = f'dt-{base_name}-{uuid.uuid4()}'
    return folder_name


class CocoDatasetSaver:
    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ):
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

    def create_dataset(self, id_map):
        self._create_coco_annotations(self.images, self.masks, id_map)

    def create_archive(self) -> str:
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
        result_objects = []
        for mask_id, mask in enumerate(mask_map(mask_unique), 1):
            if mask_id not in id_mapping:
                continue

            bbox = getting_coordinates(mask)

            obj_info = id_mapping[mask_id]
            result_objects.append(
                {
                    'mask_id': mask_id,
                    'class_name': obj_info['class'],
                    'bbox': bbox,
                    'order': obj_info['order'],
                }
            )
        for obj in result_objects:
            x, y = obj['bbox'][0][0], obj['bbox'][0][1]
            w, h = obj['bbox'][0][2], obj['bbox'][0][3]
            data_images = {
                'image_id': image_id,
                'category_id': obj['order'],
                'bbox': [x, y, w, h],
            }
            annotations.append(data_images)
        return annotations


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

    def create_dataset(self, id_map):
        for idx, (image, mask) in enumerate(zip(self.images, self.masks)):
            image_filename = f'image_{idx + 1:04d}'
            image_path = self.images_dir / f'{image_filename}.jpg'
            label_path = self.labels_dir / f'{image_filename}.txt'

            cv2.imwrite(str(image_path), image)
            self._save_yolo_annotation(image, mask, str(label_path), id_map)

        self._save_class_names(self.dataset_dir / 'classes.txt')

    def create_archive(self) -> str:
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
            result_objects = []
            for mask_id, mask in enumerate(mask_map(mask_unique), 1):
                if mask_id not in id_mapping:
                    continue
                bbox = getting_coordinates(mask)

                obj_info = id_mapping[mask_id]
                result_objects.append(
                    {
                        'mask_id': mask_id,
                        'class_name': obj_info['class'],
                        'bbox': bbox,
                        'order': obj_info['order'],
                    }
                )
            for obj in result_objects:
                x, y = obj['bbox'][0][0], obj['bbox'][0][1]
                w, h = obj['bbox'][0][2], obj['bbox'][0][3]

                x_center = x + w / 2
                y_center = y + h / 2

                norm_xc = x_center / img_width
                norm_yc = y_center / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                file.write(
                    f'{obj["order"]} {norm_xc} {norm_yc} {norm_width} {norm_height}\n'
                )
