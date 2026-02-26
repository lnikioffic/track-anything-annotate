import shutil
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import cv2
import numpy as np

from dataset_export.config import SAVE_FOLDER
from dataset_export.helpers import extract_objects
from dataset_export.utils import generate_class_folder_name


class VocDatasetSaver:
    """
    Экспорт аннотаций в формат Pascal VOC.

    Структура датасета:
    dataset/
    ├── JPEGImages/
    │   ├── 000000000000.jpg
    │   └── ...
    ├── Annotations/
    │   ├── 000000000000.xml
    │   └── ...
    ├── SegmentationClass/      # TODO: семантическая сегментация
    │   └── ...
    ├── SegmentationObject/     # TODO: инстанс сегментация
    │   └── ...
    └── labels.txt              # маппинг классов
    """

    def __init__(
        self, images: list[np.ndarray], masks: list[np.ndarray], class_names: list[str]
    ) -> None:
        self.images = images
        self.masks = masks
        self.class_to_idx = {}

        for i, name in enumerate(class_names):
            self.class_to_idx[name] = i
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names, start=1)}

        dataset_name = generate_class_folder_name(class_names)
        dataset_path = Path(SAVE_FOLDER / dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = SAVE_FOLDER / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Основные директории VOC
        self.jpeg_images_dir = self.dataset_dir / 'JPEGImages'
        self.annotations_dir = self.dataset_dir / 'Annotations'
        self.segmentation_class_dir = self.dataset_dir / 'SegmentationClass'
        self.segmentation_object_dir = self.dataset_dir / 'SegmentationObject'

        self.jpeg_images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        # TODO: раскомментировать при реализации сегментации
        # self.segmentation_class_dir.mkdir(parents=True, exist_ok=True)
        # self.segmentation_object_dir.mkdir(parents=True, exist_ok=True)

    def save(self, id_map: dict) -> None:
        """Сохраняет датасет в формате VOC."""
        self._save_images()
        self._save_annotations(id_map)
        self._save_class_labels()
        # TODO: реализовать при необходимости
        # self._save_segmentation_class()
        # self._save_segmentation_object()

    def archive(self) -> str:
        """Архивирует датасет в ZIP и удаляет исходную папку."""
        shutil.make_archive(str(self.dataset_dir), 'zip', str(self.dataset_dir))
        shutil.rmtree(str(self.dataset_dir))
        return f'{self.dataset_dir}.zip'

    def _save_images(self) -> None:
        """Сохраняет изображения в JPEGImages."""
        for img_id, image in enumerate(self.images):
            img_filename = f'{img_id:012d}.jpg'
            img_path = self.jpeg_images_dir / img_filename
            cv2.imwrite(str(img_path), image)

    def _save_annotations(self, id_map: dict) -> None:
        """Сохраняет XML аннотации в Annotations."""
        for img_id, mask in enumerate(self.masks):
            img_filename = f'{img_id:012d}.jpg'
            xml_root = self._create_xml_annotation(img_filename, mask, img_id, id_map)

            xml_filename = f'{img_id:012d}.xml'
            xml_path = self.annotations_dir / xml_filename
            self._write_xml(xml_root, xml_path)

    def _create_xml_annotation(
        self, filename: str, mask: np.ndarray, image_id: int, id_mapping: dict
    ) -> ET.Element:
        """Создаёт XML аннотацию в формате VOC."""
        height, width = mask.shape[:2]

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'VOC'
        ET.SubElement(root, 'filename').text = filename
        ET.SubElement(root, 'source').text = 'track-anything-annotate'

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'

        objects = extract_objects(mask, id_mapping)
        for obj in objects:
            obj_elem = ET.SubElement(root, 'object')
            ET.SubElement(obj_elem, 'name').text = obj['class_name']
            ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
            ET.SubElement(obj_elem, 'truncated').text = '0'
            ET.SubElement(obj_elem, 'difficult').text = '0'

            x, y, w, h = obj['bbox']
            bbox = ET.SubElement(obj_elem, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(int(x))
            ET.SubElement(bbox, 'ymin').text = str(int(y))
            ET.SubElement(bbox, 'xmax').text = str(int(x + w))
            ET.SubElement(bbox, 'ymax').text = str(int(y + h))

        return root

    def _write_xml(self, root: ET.Element, path: Path) -> None:
        """Записывает XML с красивым форматированием."""
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent='  ', encoding='utf-8')
        with open(path, 'wb') as f:
            f.write(pretty_xml)

    def _save_class_labels(self) -> None:
        """Сохраняет файл с маппингом классов."""
        labels_path = self.dataset_dir / 'labels.txt'
        with open(labels_path, 'w', encoding='utf-8') as f:
            for class_name, idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
                f.write(f'{idx} {class_name}\n')

    def _save_segmentation_class(self) -> None:
        """
        TODO: Сохраняет семантическую сегментацию.

        Каждый пиксель содержит ID класса.
        Формат: PNG с палитрой VOC.
        """
        pass

    def _save_segmentation_object(self) -> None:
        """
        TODO: Сохраняет инстанс сегментацию.

        Каждый объект имеет уникальный ID.
        Фон = 0, объекты = 1, 2, 3, ...
        """
        pass
