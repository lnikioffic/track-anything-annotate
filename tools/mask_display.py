import numpy as np

from XMem2.inference.interact.interactive_utils import color_map_np


def visualize_unique_mask(unique_mask):
    colored_mask = color_map_np[unique_mask]
    return colored_mask # .astype(np.uint8)


def visualize_wb_mask(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    map = np.array([[0, 0, 0], [255, 255, 255]])

    colored_mask = map[binary_mask]
    return colored_mask.astype(mask.dtype)


def mask_map(mask):
    labels = np.unique(mask)
    labels = labels[labels != 0].tolist()
    object_images = []

    for value in labels:
        # Создаем маску для текущего объекта
        object_mask = (mask == value).astype(np.uint8)

        # Создаем черное изображение с теми же размерами, что и маска
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Устанавливаем белый цвет для текущего объекта
        colored_mask[object_mask > 0] = [255, 255, 255]

        # Добавляем изображение объекта в список
        object_images.append(colored_mask)

    return object_images
