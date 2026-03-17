import cv2
import numpy as np


def merge_masks(masks):
    if isinstance(masks, list):
        masks = np.stack(masks)

    # Убедимся, что маски имеют корректную форму
    if masks.ndim != 3:
        raise ValueError('Маски должны быть представлены в виде массива размерности (N, H, W).')
    N, H, W = masks.shape

    # Создаем пустое цветное изображение
    mask_colored = np.zeros((H, W, 3), dtype=np.uint8)
    # Создаем список случайных цветов для каждой маски
    colors = np.random.randint(0, 255, size=(N, 3), dtype=np.uint8)
    # Создаем уникальную индексированную маску
    unique_mask = np.zeros((H, W), dtype=np.uint8)

    # binary_mask = np.any(masks > 0, axis=0).astype(np.uint8) * 255

    for i in range(N):
        mask_bool = masks[i] > 0
        # Если маска пустая, пропускаем итерацию
        if not np.any(mask_bool):
            continue

        idx = i + 1
        # Заполняем индексную маску (последняя маска в списке будет "сверху")
        unique_mask[mask_bool] = idx
        # Заполняем цветом напрямую (это в разы быстрее cv2.addWeighted)
        mask_colored[mask_bool] = colors[i]

    return mask_colored, unique_mask


def colored_mask_to_indices(mask):
    img_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    colors, inverse = np.unique(img_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    mask_indices = inverse.reshape(img_rgb.shape[:2])
    return mask_indices, colors
