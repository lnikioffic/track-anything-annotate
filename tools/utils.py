import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import area_opening, closing, disk


def clean_mask(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    mask_bool = mask.astype(bool)

    # удалить маленькие компоненты
    mask_bool = area_opening(mask_bool, area_threshold=min_size)

    # закрыть дырки на границе
    mask_bool = closing(mask_bool, disk(3))

    # заполнить внутренние дырки
    mask_bool = binary_fill_holes(mask_bool)

    return mask_bool.astype(np.uint8)


def mask_center(mask):

    if mask.ndim == 3:
        mask = mask[:, :, 0]  # берем один канал

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None
    return [int(xs.mean()), int(ys.mean())]


def mask_to_box(mask):

    ys, xs = np.where(mask > 0)

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    return [x1, y1, x2, y2]
