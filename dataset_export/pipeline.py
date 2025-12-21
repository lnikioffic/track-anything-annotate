import numpy as np

from dataset_export.factory import get_type_save_annotation


def create_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    class_names: list[str],
    id_map: dict,
    type_save: str = 'yolo',
):
    assert len(masks) == len(images)

    send_images = []
    send_masks = []
    for i in range(len(masks)):
        if i % 2 == 0:
            send_images.append(images[i])
            send_masks.append(masks[i])

    saver = get_type_save_annotation(send_images, send_masks, class_names, type_save)
    saver.save(id_map)
    print(f'Saved archive {saver.archive()}')
