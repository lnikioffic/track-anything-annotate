from tools.contour_detector import getting_coordinates
from tools.mask_display import mask_map


def extract_objects(mask_unique, id_mapping):
    objects = []
    for mask_id, mask in enumerate(mask_map(mask_unique), 1):
        if mask_id not in id_mapping:
            continue
        coords_list = getting_coordinates(mask)

        if not coords_list:
            continue

        bbox = coords_list[0]
        obj_info = id_mapping[mask_id]
        objects.append(
            {
                'mask_id': mask_id,
                'class_name': obj_info['class'],
                'bbox': bbox,
                'order': obj_info['order'],
            }
        )

    return objects
