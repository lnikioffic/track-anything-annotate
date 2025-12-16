import argparse
import json

import cv2
import numpy as np
import progressbar
import psutil
from tqdm import tqdm

from sam_controller import SegmenterController
from tools.annotations_prompts_types import AnnotationInfo, AnnotationItem
from tools.contour_detector import getting_coordinates
from tools.converter import extract_color_regions, merge_masks
from tools.data_exporter import create_dataset
from tracker_core_xmem2 import TrackerCore
from XMem2.inference.interact.interactive_utils import overlay_davis


def extract_frames(
    video_path: str,
    frames_to_propagate: int = 0,
    max_width: int = 1280,
    max_height: int = 720,
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames_to_propagate <= 0 or frames_to_propagate > count_frames:
        frames_to_propagate = count_frames

    frame_index = 0
    bar = progressbar.ProgressBar(max_value=frames_to_propagate)
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= frames_to_propagate:
            break
        if max_width is not None or max_height is not None:
            h, w = frame.shape[:2]
            ratio_w = max_width / w if max_width else float('inf')
            ratio_h = max_height / h if max_height else float('inf')
            ratio = min(ratio_w, ratio_h, 1.0)

            if ratio < 1.0:
                new_size = (int(w * ratio), int(h * ratio))
                frame = cv2.resize(frame, new_size)
        images.append(frame)
        frame_index += 1
        bar.update(frame_index)
    bar.finish()
    cap.release()
    return images


def segmentation(annotations_info: list[AnnotationInfo], image):
    segmenter_controller = SegmenterController()
    segmenter_controller.load_image(image)
    masks = []
    for ann in annotations_info:
        mode, processed_prompts = segmenter_controller.create_prompts(ann.prompt)
        results = segmenter_controller.predict_from_prompts(mode, processed_prompts)
        results = [result[np.argmax(scores)] for result, scores, logits in results]
        masks.extend(results)
    return masks


def tracking(images, template_mask):
    tracker = TrackerCore()
    masks = []
    for i in tqdm(range(len(images)), desc='Tracking'):
        current_memory_usage = psutil.virtual_memory().percent
        if current_memory_usage > 90:
            break
        if i == 0:
            mask = tracker.track(images[i], template_mask)
            masks.append(mask)
        else:
            mask = tracker.track(images[i])
            masks.append(mask)
    return masks


def get_info_prompt(annotation_item: list[AnnotationItem]):
    names_class = []
    dicts_list = []
    annotations_info = []
    order = 0
    for item in annotation_item:
        names_class.append(item['class_name'])
        prompt = item['prompt']
        if prompt['mode'] not in ['point', 'box', 'both']:
            raise ValueError(f'Invalid mode: {prompt["mode"]}')

        if prompt['mode'] == 'point':
            labels = prompt['point_coords']
            info = AnnotationInfo(
                class_name=item['class_name'],
                prompt=prompt,
                count_objects=len(labels),
                order=order,
            )
            dicts_list.append(prompt)
            annotations_info.append(info)
        elif prompt['mode'] == 'box':
            labels = prompt['boxes']
            info = AnnotationInfo(
                class_name=item['class_name'],
                prompt=prompt,
                count_objects=len(labels),
                order=order,
            )
            dicts_list.append(prompt)
            annotations_info.append(info)
        order += 1

    return names_class, annotations_info


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Annotate')
    parser.add_argument(
        '--video-path',
        type=str,
        help='Path to the video file',
        default='video-test/video.mp4',
    )
    parser.add_argument(
        '--json-path',
        type=str,
        help='Path to the json file',
        default='video-test/video.json',
    )
    parser.add_argument('--type-save', type=str, help='Type of saving', default='yolo')

    return parser.parse_args()


def main(json_path, video_path, type_save):
    images = extract_frames(video_path, 100)
    json_data = load_json(json_path)

    data = list(map(lambda x: AnnotationItem(**x), json_data))
    names_class, annotations_info = get_info_prompt(data)
    masks = segmentation(annotations_info, images[0])
    _, unique_mask = merge_masks(masks)
    mask_indices, colors = extract_color_regions(unique_mask)

    masks = tracking(images, mask_indices)

    i = 0
    id_map = {}
    for ann in annotations_info:
        key = list(ann.prompt.keys())[1]
        for p in ann.prompt[f'{key}']:
            mask_id = i + 1
            i += 1
            id_map[mask_id] = {
                'class': ann.class_name,
                'order': ann.order,
                'mask_slice_index': i,
            }
    
    print(id_map)
    create_dataset(images, masks, names_class, id_map, type_save)


if __name__ == '__main__':
    args = parse_args()
    json_path = args.json_path
    video_path = args.video_path
    type_save = args.type_save
    main(json_path, video_path, type_save)

    # json_data = load_json(json_path)

    # images = extract_frames(video_path)

    # image_copy = images[0].copy()
    # data = list(map(lambda x: AnnotationItem(**x), json_data))
    # class_name, annotations_info = get_info_prompt(data)
    # masks = segmentation(annotations_info, image_copy)
    # _, unique_mask = merge_masks(masks)
    # mask_indices, colors = extract_color_regions(unique_mask)

    # from tools.contour_detector import get_filtered_bboxes
    # from tools.mask_display import mask_map

    # def check_coords(mask):
    #     coords = []
    #     for obj in mask_map(mask):
    #         m = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    #         coords.extend(get_filtered_bboxes(m, min_area_ratio=0.001))
    #     return coords

    # coords = check_coords(unique_mask)
    # print(coords)
    # id_map = {}
    # i = 0
    # for ann in annotations_info:
    #     key = list(ann.prompt.keys())[1]
    #     for p in ann.prompt[f'{key}']:
    #         mask_id = i + 1  # ID в unique_mask начинается с 1
    #         i += 1
    #         id_map[mask_id] = {
    #             'class': ann.class_name,
    #             'order': ann.order,
    #             'mask_slice_index': i,  # Индекс для доступа к masks_array[i]
    #         }

    # print(id_map)

    # def check_coords_and_match_class(mask, id_mapping):
    #     img_height = mask.shape[0]
    #     img_width = mask.shape[1]
    #     with open('output.txt', 'w', encoding='utf-8') as file:
    #         result_objects = []
    #         for mask_id, mask in enumerate(mask_map(mask), 1):
    #             if mask_id not in id_mapping:
    #                 continue

    #             bbox = getting_coordinates(mask)

    #             obj_info = id_mapping[mask_id]
    #             result_objects.append(
    #                 {
    #                     'mask_id': mask_id,
    #                     'class_name': obj_info['class'],
    #                     'bbox': bbox,
    #                     'order': obj_info['order'],
    #                 }
    #             )
    #         for obj in result_objects:
    #             x, y = obj['bbox'][0][0], obj['bbox'][0][1]
    #             w, h = obj['bbox'][0][2], obj['bbox'][0][3]

    #             x_center = x + w / 2
    #             y_center = y + h / 2

    #             norm_xc = x_center / img_width
    #             norm_yc = y_center / img_height
    #             norm_width = w / img_width
    #             norm_height = h / img_height

    #             file.write(
    #                 f'{obj["order"]} {norm_xc} {norm_yc} {norm_width} {norm_height}\n'
    #             )

    #     return result_objects

    # objects_data = check_coords_and_match_class(unique_mask, id_map)

    # print('\n--- Final Objects Data ---')
    # for obj in objects_data:
    #     print(obj)

    # f = overlay_davis(images[0], mask_indices)
    # cv2.imshow('overlay', f)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
