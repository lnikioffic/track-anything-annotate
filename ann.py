import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np

from sam_controller import SegmenterController
from tools.converter import extract_color_regions, merge_masks
from tools.types import AnnotationItem, Prompt
from XMem2.inference.interact.interactive_utils import overlay_davis


@dataclass
class AnnotationInfo:
    class_name: str
    prompt: Prompt
    count_objects: int = 0
    order: int = 0


def merge_similar_dicts(dicts_list):
    if not dicts_list:
        return {}

    result = {}

    modes = [d['mode'] for d in dicts_list]
    if not all(m == modes[0] for m in modes):
        raise ValueError('Все режимы (mode) должны быть одинаковыми')

    result['mode'] = modes[0]

    for key in dicts_list[0].keys():
        if key == 'mode':
            continue
        values = [d[key] for d in dicts_list]

        # Если все значения - списки, объединяем их
        if all(isinstance(v, list) for v in values):
            merged = []
            for v in values:
                merged.extend(v)
            result[key] = merged
        else:
            # Иначе оставляем как список значений
            result[key] = values

    return result


def segmentation(annotations_info: list[AnnotationInfo], image):
    segmenter_controller = SegmenterController()
    segmenter_controller.load_image(image)
    masks = []
    for ann in annotations_info:
        print(ann.prompt)
        mode, processed_prompts = segmenter_controller.create_prompts(ann.prompt)
        results = segmenter_controller.predict_from_prompts(mode, processed_prompts)
        results = [result[np.argmax(scores)] for result, scores, logits in results]
        masks.extend(results)
    return masks


def get_info_prompt(annotation_item: list[AnnotationItem]):
    class_name = []
    join_promt = {}
    dicts_list = []
    annotations_info = []
    order = 1
    for item in annotation_item:
        class_name.append(item['class_name'])
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

    join_promt = 2  # merge_similar_dicts(dicts_list)
    return class_name, annotations_info, join_promt


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


if __name__ == '__main__':
    args = parse_args()
    json_data = load_json(args.json_path)
    video = cv2.VideoCapture(args.video_path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()
    data = list(map(lambda x: AnnotationItem(**x), json_data))
    class_name, annotations_info, join_promt = get_info_prompt(data)
    print(annotations_info)
    masks = segmentation(annotations_info, frame_cop)
    _, unique_mask = merge_masks(masks)
    mask_indices, colors = extract_color_regions(unique_mask)

    from tools.contour_detector import get_filtered_bboxes
    from tools.mask_display import mask_map

    def check_coords(mask):
        coords = []
        for obj in mask_map(mask):
            m = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            coords.extend(get_filtered_bboxes(m, min_area_ratio=0.001))
        return coords

    coords = check_coords(unique_mask)

    print(coords)

    print(class_name)
    print(join_promt)
    f = overlay_davis(frame, mask_indices)
    cv2.imshow('overlay', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
