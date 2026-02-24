import argparse
import json

from core.video_processor import VideoProcessor
from dataset_export.pipeline import create_dataset
from sam_controller import SamController
from tools.annotations_prompts_types import AnnotationInfo, AnnotationItem
from tracker import Tracker
from xmem2_tracker import TrackerCore


def extract_frames(
    video_path: str,
    frames_to_propagate: int | None = None,
    max_width: int = 1280,
    max_height: int = 720,
) -> list:
    processor = VideoProcessor(video_path, max_width, max_height)
    return processor.extract_all_frames(frames_to_propagate)


def get_info_prompt(
    annotation_item: list[AnnotationItem],
) -> tuple[list[str], list[AnnotationInfo]]:
    class_names: list[str] = []
    annotations_info: list[AnnotationInfo] = []
    class_names_dict: dict[str, int] = {}
    i = 0
    for item in annotation_item:
        class_name = item['class_name']
        if class_name not in class_names_dict:
            class_names_dict[class_name] = i
            class_names.append(class_name)
            i += 1

        prompt = item['prompt']
        if prompt['mode'] not in ['point', 'box', 'both']:
            raise ValueError(f'Invalid mode: {prompt["mode"]}')

        if prompt['mode'] == 'point':
            labels = prompt['point_coords']
        elif prompt['mode'] == 'box':
            labels = prompt['boxes']
        else:
            labels = prompt['boxes']

        annotation_info = AnnotationInfo(
            class_name=class_name,
            prompt=prompt,
            count_objects=len(labels),
            order=class_names_dict[class_name],
        )
        annotations_info.append(annotation_info)

    return class_names, annotations_info


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
    parser.add_argument(
        '--type-save',
        type=str,
        help='Type of saving',
        default='yolo',
    )

    return parser.parse_args()


def main(json_path: str, video_path: str, type_save: str):
    images = extract_frames(video_path)
    json_data = load_json(json_path)

    data = list(map(lambda x: AnnotationItem(**x), json_data))
    class_names, annotations_info = get_info_prompt(data)

    segmenter_controller = SamController()
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    tracker.set_image(images[0])
    mask = tracker.segment_objects(annotations_info)

    masks = tracker.track_objects(images, mask)
    tracker.reset()

    i = 0
    id_map = {}
    for ann in annotations_info:
        key = list(ann.prompt.keys())[1]
        for _ in ann.prompt[f'{key}']:
            mask_id = i + 1
            i += 1
            id_map[mask_id] = {
                'class': ann.class_name,
                'order': ann.order,
                'mask_slice_index': i,
            }

    create_dataset(images, masks, class_names, id_map, type_save)


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
    # from tools.contour_detector import getting_coordinates
    # from XMem2.inference.interact.interactive_utils import overlay_davis

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
