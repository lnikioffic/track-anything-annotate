import argparse
import json

from core.video_processor import VideoProcessor
from dataset_export.pipeline import create_dataset
from sam_controller import SamController
from segmenter import Sam2ModelSize, Segmenter
from tools.annotations_prompts_types import AnnotationInfo, AnnotationItem
from tracker import Tracker
from xmem2_tracker import TrackerCore


def extract_frames(
    video_path: str,
    frames_to_propagate: int | None = None,
    max_width: int = 1280,
    max_height: int = 720,
):
    processor = VideoProcessor(video_path, max_width, max_height)
    video_info = processor.get_video_info()
    frames = processor.extract_all_frames(frames_to_propagate)
    return frames, video_info


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
    images, video_info = extract_frames(video_path)
    json_data = load_json(json_path)

    data = list(map(lambda x: AnnotationItem(**x), json_data))
    class_names, annotations_info = get_info_prompt(data)

    segmenter = Segmenter(Sam2ModelSize.Large)
    segmenter_controller = SamController(segmenter)
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
    return images, masks, video_info


if __name__ == '__main__':
    args = parse_args()
    json_path = args.json_path
    video_path = args.video_path
    type_save = args.type_save
    images, masks, video_info = main(json_path, video_path, type_save)

    import cv2

    from tools.overlay_image import painter_borders
    from XMem2.inference.interact.interactive_utils import overlay_davis

    filename_border = 'test_border.mp4'
    filename_overlay = 'test_overlay.mp4'
    frame_size = (video_info.width, video_info.height)
    output_overlay = cv2.VideoWriter(
        filename_overlay,
        cv2.VideoWriter.fourcc(*'mp4v'),
        video_info.fps,
        frame_size,
    )
    output_border = cv2.VideoWriter(
        filename_border,
        cv2.VideoWriter.fourcc(*'mp4v'),
        video_info.fps,
        frame_size,
    )

    for frame, mask in zip(images, masks):
        overlay = overlay_davis(frame, mask)
        border = painter_borders(frame, mask)
        output_overlay.write(overlay)
        output_border.write(border)
    output_overlay.release()
    output_border.release()
    cv2.destroyAllWindows()
