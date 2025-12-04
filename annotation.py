import argparse
from typing import Any

import cv2
import numpy as np

from interactive_video import InteractVideo
from sam_controller import SegmenterController
from tools.data_exporter import get_type_save_annotation
from tools.types import AnnotationInfo, Prompt
from tracker import Tracker
from tracker_core_xmem2 import TrackerCore


def create_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    names_class: list[str],
    type_save: str = 'yolo',
):
    assert len(masks) == len(images)

    send_images = []
    send_masks = []
    for i in range(len(masks)):
        if i % 2 == 0:
            send_images.append(images[i])
            send_masks.append(masks[i])

    saver = get_type_save_annotation(send_images, send_masks, names_class, type_save)
    saver.create_dataset()
    print(f'Saved archive {saver.create_archive()}')


def process_keypoint(
    tracker: Tracker,
    frame_idx: int,
    next_frame_idx: int,
    coords: list[Any],
    frames_path: list[str],
    annotations: list[dict[str, Any]],
) -> None:
    if not coords:
        return
    try:
        frame = np.array(cv2.imread(frames_path[frame_idx]))
        tracker.sam_controller.load_image(frame)
        prompts: Prompt = {
            'mode': 'point',
            'point_coords': coords,
            'point_labels': [1] * len(coords),
        }
        mask = tracker.select_object(prompts)
        tracker.sam_controller.reset_image()
        annotations.append(
            {
                'gap': [frame_idx, next_frame_idx],
                'frame': frame_idx,
                'mask': mask,
            }
        )
    except Exception as e:
        print(f'Error processing keypoint (frame {frame_idx}): {e}')


def process_keypoints(
    tracker: Tracker, results: AnnotationInfo, annotations: list[dict[str, Any]]
) -> None:
    try:
        keypoints_keys = list(results['keypoints'].keys())
        for i in range(len(keypoints_keys) - 1):
            current_frame = keypoints_keys[i]
            next_frame = keypoints_keys[i + 1]
            current_coords = results['keypoints'][current_frame]
            process_keypoint(
                tracker,
                current_frame,
                next_frame,
                current_coords,
                results['frames_path'],
                annotations,
            )
    except Exception as e:
        print(f'Error process_keypoints: {e}')


def get_masks_and_images(
    tracker: Tracker, annotations: list[dict], results: AnnotationInfo
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masks: list[np.ndarray] = []
    images_ann: list[np.ndarray] = []
    for annotation in annotations:
        current_frame, next_frame = annotation['gap']
        if annotation['mask'] is not None:
            frame_sources = results['frames_path'][int(current_frame) : int(next_frame)]
            images = [np.array(cv2.imread(f)) for f in frame_sources]
            mask = tracker.tracking(images, annotation['mask'])
            tracker.tracker.clear_memory()
            masks.extend(mask)
            images_ann.extend(images)
    return images_ann, masks


def main(video_path: str, names_class: list[str], type_save: str):
    video = InteractVideo(video_path)
    video.extract_frames()
    video.collect_keypoints()
    results = video.get_results()

    segmenter_controller = SegmenterController()
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    annotations: list[dict] = []
    process_keypoints(tracker, results, annotations)

    print(f'Count of segments: {len(annotations)}')

    images_ann, masks = get_masks_and_images(tracker, annotations, results)
    create_dataset(images_ann, masks, names_class, type_save)


def parse_args():
    parser = argparse.ArgumentParser(description='Annotation tool')
    parser.add_argument(
        '--video-path',
        type=str,
        help='Path to the video file',
        default='video-test/video.mp4',
    )
    parser.add_argument(
        '--names-class', type=str, nargs='+', help='Names of classes', default=['thing']
    )
    parser.add_argument('--type-save', type=str, help='Type of saving', default='yolo')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    path = args.video_path
    name = args.names_class
    type_save = args.type_save
    main(path, name, type_save)
