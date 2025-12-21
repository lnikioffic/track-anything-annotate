import argparse
from typing import Any

import cv2
import numpy as np

from dataset_export.pipeline import create_dataset
from interactive_video import InteractVideo
from sam_controller import SamController
from tools.annotations_prompts_types import (
    AnnotationInfo,
    AnnotationVideoInfo,
    PointPrompt,
)
from tracker import Tracker
from xmem2_tracker import TrackerCore


def create_track_seed_from_points(
    tracker: Tracker,
    frame_idx: int,
    next_frame_idx: int,
    point_coords: list[Any],
    frames_path: list[str],
):
    if not point_coords:
        return None

    frame = np.array(cv2.imread(frames_path[frame_idx]))

    prompts: PointPrompt = {
        'mode': 'point',
        'point_coords': point_coords,
        'point_labels': [1] * len(point_coords),
    }

    annotation_info = AnnotationInfo(
        class_name='class_name',
        prompt=prompts,
        count_objects=len(point_coords),
        order=0,
    )

    tracker.set_image(frame)
    mask = tracker.segment_objects([annotation_info])
    tracker.reset()

    track_seed = {
        'start_frame': frame_idx,
        'end_frame': next_frame_idx,
        'mask': mask,
    }

    return track_seed, annotation_info


def create_track_seeds_from_keyframes(
    tracker: Tracker, video_info: AnnotationVideoInfo
):
    keypoints_keys = list(video_info['keypoints'].keys())
    track_seeds = []
    annotations_info = []
    for i in range(len(keypoints_keys) - 1):
        start_frame = keypoints_keys[i]
        end_frame = keypoints_keys[i + 1]
        point_coords = video_info['keypoints'][start_frame]

        result = create_track_seed_from_points(
            tracker,
            start_frame,
            end_frame,
            point_coords,
            video_info['frames_path'],
        )
        if result is None:
            continue

        print(f'{start_frame=}, {end_frame=}')
        track_seed, annotation_info = result

        track_seeds.append(track_seed)
        annotations_info.append(annotation_info)
    return track_seeds, annotations_info


def track_masks_from_seeds(
    tracker: Tracker, track_seeds: list[dict], video_info: AnnotationVideoInfo
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tracked_frames: list[np.ndarray] = []
    tracked_masks: list[np.ndarray] = []
    for seed in track_seeds:
        start_frame, end_frame = seed['start_frame'], seed['end_frame']
        if seed['mask'] is not None:
            frame_sources = video_info['frames_path'][int(start_frame) : int(end_frame)]
            images = [np.array(cv2.imread(f)) for f in frame_sources]

            masks = tracker.track_objects(images, seed['mask'])

            tracker.reset()
            tracked_masks.extend(masks)
            tracked_frames.extend(images)
    return tracked_frames, tracked_masks


def main(video_path: str, class_names: list[str], type_save: str):
    video = InteractVideo(video_path)
    video.extract_frames()
    video.collect_keypoints()

    video_info = video.get_results()

    segmenter_controller = SamController()
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    track_seeds: list[dict] = []
    annotations_info = []
    track_seed, info = create_track_seeds_from_keyframes(tracker, video_info)

    track_seeds.extend(track_seed)
    annotations_info.extend(info)
    print(f'Count of segments: {len(track_seeds)}')

    images_ann, masks = track_masks_from_seeds(tracker, track_seeds, video_info)
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
    create_dataset(images_ann, masks, class_names, id_map, type_save)


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
    names = args.names_class
    type_save = args.type_save
    main(path, names, type_save)
