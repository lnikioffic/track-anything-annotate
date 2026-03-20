import cv2
import numpy as np
import psutil
import torch
from tqdm import tqdm

from sam_controller import SamController, SegmentationService
from tools.annotations_prompts_types import AnnotationInfo, Prompt
from tools.converter import colored_mask_to_indices, merge_masks
from tools.mask_display import mask_map
from tools.utils import mask_center
from xmem2_tracker import TrackerCore
import gc

class Tracker:
    def __init__(self, segmenter_controller: SamController, tracker_core: TrackerCore):
        self._segmentation = SegmentationService(segmenter_controller)
        self.tracker = tracker_core
        print(f'used {TrackerCore.name_version}')

    @property
    def sam_controller(self) -> SamController:
        return self._segmentation.sam_controller

    def set_image(self, image: np.ndarray):
        self.sam_controller.set_image(image)

    def reset_image(self):
        self.sam_controller.reset_image()

    def segment_objects(self, annotations_info: list[AnnotationInfo]) -> np.ndarray:
        return self._segmentation.segment_objects(annotations_info)

    def sam_ref(self, frame, mask):
        self.sam_controller.reset_image()
        centers = []
        for m in mask_map(mask):
            centers.append(mask_center(m))

        print(centers)
        prompts = []
        for center in centers:
            prompt = {
                'point_coords': np.array([center]),
                'point_labels': np.array([1]),
            }
            prompts.append((prompt, False))

        self.sam_controller.set_image(frame)
        results = self.sam_controller.predict_from_prompts('point', prompts)
        results = [result[np.argmax(scores)] for result, scores, logits in results]
        _, unique_mask = merge_masks(results)
        mask_indices, _ = colored_mask_to_indices(unique_mask)
        self.sam_controller.reset_image()
        return mask_indices

    def track_objects(
        self,
        frames: list[np.ndarray],
        template_mask: np.ndarray,
        exhaustive: bool = False,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []

        for i in tqdm(range(len(frames)), desc='Tracking'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break

            is_last_frame = i == len(frames) - 1
            """
             TODO: улучшение точности
                - надо проверять сколько масок в трекере
                - смотреть сколько объектов обнаруживается
                - если они не совпадают добавлять к новым маскам маску из трекера
            """
            # if i in (0, 5, 10):
            #     if masks:
            #         m = masks[-1].copy()
            #         mask_new = self.sam_ref(frames[i], m)
            #         template_mask = mask_new
            #         exhaustive = True
            if i % 10 == 0:
                if self.tracker.is_cuda:
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    gc.collect()
            if i == 0:
                mask = self.tracker.track(frames[i], template_mask, exhaustive, end=is_last_frame)
                masks.append(mask)
            else:
                mask = self.tracker.track(frames[i], end=is_last_frame)
                masks.append(mask)
        return masks

    def reset(self):
        self.reset_image()
        self.tracker.clear_memory()

    # def tracking_cut(
    #     self,
    #     frames: list[np.ndarray],
    #     templates_masks: dict[str, np.ndarray],
    #     exhaustive: bool = False,
    # ):
    #     masks = []
    #     for i in tqdm(range(len(frames)), desc='Tracking_cut'):
    #         current_memory_usage = psutil.virtual_memory().percent
    #         if current_memory_usage > 90:
    #             break

    #         if str(i) in templates_masks:
    #             template_mask = templates_masks[str(i)]

    #         if i == 0 and str(i) in templates_masks:
    #             mask = self.tracker.track(frames[i], template_mask, exhaustive)
    #             masks.append(mask)
    #         else:
    #             mask = self.tracker.track(frames[i])
    #             masks.append(mask)

    #         if len(templates_masks) > 1:
    #             exhaustive = True

    #     return masks


if __name__ == '__main__':
    from interactive_video import InteractVideo
    from segmenter import Sam2ModelSize, Segmenter
    from tools.overlay_image import painter_borders
    from XMem2.inference.interact.interactive_utils import overlay_davis

    path = 'video-test/video.mp4'
    key_interval = 3
    video = InteractVideo(path, key_interval)
    video.extract_frames()
    video.collect_keypoints()

    model_size = Sam2ModelSize.Large
    segmenter = Segmenter(model_size)
    segmenter_controller = SamController(segmenter)
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    frames = video.get_frames()

    prompts: Prompt = {
        'mode': 'point',
        'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
        'point_labels': [1, 1, 1, 1],
    }

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': results['keypoints'][list(results['keypoints'].keys())[0]],
    #     'point_labels': [1, 1],
    # }

    tracker.set_image(frames[0])
    mask = tracker.segment_objects(prompts)
    image_m = overlay_davis(frames[0], mask)
    cv2.imshow('image', image_m)
    cv2.imshow('imageza;epa', frames[63])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    masks = tracker.track_objects(frames, mask)

    filename = 'helmet_border.mp4'
    video_info = video.video_processor.get_video_info()
    frame_size = (video_info.width, video_info.height)
    output = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*'XVID'), video_info.fps, frame_size)
    for frame, mask in zip(frames, masks):
        f = painter_borders(frame, mask)
        # f = overlay_davis(frame, mask)
        output.write(f)
    output.release()
    cv2.destroyAllWindows()

    print(f'Видео записано в файл: {filename}')
