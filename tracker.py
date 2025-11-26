import cv2
import numpy as np
import psutil
from tqdm import tqdm

from tools.converter import merge_masks, extract_color_regions
from tracker_core_xmem2 import TrackerCore
from tools.overlay_image import painter_borders
from XMem2.inference.interact.interactive_utils import overlay_davis
from sam_controller import SegmenterController
from interactive_video import InteractVideo


class Tracker:
    def __init__(
        self, segmenter_controller: SegmenterController, tracker_core: TrackerCore
    ):
        self.sam_controller = segmenter_controller
        self.tracker = tracker_core
        print(f'used {TrackerCore.name_version}')

    def select_object(self, prompts: dict) -> np.ndarray:
        # maskss = []
        # for point in points:
        #     prompts = {
        #         'point_coords': np.array([point]),
        #         'point_labels': np.array([1]),
        #     }
        #     masks, scores, logits = self.segmenter.predict(prompts, 'point')
        #     maskss.append(masks[np.argmax(scores)])
        results = self.sam_controller.predict_from_prompts(prompts)
        results_masks = [
            result[np.argmax(scores)] for result, scores, logits in results
        ]
        mask, unique_mask = merge_masks(results_masks)
        mask_indices, colors = extract_color_regions(unique_mask)
        print('Классы:', np.unique(mask_indices))
        return mask_indices

    def tracking(
        self,
        frames: list[np.ndarray],
        template_mask: np.ndarray,
        exhaustive: bool = False,
    ) -> list:
        masks = []
        for i in tqdm(range(len(frames)), desc='Tracking'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break
            """
             TODO: улучшение точности
                - надо проверять сколько масок в трекере
                - смотреть сколько объектов обнаруживается
                - если они не совпадают добавлять к новым маскам маску из трекера
            """
            if i == 0:
                mask = self.tracker.track(frames[i], template_mask, exhaustive)
                masks.append(mask)
            else:
                mask = self.tracker.track(frames[i])
                masks.append(mask)
        return masks

    def tracking_cut(
        self,
        frames: list[np.ndarray],
        templates_masks: dict[str, np.ndarray],
        exhaustive: bool = False,
    ):
        masks = []
        for i in tqdm(range(len(frames)), desc='Tracking_cut'):
            current_memory_usage = psutil.virtual_memory().percent
            if current_memory_usage > 90:
                break

            if str(i) in templates_masks:
                template_mask = templates_masks[str(i)]

            if i == 0 and str(i) in templates_masks:
                mask = self.tracker.track(frames[i], template_mask, exhaustive)
                masks.append(mask)
            else:
                mask = self.tracker.track(frames[i])
                masks.append(mask)

            if len(templates_masks) > 1:
                exhaustive = True

        return masks


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    key_interval = 3
    video = InteractVideo(path, key_interval)
    video.extract_frames()
    video.collect_keypoints()
    results = video.get_results()

    segmenter_controller = SegmenterController()
    tracker_core = TrackerCore()
    tracker = Tracker(segmenter_controller, tracker_core)

    frame_sources = results['frames_path']
    frames = [cv2.imread(f) for f in frame_sources]
    
    prompts = {
        'mode': 'point',
        'point_coords': [[531, 230], [45, 321], [226, 360], [194, 313]],
        'point_labels': [1, 1, 1, 1],
    }

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': results['keypoints'][list(results['keypoints'].keys())[0]],
    #     'point_labels': [1, 1],
    # }

    tracker.sam_controller.load_image(frames[0])
    mask = tracker.select_object(prompts)
    image_m = overlay_davis(frames[0], mask)
    cv2.imshow('image', image_m)
    cv2.imshow('imageza;epa', frames[63])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    masks = tracker.tracking(frames, mask)

    # result = []
    # print(len(results['keypoints']))
    # if len(results['keypoints']) == 1:
    #     current_frame = list(results['keypoints'].keys())[0]
    #     next_frame = len(results["frames"])
    #     current_coords = results['keypoints'][current_frame]

    #     if current_coords:
    #         tracker.sam_controller.load_image(results['frames'][int(current_frame)])
    #         prompts = {
    #             'mode': 'point',
    #             'point_coords': current_coords,
    #             'point_labels': [1] * len(current_coords),
    #         }
    #         mask = tracker.select_object(prompts)
    #         tracker.sam_controller.reset_image()
    #         result.append(
    #             {
    #                 'gap': [current_frame, next_frame],
    #                 'frame': current_frame,
    #                 'mask': mask,
    #             }
    #         )

    # for i in range(len(results['keypoints']) - 1):
    #     current_frame = list(results['keypoints'].keys())[i]
    #     next_frame = list(results['keypoints'].keys())[i + 1]
    #     current_coords = results['keypoints'][current_frame]

    #     if current_coords:
    #         tracker.sam_controller.load_image(results['frames'][int(current_frame)])
    #         prompts = {
    #             'mode': 'point',
    #             'point_coords': current_coords,
    #             'point_labels': [1] * len(current_coords),
    #         }
    #         mask = tracker.select_object(prompts)
    #         tracker.sam_controller.reset_image()
    #         result.append(
    #             {
    #                 'gap': [current_frame, next_frame],
    #                 'frame': current_frame,
    #                 'mask': mask,
    #             }
    #         )

    # print(len(result))
    # masks = []
    # for res in result:
    #     current_frame, next_frame = res['gap']
    #     if res['mask'] is not None:
    #         print(current_frame, next_frame)
    #         images = results['frames'][int(current_frame) : int(next_frame)]
    #         mask = tracker.tracking(images, res['mask'])
    #         tracker.tracker.clear_memory()
    #         masks += mask
    # else:
    #     print(current_frame, next_frame)
    #     m = []
    #     for _ in range(current_frame, next_frame):
    #         height, width, _ = frames[current_frame].shape
    #         binary_mask = np.zeros((height, width), dtype=np.uint8)
    #         binary_mask[:, :] = 1
    #         m.append(binary_mask)
    #     masks += m

    filename = 'helmet_border.mp4'
    output = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*'XVID'), video.fps, video.frame_size
    )
    for frame, mask in zip(frames, masks):
        f = painter_borders(frame, mask)
        # f = overlay_davis(frame, mask)
        output.write(f)
    # Освобождаем ресурсы
    output.release()
    cv2.destroyAllWindows()

    print(f'Видео записано в файл: {filename}')
