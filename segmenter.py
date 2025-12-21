import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import DEVICE
from tools.converter import colored_mask_to_indices, merge_masks
from tools.mask_display import visualize_unique_mask
from XMem2.inference.interact.interactive_utils import overlay_davis


class Segmenter:
    def __init__(self, device: str = DEVICE):
        self.device = device
        sam2_checkpoint = 'checkpoints/sam2.1_hiera_large.pt'
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        build = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(build)
        self.embedded = False

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        self.original_image = image
        if self.embedded:
            print('please reset_image')
            return
        self.predictor.set_image(image)
        self.embedded = True

    @torch.no_grad()
    def reset_image(self):
        self.predictor.reset_predictor()
        self.embedded = False

    def predict(self, prompt, mode='point', multimask=True):
        assert self.embedded, 'dont set image'
        assert mode in ['point', 'box', 'both'], 'mode can be point, box or both'

        if mode == 'point':
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt['point_coords'],
                point_labels=prompt['point_labels'],
                multimask_output=multimask,
            )
        elif mode == 'box':
            masks, scores, logits = self.predictor.predict(
                box=prompt['boxes'],
                multimask_output=multimask,
            )
        elif mode == 'both':
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt['point_coords'],
                point_labels=prompt['point_labels'],
                box=prompt['boxes'],
                multimask_output=multimask,
            )
        else:
            raise ValueError('Invalid mode')

        return masks, scores, logits


if __name__ == '__main__':
    path = 'video-test/truck.jpg'
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()

    bboxes = [[476, 166, 578, 320], [8, 252, 99, 401], [106, 335, 317, 425]]
    points = [[531, 230], [45, 321], [226, 360], [194, 313]]

    prompts = {
        'mode': 'point',
        'point_coords': [[531, 230], [45, 321], [[226, 360], [194, 313]]],
        'point_labels': [1, 1, [1, 0]],
    }

    # prompts = {
    #     'mode': 'point',
    #     'point_coords': [[[531, 230], [45, 321]], [226, 360], [194, 313]],
    #     'point_labels': [[1, 0], 1, 1],
    # }

    # prompts = {
    #     'mode': 'box',
    #     'boxes': [
    #         [476, 166, 578, 320],
    #         [8, 252, 99, 401],
    #         [106, 335, 317, 425],
    #         [155, 283, 225, 339],
    #     ],
    # }

    # prompts = {
    #     'mode': 'both',
    #     'point_coords': [[575, 750]],
    #     'point_labels': [0],
    #     'boxes': [[425, 600, 700, 875]],
    # }

    # prompts = {
    #     'mode': 'box',
    #     'boxes': [
    #         [75, 275, 1725, 850],
    #         [425, 600, 700, 875],
    #         [1375, 550, 1650, 800],
    #         [1240, 675, 1400, 750],
    #     ],
    # }

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg = Segmenter()
    seg.set_image(frame)

    maskss = []
    if prompts['mode'] == 'point':
        for point_c, point_l in zip(prompts['point_coords'], prompts['point_labels']):
            prompt = {
                'point_coords': np.array([point_c]),
                'point_labels': np.array([point_l]),
                'boxes': None,
            }
            masks, scores, logits = seg.predict(prompt, prompts['mode'])
            maskss.append(masks[np.argmax(scores)])
    elif prompts['mode'] == 'box':
        for box in prompts['boxes']:
            prompt = {
                'boxes': np.array([box]),
            }
            masks, scores, logits = seg.predict(prompt, prompts['mode'], multimask=True)
            maskss.append(masks[np.argmax(scores)])
    else:
        masks, scores, logits = seg.predict(prompts, prompts['mode'], multimask=False)
        maskss = masks
        print(len(masks))

    print(len(maskss))

    if len(maskss) < 1:
        maskss = []
        for mask in maskss:
            # mask = show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            mask = mask.squeeze(0).astype(np.uint8)
            maskss.append(mask)

    mask, unique_mask = merge_masks(maskss)

    mask_indices, colors = colored_mask_to_indices(unique_mask)
    print('Классы:', np.unique(mask_indices))

    f = overlay_davis(frame, mask_indices)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    mask = visualize_unique_mask(mask_indices)
    cv2.imshow('mask', mask)
    cv2.imshow('overlay', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
