import cv2
import numpy as np
import psutil
import torch
from XMem2.inference.inference_core import InferenceCore
from XMem2.model.network import XMem
from XMem2.inference.data.mask_mapper import MaskMapper
from config import XMEM_CONFIG, DEVICE
from torchvision import transforms
from XMem2.util.range_transform import im_normalization
from XMem2.inference.interact.interactive_utils import overlay_davis
from segmenter import Segmenter
from tools.mask_display import visualize_wb_mask, mask_map
from tools.contour_detector import getting_coordinates
from tools.converter import merge_masks, extract_color_regions


class TrackerCore:

    name_version = 'XMem2'

    def __init__(self, device: str = DEVICE):
        self.device = device
        if self.device.lower() != 'cpu':
            self.network = XMem(XMEM_CONFIG, 'checkpoints/XMem.pth').eval().to('cuda')
        else:
            self.network = XMem(
                XMEM_CONFIG, 'checkpoints/XMem.pth', map_location='cpu'
            ).eval()
        self.processor = InferenceCore(self.network, XMEM_CONFIG)

        self.im_transform = transforms.Compose(
            [transforms.ToTensor(), im_normalization]
        )
        self.mapper = MaskMapper()

    @torch.no_grad()
    def track(
        self, frame: np.ndarray, mask_segmet: np.ndarray | None = None, exhaustive=False
    ):
        if mask_segmet is not None:
            mask, labels = self.mapper.convert_mask(mask_segmet, exhaustive)
            mask = torch.Tensor(mask).to(self.device)
            self.processor.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None

        frame_tensor = self.im_transform(frame).to(self.device)
        probs = self.processor.step(frame_tensor, mask, labels)

        out_mask = torch.argmax(probs, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
        final_mask = np.zeros_like(out_mask)

        # map back
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        return final_mask

    @torch.no_grad()
    def clear_memory(self):
        self.processor.clear_memory()
        self.mapper.clear_lables()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()

    bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
    points = [[531, 230], [45, 321], [226, 360], [194, 313]]
    
    # первый енот [(487, 176, 574, 318)] второй самый левый кот [(11, 267, 111, 415)] третий передний кот [(98, 300, 321, 443)]
    # четвертый задний кот [(158, 292, 224, 343)]
    mode = 'point'
    prompts = {
        'point_coords': np.array([[531, 230], [45, 321], [226, 360], [194, 313]]),
        'point_labels': np.array([1] * len(points)),
    }

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg = Segmenter()
    seg.set_image(frame)

    maskss = []
    for point in points:
        prompts = {
            'point_coords': np.array([point]),
            'point_labels': np.array([1]),
        }
        masks, scores, logits = seg.predict(prompts, mode)
        maskss.append(masks[np.argmax(scores)])
    _, unique_mask = merge_masks(maskss)
    mask_indices, colors = extract_color_regions(unique_mask)
    
    from tools.mask_display import mask_map
    from tools.contour_detector import get_filtered_bboxes
    
    def check_coords(mask):
        coords = []
        for obj in mask_map(mask):
            m = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            coords.extend(get_filtered_bboxes(m, min_area_ratio=0.001))
        return coords
    
    coords = check_coords(unique_mask)
    
    print(coords)
    print('Классы:', np.unique(mask_indices))

    masks = []
    images = []
    traker = TrackerCore()
    frames_to_propagate = 200
    current_frame_index = 0
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        current_memory_usage = psutil.virtual_memory().percent
        # print(current_memory_usage)
        if current_memory_usage > 90:
            break
        ret, frame_v = cap.read()
        if not ret:
            break
        if current_frame_index > frames_to_propagate:
            break

        if current_frame_index == 0:
            mask = traker.track(frame_v, mask_indices)
            masks.append(mask)
            images.append(frame_v)
        else:
            mask = traker.track(frame_v)
            masks.append(mask)
            images.append(frame_v)

        current_frame_index += 1
    video.release()
    
    coords = check_coords(masks[1])
    print(coords)
    
    coords = check_coords(masks[10])
    print(coords)
    
    coords = check_coords(masks[40])
    print(coords)
    
    im3 = visualize_wb_mask(masks[200])
    ima = images[200].copy()
    for m in mask_map(masks[200]):
        for box in getting_coordinates(m):
            (x, y, w, h) = [v for v in box]
            cv2.rectangle(ima, (x, y), (x + w, y + h), (0, 255, 0), 2)
    image_m = overlay_davis(images[200], masks[200])
    cv2.imshow('image_m200', image_m)
    cv2.imshow('ima_rect', ima)
    cv2.imshow('im2', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
