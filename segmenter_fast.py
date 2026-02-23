import cv2
import numpy as np
import torch
from fastsam import FastSAM, FastSAMPredictor, FastSAMPrompt
from fastsam.utils import convert_box_xywh_to_xyxy

from config import DEVICE
from tools.contour_detector import getting_coordinates
from tools.mask_display import visualize_unique_mask, visualize_wb_mask


class Segmenter:
    def __init__(self, model_path: str, device: str = DEVICE) -> None:
        self.device = device
        self.model = FastSAM(model_path)
        overrides = self.model.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(device=self.device, retina_masks=True, imgsz=1024, conf=0.7, iou=0.9)
        overrides['mode'] = 'predict'
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = False
        self.model.predictor = FastSAMPredictor(overrides=overrides)
        self.model.predictor.setup_model(model=self.model.model, verbose=False)

    @property
    def prompt(self):
        return self.prompt_process

    @prompt.setter
    @torch.no_grad()
    def prompt(self, image: np.ndarray):
        everything_results = self.model.predictor(image)
        self.prompt_process = FastSAMPrompt(image, everything_results, device=self.device)

    def get_mask_by_box_prompt(self, bboxes: list[int]):
        box_prompt = [convert_box_xywh_to_xyxy(box) for box in bboxes]
        self.mask = self.prompt_process.box_prompt(bboxes=box_prompt)

    def get_mask_by_point_prompt(self, points: list[int]):
        labels = [1 for _ in range(len(points))]
        self.mask = self.prompt_process.point_prompt(points, labels)

    def convert_mask_to_color(self):
        mask = self.mask
        mask = np.uint8(mask) * 255
        mask_end = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        colors = [tuple(list(np.random.randint(0, 255, size=3))) for _ in range(len(mask))]

        for i in range(len(mask)):
            color_mask = np.zeros_like(mask_end)
            color_mask[mask[i] > 0] = colors[i]
            mask_end = cv2.addWeighted(mask_end, 1, color_mask, 1, 0.0)

        unique_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)

        for i in range(mask.shape[0]):
            unique_mask[mask[i] > 0] = i + 1

        return mask_end, unique_mask

    def annotated_frame(self) -> np.ndarray:
        annotated_frame = self.prompt_process.plot_to_result(
            annotations=self.mask,
            withContours=False,
            better_quality=True,
        )
        return annotated_frame

    @torch.no_grad()
    def clear_memory(self):
        torch.cuda.empty_cache()


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Если нажата левая кнопка мыши
        points = []
        points.append((x, y))  # Добавляем координаты точки в список
        print(f'Координаты: ({x}, {y})')  # Выводим координаты в консоль


if __name__ == '__main__':
    path = 'video-test/video.mp4'
    video = cv2.VideoCapture(path)
    ret, frame = video.read()
    frame_cop = frame.copy()
    video.release()
    bboxes = [(476, 166, 102, 154), (8, 252, 91, 149), (106, 335, 211, 90)]
    points = [[531, 230], [45, 321], [226, 360]]
    # cv2.namedWindow("Image")
    # cv2.setMouseCallback("Image", get_coordinates)
    # cv2.imshow("Image", frame_cop)
    # for _ in range(2):
    #     bbox = cv2.selectROI(frame_cop)
    #     bboxes.append(bbox)
    seg = Segmenter('models/FastSAM-x.pt')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg.prompt = frame
    seg.get_mask_by_box_prompt(bboxes)
    # seg.get_mask_by_point_prompt(points)
    mask, unique_mask = seg.convert_mask_to_color()
    print(np.unique(unique_mask))
    mask = visualize_wb_mask(unique_mask)
    for box in getting_coordinates(mask):
        (x, y, w, h) = [v for v in box]
        cv2.rectangle(frame_cop, (x, y), (x + w, y + h), (0, 255, 0), 2)
    im = seg.annotated_frame()
    cv2.imshow('frame', frame_cop)
    cv2.imshow('mask', mask)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
