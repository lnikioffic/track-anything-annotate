from typing import Any, Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from sam_controller import SamController
from tools.annotations_prompts_types import AnnotationInfo
from tracker import Tracker
from xmem2_tracker import TrackerCore


class GradioAnnotator:
    def __init__(self):
        self.segmenter_controller = SamController()
        self.tracker_core = TrackerCore()
        self.tracker = Tracker(self.segmenter_controller, self.tracker_core)
        self.frames: list[np.ndarray] = []
        self.video_state: dict[str, Any] = {}
        self.annotations_state: dict[str, Any] = {'frame_id': 0, 'points': []}

    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = 100,
    ) -> tuple[np.ndarray, list, dict, str]:
        """
        Returns:
            tuple: Первый кадр, список кадров, состояние видео, информация.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.frames = []
            frames_to_extract = min(max_frames or count_frames, count_frames)

            while cap.isOpened():
                if len(self.frames) >= frames_to_extract:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

            self.video_state = {
                'fps': fps,
                'count_frames': count_frames,
                'extracted_frames': len(self.frames),
            }

            if self.frames:
                self.tracker.set_image(self.frames[0])

            video_info = (
                f'FPS: {fps:.2f}, Кадров: {count_frames}, Будет обработано: {len(self.frames)}'
            )

            print(f'Extracted {len(self.frames)} frames from video')
            return (
                self.frames[0] if self.frames else None,
                self.frames,
                self.video_state,
                video_info,
            )

        except Exception as e:
            print(f'Error extracting frames: {e}')
            raise

    def on_image_click(
        self,
        image: np.ndarray,
        evt: gr.SelectData,
    ) -> tuple[Image.Image, dict, str]:
        """
        Обработка клика по изображению.

        Args:
            image: Изображение.
            evt: Событие выбора.

        Returns:
            tuple: Изображение с точками, состояние, информация.
        """
        x, y = evt.index[0], evt.index[1]
        self.annotations_state['points'].append([x, y])

        # Отрисовка всех точек
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        for point in self.annotations_state['points']:
            xp, yp = point
            draw.ellipse(
                (xp - 5, yp - 5, xp + 5, yp + 5),
                fill='blue',
                outline='white',
            )

        mask_info = (
            f'Выбрано объектов: {len(self.annotations_state["points"])}, '
            f'Координаты: {self.annotations_state["points"]}'
        )

        return img, self.annotations_state, mask_info

    def annotate(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, dict, str]:
        """
        Returns:
            tuple: Аннотированный кадр, состояние, информация.
        """
        if not self.annotations_state['points']:
            return frame, self.video_state, 'Поставьте точки на объекты'

        prompts = {
            'mode': 'point',
            'point_coords': self.annotations_state['points'],
            'point_labels': [1] * len(self.annotations_state['points']),
        }

        annotation_info = AnnotationInfo(
            class_name='object',
            prompt=prompts,
            count_objects=len(self.annotations_state['points']),
            order=0,
        )

        print(f'Annotating with {len(self.annotations_state["points"])} points')

        from XMem2.inference.interact.interactive_utils import overlay_davis

        mask = self.tracker.segment_objects([annotation_info])
        self.tracker.reset()

        self.video_state['mask'] = mask
        image = overlay_davis(frame, mask)

        return image, self.video_state, f'Маска создана: {mask.shape}'

    def track(self) -> tuple[dict, list, str]:
        """
        Трекинг по всем кадрам.

        Returns:
            tuple: Состояние, аннотированные кадры, информация.
        """
        from XMem2.inference.interact.interactive_utils import overlay_davis

        if 'mask' not in self.video_state:
            return self.video_state, [], 'Сначала создайте маску'

        masks = self.tracker.track_objects(self.frames, self.video_state['mask'])
        self.tracker.reset()

        annotated_images = [overlay_davis(frame, mask) for frame, mask in zip(self.frames, masks)]

        self.video_state['annotation_masks'] = masks
        self.video_state['annotation_images'] = annotated_images

        print(f'Tracking completed: {len(masks)} frames')
        return (
            self.video_state,
            annotated_images,
            f'Аннотированных кадров: {len(annotated_images)}',
        )


def create_app() -> gr.Blocks:
    """
    Создание Gradio приложения.

    Returns:
        gr.Blocks: Gradio приложение.
    """
    annotator = GradioAnnotator()

    with gr.Blocks(title='Video Annotation Tool') as demo:
        gr.Markdown('# Трекинг объектов на видео')

        # Состояния
        frames_state = gr.State([])
        annotations_state = gr.State({'frame_id': 0, 'points': []})
        video_state = gr.State({})

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label='Загрузите видео')
                extract_btn = gr.Button('Извлечь кадры')

            with gr.Column():
                first_frame = gr.Image(
                    label='Кадр для выбора объектов',
                    interactive=True,
                )
                annotate_btn = gr.Button('Получить маску')

        with gr.Row():
            video_info = gr.Textbox(label='Информация о видео')
            mask_info = gr.Textbox(label='Информация о разметке')

        with gr.Row():
            with gr.Column():
                annotation_info = gr.Textbox(label='Информация о трекинге')
                tracking_btn = gr.Button('Трекинг')

            annotated_gallery = gr.Gallery(
                label='Все кадры с разметкой',
                columns=4,
                height='auto',
            )

        # Обработчики
        extract_btn.click(
            fn=annotator.extract_frames,
            inputs=[video_input],
            outputs=[first_frame, frames_state, video_state, video_info],
        )

        first_frame.select(
            fn=annotator.on_image_click,
            inputs=[first_frame],
            outputs=[first_frame, annotations_state, mask_info],
        )

        annotate_btn.click(
            fn=annotator.annotate,
            inputs=[first_frame],
            outputs=[first_frame, video_state, mask_info],
        )

        tracking_btn.click(
            fn=annotator.track,
            inputs=[],
            outputs=[video_state, annotated_gallery, annotation_info],
        )

    return demo
