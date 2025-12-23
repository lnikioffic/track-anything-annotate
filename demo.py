import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from config import DEVICE
from sam_controller import SamController
from tools.annotations_prompts_types import AnnotationInfo
from tracker import Tracker
from XMem2.inference.interact.interactive_utils import overlay_davis
from xmem2_tracker import TrackerCore


# --- Извлечение всех кадров ---
def extract_all_frames(video_input):
    video_path = video_input
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if len(frames) == 100 and DEVICE == 'cpu':
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print('read_frame_source:{} error. {}\n'.format(video_path, str(e)))
    video_state = {
        'fps': fps,
        'count_frames': count_frames,
    }
    tracker.set_image(frames[0])
    video_info = f'FPS: {video_state["fps"]} , Кадров: {video_state["count_frames"]}, Будет обработано: {len(frames)}'
    return frames[0], frames, video_state, video_info


# --- Ручная разметка точками (первый кадр) ---
def on_image_click(image, evt: gr.SelectData, annotations_state):
    x, y = evt.index[0], evt.index[1]
    annotations_state['point'].append([x, y])

    # Отрисовка всех точек
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for ann in annotations_state['point']:
        x_p, y_p = ann
        draw.ellipse((x_p - 5, y_p - 5, x_p + 5, y_p + 5), fill='blue')
    mask_info = f'Выбрано объектов: {len(annotations_state["point"])}, Координаты: {annotations_state["point"]}'
    return img, annotations_state, mask_info


# --- Разметка всех кадров ---
def tracking(frames: list[np.ndarray], video_state: dict):
    masks = tracker.track_objects(frames, video_state['mask'])
    video_state['annotation_masks'] = masks
    video_state['annotation_images'] = [
        overlay_davis(frame, mask) for frame, mask in zip(frames, masks)
    ]
    tracker.reset()
    annotation_info = f'Аннотированных кадров: {len(video_state["annotation_images"])}'
    return video_state, video_state['annotation_images'], annotation_info


# --- Аннотация ---
def annotations(
    frame: np.ndarray, annotations_state: dict, video_state: dict, mask_info
):
    if len(annotations_state['point']) == 0:
        mask_info = 'Поставьте точки на объекты'
        return frame, video_state, mask_info
    prompts = {
        'mode': 'point',
        'point_coords': annotations_state['point'],
        'point_labels': [1] * len(annotations_state['point']),
    }
    annotation_info = AnnotationInfo(
        class_name='object',
        prompt=prompts,
        count_objects=len(annotations_state['point']),
        order=0,
    )
    print(prompts)

    mask = tracker.segment_objects([annotation_info])
    tracker.reset()
    image = overlay_davis(frame, mask)
    video_state['mask'] = mask
    return image, video_state, mask_info


segmenter_controller = SamController()
tracker_core = TrackerCore()
tracker = Tracker(segmenter_controller, tracker_core)

# --- Интерфейс Gradio ---
with gr.Blocks() as demo:
    # Состояния
    frames = gr.State([])
    annotations_state = gr.State({'frame_id': 0, 'point': []})
    video_state = gr.State(
        {
            'fps': 30,
            'count_frames': 0,
            'mask': None,
            'annotation_masks': [],
            'annotation_images': [],
        }
    )

    gr.Markdown('# Трекинг объектов на видео')

    with gr.Row():
        video_input = gr.Video(label='Загрузите видео')

        with gr.Column():
            first_frame = gr.Image(label='Кадр для выбора объектов', interactive=True)
            with gr.Row():
                annotations_btn = gr.Button('Получить маску')

    with gr.Row():
        video_info = gr.Textbox(label='Информация о видео')
        mask_info = gr.Textbox(label='Информация разметке')

    with gr.Row():
        with gr.Row():
            annotation_info = gr.Textbox(label='Информация о трекинге')
            with gr.Column():
                tracking_btn = gr.Button('Трекинг')
        with gr.Column():
            annotated_gallery = gr.Gallery(label='Все кадры с разметкой', columns=3)

    video_input.change(
        extract_all_frames,
        inputs=video_input,
        outputs=[first_frame, frames, video_state, video_info],
    )

    # Обработка кликов
    first_frame.select(
        on_image_click,
        inputs=[first_frame, annotations_state],
        outputs=[first_frame, annotations_state, mask_info],
    )

    annotations_btn.click(
        annotations,
        inputs=[first_frame, annotations_state, video_state, mask_info],
        outputs=[first_frame, video_state, mask_info],
    )

    tracking_btn.click(
        tracking,
        inputs=[frames, video_state],
        outputs=[video_state, annotated_gallery, annotation_info],
    )

demo.launch(
    debug=True,
    share=True,
)
