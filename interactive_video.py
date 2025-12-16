import tempfile
from pathlib import Path

import cv2
import numpy as np
import progressbar

from tools.annotations_prompts_types import AnnotationVideoInfo


class InteractVideo:
    def __init__(
        self, video_path: str, keyframe_interval: int = 3, max_points: int = 10
    ):
        self.video_path = video_path
        self.tmpdir = tempfile.TemporaryDirectory()
        self.frames_path: list[str] = []
        self.keypoints: dict[
            int, list[tuple[int, int]]
        ] = {}  # {frame_index: [(x1,y1), (x2,y2), ...]}
        self.keyframe_interval = keyframe_interval
        self.current_frame_idx = 0
        self.history: list[int] = []  # For tracking skipped frames
        self.max_points = max_points
        self.fps = 0.0
        self.count_frames = 0
        self.current_points: list[tuple[int, int]] = []

    def extract_frames(
        self, frames_to_propagate: int = 0, max_width: int = 1280, max_height: int = 720
    ):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {self.video_path}')

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frames_to_propagate <= 0 or frames_to_propagate > self.count_frames:
            frames_to_propagate = self.count_frames

        frame_index = 0

        original_width = int(cap.get(3))
        original_height = int(cap.get(4))
        self.frame_size = (original_width, original_height)

        print(f'Extracting frames from {self.video_path} into a temporary dir...')
        bar = progressbar.ProgressBar(max_value=frames_to_propagate)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index >= frames_to_propagate:
                break
            # Проверка и изменение размера кадра
            if max_width is not None or max_height is not None:
                h, w = frame.shape[:2]
                ratio_w = max_width / w if max_width else float('inf')
                ratio_h = max_height / h if max_height else float('inf')
                ratio = min(ratio_w, ratio_h, 1.0)  # Не увеличиваем изображение

                if ratio < 1.0:
                    new_size = (int(w * ratio), int(h * ratio))
                    frame = cv2.resize(frame, new_size)
                    # Обновляем размер кадра для первого кадра
                    if frame_index == 0:
                        self.frame_size = new_size
            frame_path = Path(self.tmpdir.name) / f'frame_{frame_index:06d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            self.frames_path.append(str(frame_path))
            frame_index += 1
            bar.update(frame_index)
        bar.finish()
        cap.release()

    def get_frames_path(self):
        return self.frames_path

    def collect_keypoints(self):
        """Собирает ключевые точки с поддержкой навигации"""
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.mouse_callback)

        saved_flag = False

        while 0 <= self.current_frame_idx < len(self.frames_path):
            frame = np.array(cv2.imread(self.frames_path[self.current_frame_idx]))
            is_keyframe = self.current_frame_idx % self.keyframe_interval == 0

            if is_keyframe:
                self.current_points = self.keypoints.get(
                    self.current_frame_idx, []
                ).copy()
                self.show_frame_with_controls(frame.copy())

                while True:
                    key = cv2.waitKey(100)

                    if key == ord('s'):  # Enter
                        self.keypoints[self.current_frame_idx] = (
                            self.current_points.copy()
                        )
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        saved_flag = True
                        break
                    elif key == ord('w'):  # empty
                        self.keypoints[self.current_frame_idx] = []
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        saved_flag = False
                        break
                    elif key == ord('d'):  # next
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        break
                    elif key == ord('a') and self.history:
                        self.current_frame_idx = self.history.pop()
                        break
                    elif key in (ord('q'), 27):
                        cv2.destroyAllWindows()
                        if saved_flag:
                            self.keypoints[len(self.frames_path) - 1] = []
                        return
            else:
                # Показываем обычные кадры без остановки
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(1)
                if key in [ord('q'), 27]:
                    if saved_flag:
                        self.keypoints[len(self.frames_path) - 1] = []
                    break
                self.current_frame_idx += 1

        cv2.destroyAllWindows()

    def show_frame_with_controls(self, frame):
        """Показывает кадр с элементами управления"""
        self.current_frame = frame
        h, w = self.current_frame.shape[:2]

        # Панель управления
        cv2.rectangle(self.current_frame, (0, 0), (w, 43), (0, 0, 0), -1)
        cv2.putText(
            self.current_frame,
            f'Frame {self.current_frame_idx} from {len(self.frames_path)}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.current_frame,
            '[S] Save [W] Empty [D] Next [A] Back [Q] Quit',
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Grid
        cv2.line(self.current_frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(self.current_frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # Points
        for x, y in self.current_points:
            cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Frame', self.current_frame)

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_frame_idx % self.keyframe_interval != 0:
            return
        if len(self.current_points) >= self.max_points:
            print('Point limit')
            return
        print(f'Frame {self.current_frame_idx}')
        self.current_points.append((x, y))
        print(f'Point added: ({x}, {y})')
        cv2.circle(self.current_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Frame', self.current_frame)

    def get_results(self) -> AnnotationVideoInfo:
        return {
            'frames_path': self.frames_path,
            'keypoints': self.keypoints,
        }


if __name__ == '__main__':
    controller = InteractVideo('video-test/video.mp4')
    controller.extract_frames()  # Сначала извлекаем все кадры
    controller.collect_keypoints()
    results = controller.get_results()
    print(f'Всего кадров: {len(results["frames_path"])}')
    for frame_idx, points in results['keypoints'].items():
        if points:
            print(f'Кадр {frame_idx}: {len(points)} точек')
        else:
            print(f'Пустой кадр {frame_idx}')

    select_masks = {}
    points_frames = []
    for frame_idx, points in results['keypoints'].items():
        if points:
            select_masks[frame_idx] = len(points)
        points_frames.append(int(frame_idx))
    points_frames.append(len(controller.frames))

    print(f'{len(select_masks)=}')
    print(f'{select_masks=}')
    print(f'{len(points_frames)=}')
    print(f'{points_frames}')

    frames_idx = list(results['keypoints'].keys())
    result = []
    for i in range(len(frames_idx) - 1):
        current_frame = frames_idx[i]
        current_coords = results['keypoints'][current_frame]

        next_frame = frames_idx[i + 1]
        result.append(
            {
                'gap': [current_frame, next_frame],
                'frame': current_frame,
                'coords': current_coords if current_coords else None,
            }
        )

    print(result)
