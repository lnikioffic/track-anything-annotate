import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import progressbar


class InteractVideo:
    def __init__(self, video_path: str, keyframe_interval: int = 3, max_points_per_frame: int = 10):
        self.video_path = video_path
        self.keyframe_interval = keyframe_interval
        self.max_points_per_frame = max_points_per_frame

        self.frames_filepaths: List[str] = []
        self.keypoints_per_frame: dict[int, List[Tuple[int, int]]] = {}
        self.current_frame_idx: int = 0
        self.current_points: List[Tuple[int, int]] = []
        self.history: List[int] = []

        self.fps: float = 0.0
        self.frame_size: Tuple[int, int] = (0, 0)
        self.count_frames: int = 0

        # Temporary directory для всех кадров
        self._tmpdir = tempfile.TemporaryDirectory()
        self._window_name = 'Frame'

        # GUI state
        self._key_pressed: int = -1
        self._stop_flag: bool = False

    # ----------------------------
    # Извлечение кадров
    # ----------------------------
    def extract_frames(self, frames_to_propagate: int = 0, max_width: int = 1280, max_height: int = 720):
        """Извлекает кадры из видео и сохраняет их во временную директорию."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {self.video_path}')

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_propagate = min(frames_to_propagate or self.count_frames, self.count_frames)

        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f'Extracting {frames_to_propagate} frames from {self.video_path}...')

        bar = progressbar.ProgressBar(max_value=frames_to_propagate)
        self.frames_filepaths.clear()

        for idx in range(frames_to_propagate):
            ret, frame = cap.read()
            if not ret:
                break

            # Изменяем размер кадра при необходимости
            h, w = frame.shape[:2]
            ratio_w = max_width / w if max_width else 1.0
            ratio_h = max_height / h if max_height else 1.0
            ratio = min(ratio_w, ratio_h, 1.0)

            if ratio < 1.0:
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
                if idx == 0:
                    self.frame_size = (frame.shape[1], frame.shape[0])

            frame_path = Path(self._tmpdir.name) / f'frame_{idx:06d}.jpg'
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f'Failed to write frame {idx} to {frame_path}')

            self.frames_filepaths.append(str(frame_path))
            bar.update(idx + 1)

        bar.finish()
        cap.release()
        print(f'Frames extracted: {len(self.frames_filepaths)}')

    # ----------------------------
    # Keypoints / GUI
    # ----------------------------
    def start_annotation(self):
        """Запуск интерактивного режима с событиями."""
        if not self.frames_filepaths:
            raise RuntimeError("No frames extracted. Call extract_frames first.")
    
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._on_mouse_click)
    
        self.current_frame_idx = 0
        self.history.clear()
    
        while self.current_frame_idx < len(self.frames_filepaths):
            frame_path = self.frames_filepaths[self.current_frame_idx]
            frame = cv2.imread(frame_path)
            if frame is None:
                self.current_frame_idx += 1
                continue
    
            # Загружаем точки текущего кадра, если есть
            self.current_points = self.keypoints_per_frame.get(self.current_frame_idx, []).copy()
    
            self._show_frame(frame)
            key = cv2.waitKey(0)
    
            if key == ord('s'):
                # Сохраняем только для keyframe
                if self.current_frame_idx % self.keyframe_interval == 0:
                    self.keypoints_per_frame[self.current_frame_idx] = self.current_points.copy()
                self.history.append(self.current_frame_idx)
                self.current_frame_idx += 1
            elif key == ord('w'):
                if self.current_frame_idx % self.keyframe_interval == 0:
                    self.keypoints_per_frame[self.current_frame_idx] = []
                self.history.append(self.current_frame_idx)
                self.current_frame_idx += 1
            elif key == ord('d'):
                self.history.append(self.current_frame_idx)
                self.current_frame_idx += 1
            elif key == ord('a') and self.history:
                self.current_frame_idx = self.history.pop()
            elif key in (ord('q'), 27):
                break
    
        cv2.destroyAllWindows()

    def _show_frame(self, frame: np.ndarray):
        """Отрисовка кадра с точками и GUI элементами."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 43), (0, 0, 0), -1)
        cv2.putText(frame, f'Frame {self.current_frame_idx+1}/{len(self.frames_filepaths)}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, '[S] Save [W] Empty [D] Next [A] Back [Q] Quit',
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Сетка
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        # Точки
        for x, y in self.current_points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow(self._window_name, frame)

    def _on_mouse_click(self, event, x, y, flags, param):
        """Добавление точки по клику мыши."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_frame_idx % self.keyframe_interval != 0:
            return
        if len(self.current_points) >= self.max_points_per_frame:
            print("Point limit reached")
            return
    
        # Сохраняем точку
        self.current_points.append((x, y))
        print(f'Frame {self.current_frame_idx}, point added: ({x}, {y})')
    
        # Перерисовываем кадр с новой точкой
        frame_path = self.frames_filepaths[self.current_frame_idx]
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Отрисовка всех точек текущего кадра
            for px, py in self.current_points:
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            # Добавляем GUI элементы (если нужно)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 43), (0, 0, 0), -1)
            cv2.putText(frame, f'Frame {self.current_frame_idx+1}/{len(self.frames_filepaths)}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, '[S] Save [W] Empty [D] Next [A] Back [Q] Quit',
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(self._window_name, frame)

    def get_results(self):
        return {
            'frames_path': self.frames_filepaths,
            'keypoints': self.keypoints_per_frame,
        }

if __name__ == '__main__':
    controller = InteractVideo('video-test/video.mp4')
    controller.extract_frames()  # Сначала извлекаем все кадры
    controller.start_annotation()
    results = controller.get_results()
    print(results['keypoints'])
    