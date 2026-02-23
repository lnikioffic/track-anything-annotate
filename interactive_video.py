import cv2
import numpy as np
import progressbar

from core.video_processor import VideoProcessor
from tools.annotations_prompts_types import AnnotationVideoInfo
from ui.cli.frame_viewer import FrameViewer


class InteractVideo:
    """
    Интерактивное видео для сбора ключевых точек.

    Позволяет пользователю выбирать точки на ключевых кадрах
    для последующей сегментации и трекинга.

    Example:
        >>> video = InteractVideo('video.mp4')
        >>> video.collect_keypoints()
        >>> frames, keypoints = video.get_results()
    """

    def __init__(
        self,
        video_path: str,
        keyframe_interval: int = 3,
        max_points_per_frame: int = 15,
        max_width: int = 1280,
        max_height: int = 720,
    ):
        """
        Инициализация интерактивного видео.

        Args:
            video_path: Путь к видеофайлу.
            keyframe_interval: Интервал ключевых кадров.
            max_points_per_frame: Максимум точек на кадр.
            max_width: Максимальная ширина кадра.
            max_height: Максимальная высота кадра.
        """
        self.video_processor = VideoProcessor(video_path, max_width, max_height)
        self.keyframe_interval = keyframe_interval
        self.max_points_per_frame = max_points_per_frame

        self.frames: list[np.ndarray] = []
        self.keypoints_per_frame: dict[int, list[tuple[int, int]]] = {}
        self.current_frame_idx = 0
        self.current_points: list[tuple[int, int]] = []
        self.history: list[int] = []

        self.fps = 0.0
        self.count_frames = 0
        self.frame_size: tuple[int, int] = (0, 0)

        self.frame_viewer = FrameViewer()
        # Создаём окно перед установкой обработчика мыши
        cv2.namedWindow(self.frame_viewer.window_name)
        self._setup_mouse_callback()

    def _setup_mouse_callback(self) -> None:
        """Настройка обработчика мыши."""

        def mouse_callback(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if self.current_frame_idx % self.keyframe_interval != 0:
                return
            if len(self.current_points) >= self.max_points_per_frame:
                print(f'Point limit reached: {self.max_points_per_frame}')
                return

            print(f'Frame {self.current_frame_idx}')
            self.current_points.append((x, y))
            print(f'Point added: ({x}, {y})')

            # Отрисовка точки
            cv2.circle(
                self.frame_viewer.current_frame,
                (x, y),
                self.frame_viewer.POINT_RADIUS,
                self.frame_viewer.POINT_COLOR,
                -1,
            )
            cv2.imshow(self.frame_viewer.window_name, self.frame_viewer.current_frame)

        self.frame_viewer.set_mouse_callback(
            self.frame_viewer.window_name,
            mouse_callback,
        )

    def extract_frames(
        self,
        frames_to_propagate: int = 0,
        max_width: int = 1280,
        max_height: int = 720,
    ) -> None:
        """
        Извлечение кадров из видео.

        Args:
            frames_to_propagate: Количество кадров для обработки.
            max_width: Максимальная ширина.
            max_height: Максимальная высота.
        """
        cap = cv2.VideoCapture(self.video_processor.video_path)

        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {self.video_processor.video_path}')

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_to_propagate = min(
            frames_to_propagate or self.count_frames,
            self.count_frames,
        )

        self.frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        print(f'Extracting frames from {self.video_processor.video_path}...')

        bar = progressbar.ProgressBar(max_value=frames_to_propagate)
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index >= frames_to_propagate:
                break

            h, w = frame.shape[:2]
            ratio_w = max_width / w if max_width else 1.0
            ratio_h = max_height / h if max_height else 1.0
            ratio = min(ratio_w, ratio_h, 1.0)

            if ratio < 1.0:
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
                if frame_index == 0:
                    self.frame_size = (frame.shape[1], frame.shape[0])

            self.frames.append(frame)
            frame_index += 1
            bar.update(frame_index)

        bar.finish()
        cap.release()

        print(f'Extracted {len(self.frames)} frames')

    def collect_keypoints(self) -> None:
        """Сбор ключевых точек с поддержкой навигации."""
        if not self.frames:
            raise ValueError('No frames loaded. Call extract_frames() first.')

        saved_flag = False

        while 0 <= self.current_frame_idx < len(self.frames):
            frame = self.frames[self.current_frame_idx]
            is_keyframe = self.current_frame_idx % self.keyframe_interval == 0

            if is_keyframe:
                self.current_points = self.keypoints_per_frame.get(
                    self.current_frame_idx, []
                ).copy()
                self._show_frame_with_controls(frame.copy())

                while True:
                    key = cv2.waitKey(100)

                    # Обработка закрытия окна
                    if key == -1:
                        # Проверяем не закрыто ли окно
                        if (
                            cv2.getWindowProperty(
                                self.frame_viewer.window_name, cv2.WND_PROP_VISIBLE
                            )
                            < 1
                        ):
                            self.frame_viewer.close()
                            return

                    if key == ord('s'):  # Save
                        self.keypoints_per_frame[self.current_frame_idx] = (
                            self.current_points.copy()
                        )
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        saved_flag = True
                        break
                    elif key == ord('w'):  # Empty
                        self.keypoints_per_frame[self.current_frame_idx] = []
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        saved_flag = False
                        break
                    elif key == ord('d'):  # Next
                        self.history.append(self.current_frame_idx)
                        self.current_frame_idx += 1
                        break
                    elif key == ord('a') and self.history:
                        self.current_frame_idx = self.history.pop()
                        break
                    elif key in (ord('q'), 27):
                        self.frame_viewer.close()
                        if saved_flag:
                            self.keypoints_per_frame[len(self.frames) - 1] = []
                        return
            else:
                # Показываем обычные кадры без остановки
                cv2.imshow(self.frame_viewer.window_name, frame)
                key = cv2.waitKey(1)
                if key in [ord('q'), 27]:
                    if saved_flag:
                        self.keypoints_per_frame[len(self.frames) - 1] = []
                    break
                self.current_frame_idx += 1

        self.frame_viewer.close()

    def _show_frame_with_controls(self, frame: np.ndarray) -> None:
        """
        Показ кадра с элементами управления.

        Args:
            frame: Кадр для отображения.
        """
        self.frame_viewer.current_frame = frame
        h, w = self.frame_viewer.current_frame.shape[:2]

        # Панель управления
        cv2.rectangle(
            self.frame_viewer.current_frame,
            (0, 0),
            (w, 43),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            self.frame_viewer.current_frame,
            f'Frame {self.current_frame_idx} from {len(self.frames)}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.frame_viewer.current_frame,
            '[S] Save [W] Empty [D] Next [A] Back [Q] Quit',
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Grid
        cv2.line(
            self.frame_viewer.current_frame,
            (w // 2, 0),
            (w // 2, h),
            (0, 255, 0),
            1,
        )
        cv2.line(
            self.frame_viewer.current_frame,
            (0, h // 2),
            (w, h // 2),
            (0, 255, 0),
            1,
        )

        # Points
        for x, y in self.current_points:
            cv2.circle(
                self.frame_viewer.current_frame,
                (x, y),
                5,
                (0, 0, 255),
                -1,
            )

        cv2.imshow(self.frame_viewer.window_name, self.frame_viewer.current_frame)

    def get_results(self) -> AnnotationVideoInfo:
        """
        Получение результатов сбора.

        Returns:
            AnnotationVideoInfo: Информация о видео и ключевых точках.
        """
        return {
            'frames_path': [],  # Кадры в памяти, пути не нужны
            'keypoints': self.keypoints_per_frame,
        }

    def get_frames(self) -> list[np.ndarray]:
        """Получение кадров."""
        return self.frames.copy()

    def get_keypoints(self) -> dict[int, list[tuple[int, int]]]:
        """Получение ключевых точек."""
        return self.keypoints_per_frame.copy()
