import cv2
import numpy as np

from core.video_processor import VideoProcessor
from ui.cli.frame_viewer import FrameViewer


class InteractVideo:
    """
    Example:
        >>> video = InteractVideo('video.mp4')
        >>> video.collect_keypoints()
        >>> frames = video.get_frames()
        >>> keypoints = video.get_keypoints()
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
            if self.frame_viewer.current_frame is not None:
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
        frames_to_propagate: int | None = None,
    ) -> None:
        self.frames = self.video_processor.extract_all_frames(max_frames=frames_to_propagate)

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

                self.frame_viewer.show_frame(
                    frame.copy(),
                    points=self.current_points.copy(),
                    frame_index=self.current_frame_idx,
                    total_frames=len(self.frames),
                )

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
                    elif key == ord('a') and self.history:  # Back
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

    def get_frames(self) -> list[np.ndarray]:
        """Получение кадров."""
        return self.frames.copy()

    def get_keypoints(self) -> dict[int, list[tuple[int, int]]]:
        """Получение ключевых точек."""
        return self.keypoints_per_frame.copy()
